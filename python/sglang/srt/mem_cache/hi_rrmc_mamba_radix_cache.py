from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.mamba_radix_cache import TreeNode, get_last_access_time
from sglang.srt.mem_cache.radix_cache import compute_node_hash_values
from sglang.srt.mem_cache.rrmc_mamba_radix_cache import (
    RRMC_RANKED_EVICTION_POLICIES,
    RRMCMambaRadixCache,
)

logger = logging.getLogger(__name__)


class HiRRMCMambaRadixCache(RRMCMambaRadixCache, HiMambaRadixCache):
    """Hierarchical RRMC cache with document-boundary Mamba states.

    This class does not tune KV/Mamba pool ratios. It keeps the existing SGLang
    command-line memory sizing behavior and only changes cache granularity:
    RRMC document segments are inserted as radix nodes, and the document-boundary
    Mamba state is backed up/restored through the same HiCache path as the KV.
    """

    def __init__(self, params, server_args):
        self._rrmc_write_back_node_ids: set[int] = set()
        HiMambaRadixCache.__init__(self, params=params, server_args=server_args)
        self._init_rrmc_runtime_fields(server_args)
        self._init_rrmc_eviction_policy(params, server_args)
        logger.info(
            "Initialized HiRRMCMambaRadixCache with eviction policy=%s, "
            "ours_alpha=%s, depth_lambda=%s, depth_efficient_alpha=%s, "
            "depth_efficient_beta=%s, segment_size=%s, admission=%s, "
            "admission_min_accesses=%s",
            self.rrmc_radix_eviction_policy,
            self.ours_evict_alpha,
            self.depth_aware_evict_lambda,
            self.depth_efficient_aware_alpha,
            self.depth_efficient_aware_beta,
            self.rrmc_segment_size,
            self.enable_rrmc_admission,
            self.rrmc_admission_min_accesses,
        )

    @staticmethod
    def _ensure_hicache_node_fields(node: TreeNode) -> None:
        if not hasattr(node, "hit_count"):
            node.hit_count = 0

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        self._ensure_hicache_node_fields(node)
        return super()._inc_hit_count(node, chunked=chunked)

    def _init_rrmc_runtime_fields(self, server_args) -> None:
        self.enable_rrmc_admission = bool(
            getattr(server_args, "enable_rrmc_admission", False)
        )
        self.rrmc_admission_min_accesses = max(
            1, int(getattr(server_args, "rrmc_admission_min_accesses", 2))
        )
        self.rrmc_admission_counts: dict[tuple, int] = {}
        configured_segment_size = getattr(server_args, "rrmc_segment_size", None)
        if configured_segment_size is None:
            configured_segment_size = (
                server_args.chunked_prefill_size
                or server_args.max_prefill_tokens
                or 2048
            )
        self.rrmc_segment_size = max(1, int(configured_segment_size))
        self._warned_page_size = False
        self._warned_bad_metadata = False
        self._rrmc_write_back_node_ids.clear()
        self._init_rrmc_ranked_eviction_heaps()

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "_rrmc_write_back_node_ids"):
            self._rrmc_write_back_node_ids.clear()
        if hasattr(self, "_rrmc_ranked_heaps"):
            self._init_rrmc_ranked_eviction_heaps()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        req = params.req
        if self.disable or req is None:
            return self._empty_match_result()

        segment_specs = self._get_cacheable_prefix_segments(req, len(params.key))
        if segment_specs is None or len(segment_specs) == 0:
            return self._empty_match_result()

        node = self.root_node
        matched_nodes: list[TreeNode] = []
        matched_device_values: list[torch.Tensor] = []
        best_last_node = self.root_node
        best_value_len = 0

        extra_key = req.extra_key
        for spec in segment_specs:
            tree_key = self._make_tree_key(node, extra_key, spec)
            child = node.children.get(tree_key)
            if child is None:
                break
            child_removed = self._repair_invalid_rrmc_boundary_state(child)
            if child_removed:
                break
            if child.evicted and not child.backuped:
                break

            matched_nodes.append(child)
            if not child.evicted:
                matched_device_values.append(child.value)
            node = child

            if self._rrmc_boundary_matchable(child):
                if self._rrmc_has_contiguous_kv_residency(matched_nodes):
                    best_last_node = child
                    best_value_len = len(matched_device_values)

        if best_last_node is self.root_node:
            return self._empty_match_result()

        best_node_index = matched_nodes.index(best_last_node)
        self._record_rrmc_path_access(matched_nodes[: best_node_index + 1])

        if params.log_stats:
            self._log_rrmc_stats(
                hit_blocks=sum(
                    1
                    for node in matched_nodes
                    if node.mamba_value is not None or node.mamba_backuped
                ),
                hit_tokens=int(getattr(best_last_node, "rrmc_prefix_tokens", 0)),
                evicted_blocks=0,
                evicted_tokens=0,
            )

        self._inc_hit_count(best_last_node, chunked=False)

        return self._match_post_processor(
            params=params,
            value=matched_device_values,
            last_node=best_last_node,
            best_value_len=best_value_len,
        )

    def _rrmc_has_contiguous_kv_residency(self, nodes: list[TreeNode]) -> bool:
        """Return whether matched KV can be represented by MatchResult.

        HiCache's scheduler represents a hit as device KV prefix plus an optional
        host-resident suffix. A host KV node followed by a device KV child cannot
        be encoded without dropping the middle host KV from prefix_indices, which
        corrupts request-to-token mapping. Treat that mixed pattern as a miss.
        """
        seen_host_kv = False
        for node in nodes:
            if node.value is not None:
                if seen_host_kv:
                    return False
                continue
            if node.host_value is not None:
                seen_host_kv = True
                continue
            return False
        return True

    def _rrmc_boundary_matchable(self, node: TreeNode) -> bool:
        if not getattr(node, "rrmc_is_block_end", False):
            return False
        if node.value is not None and node.mamba_value is not None:
            return True
        if node.evicted and node.backuped and node.mamba_backuped:
            return True
        return False

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[torch.Tensor],
        last_node: TreeNode,
        best_value_len: int,
    ) -> MatchResult:
        return HiMambaRadixCache._match_post_processor(
            self, params, value, last_node, best_value_len
        )

    def init_load_back(self, params: InitLoadBackParams):
        last_node = params.last_host_node
        if (
            last_node is not self.root_node
            and last_node.mamba_evicted
            and last_node.mamba_backuped
        ):
            if not last_node.evicted:
                return (
                    torch.empty((0,), dtype=torch.int64, device=self.device),
                    self.root_node,
                )
            loading_values = self.load_back(
                last_node,
                params.mem_quota,
                req=None,
            )
            if loading_values is not None:
                self._sync_pending_load_back()
                self._copy_restored_mamba_to_req(last_node, params.req)
                return loading_values, last_node
            if params.req is not None:
                params.req.prefix_indices = torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                )
                params.req.host_hit_length = 0
                params.req.cache_protected_len = 0
                params.req.last_node = self.root_node
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                self.root_node,
            )

        return HiMambaRadixCache.init_load_back(self, params)

    def _sync_pending_load_back(self) -> None:
        producer_id = self.cache_controller.start_loading()
        if producer_id < 0:
            return
        for _, finish_event, _ in self.cache_controller.ack_load_queue:
            finish_event.synchronize()
        self.loading_check()

    def _copy_restored_mamba_to_req(self, node: TreeNode, req) -> None:
        if req is None or node.mamba_value is None:
            return

        if req.mamba_pool_idx is None:
            dst_index = self._alloc_with_evict(
                self.req_to_token_pool.mamba_pool,
                len(node.mamba_value),
                self.evict_mamba,
                lock_node=node,
                error_message="Cannot alloc request mamba cache for restored HiRRMC hit",
            )
            req.mamba_pool_idx = dst_index[0]
        else:
            dst_index = req.mamba_pool_idx.unsqueeze(0)

        self.req_to_token_pool.mamba_pool.copy_from(node.mamba_value, dst_index)

    def _print_helper(self, node: TreeNode, indent: int) -> None:
        return RRMCMambaRadixCache._print_helper(self, node, indent)

    def write_backup(self, node: TreeNode, write_back=False):
        """Back up KV and any boundary Mamba state together for RRMC nodes."""
        if node.mamba_value is not None and node.mamba_host_value is not None:
            if self.mamba_host_lru_list.in_list(node):
                self.mamba_host_lru_list.reset_node_mru(node)

        if node.id in self.ongoing_write_through:
            self.writing_check(write_back=True)
        if write_back:
            self._rrmc_write_back_node_ids.add(node.id)
        extra_pools = self.mamba_backup_transfers(node)
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
            extra_pools=extra_pools,
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
                extra_pools=extra_pools,
            )
        if host_indices is None:
            if write_back:
                self._rrmc_write_back_node_ids.discard(node.id)
            return 0

        node.host_value = host_indices
        if extra_pools:
            self.mamba_backup_commit(node, extra_pools)
        assert len(node.host_value) > 0
        self.ongoing_write_through[node.id] = node
        if not write_back:
            self.inc_lock_ref(node)
        return len(host_indices)

    def _finish_write_ack(self, ack_id) -> None:
        backuped_node = self.ongoing_write_through.pop(ack_id, None)
        if backuped_node is None:
            self._rrmc_write_back_node_ids.discard(ack_id)
            return

        is_write_back = ack_id in self._rrmc_write_back_node_ids
        self._rrmc_write_back_node_ids.discard(ack_id)
        if not is_write_back:
            self.dec_lock_ref(backuped_node)
        if self.enable_storage and backuped_node.host_value is not None:
            self.write_backup_storage(backuped_node)

    def writing_check(self, write_back=False):
        if write_back:
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        self._finish_write_ack(ack_id)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        if len(self.ongoing_write_through) == 0:
            return

        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1

        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())

        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                self._finish_write_ack(ack_id)
            finish_count -= 1

    def _new_segment_node(
        self,
        parent: TreeNode,
        tree_key,
        segment,
        token_ids: list[int],
        kv_values: torch.Tensor,
        extra_key: Optional[str],
    ) -> TreeNode:
        node = super()._new_segment_node(
            parent=parent,
            tree_key=tree_key,
            segment=segment,
            token_ids=token_ids,
            kv_values=kv_values,
            extra_key=extra_key,
        )
        self._ensure_hicache_node_fields(node)
        if self.enable_storage:
            node.hash_value = compute_node_hash_values(node, self.page_size)
        self._update_leaf_status(node)
        self._update_leaf_status(parent)
        return node

    def _ensure_mamba_on_node(
        self,
        req,
        node: TreeNode,
        boundary_slot: Optional[torch.Tensor] = None,
    ) -> bool:
        created = super()._ensure_mamba_on_node(req, node, boundary_slot)
        if created:
            self._update_leaf_status(node)
            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(node, chunked=False)
        return created

    def _cache_segment_path(
        self,
        req,
        segment_specs,
        token_ids: list[int],
        kv_indices: torch.Tensor,
        duplicate_free_from: int,
        chunked: bool,
    ) -> tuple[int, TreeNode]:
        del chunked  # RRMC counts explicit matches, not token-level chunk hits.

        cache_until = self._rrmc_cache_prefix_limit(req, segment_specs)
        if cache_until <= 0:
            return req.cache_protected_len, self.root_node

        node = self.root_node
        extra_key = req.extra_key
        duplicate_ranges: list[tuple[int, int]] = []
        for segment in segment_specs:
            if segment.end > cache_until:
                break

            tree_key = self._make_tree_key(node, extra_key, segment)
            child = node.children.get(tree_key)
            if child is None:
                child = self._new_segment_node(
                    parent=node,
                    tree_key=tree_key,
                    segment=segment,
                    token_ids=token_ids[segment.start : segment.end],
                    kv_values=kv_indices[segment.start : segment.end],
                    extra_key=extra_key,
                )
            else:
                self._ensure_hicache_node_fields(child)
                child_removed = self._repair_invalid_rrmc_boundary_state(child)
                if child_removed:
                    child = self._new_segment_node(
                        parent=node,
                        tree_key=tree_key,
                        segment=segment,
                        token_ids=token_ids[segment.start : segment.end],
                        kv_values=kv_indices[segment.start : segment.end],
                        extra_key=extra_key,
                    )
                else:
                    child.last_access_time = get_last_access_time()
                    if child.evicted:
                        # Existing host-resident KV cannot be stitched into the
                        # current request's GPU prefix here. It must be restored
                        # by the scheduler load-back path after a prefix match.
                        self._update_leaf_status(child)
                        break
                    if self.full_lru_list.in_list(child):
                        self.full_lru_list.reset_node_mru(child)
                    if child.mamba_value is not None and self.mamba_lru_list.in_list(
                        child
                    ):
                        self.mamba_lru_list.reset_node_mru(child)
                    self._rrmc_mark_node_updated(child)
                    if segment.end > duplicate_free_from:
                        duplicate_ranges.append((segment.start, segment.end))

            node = child
            if segment.is_block_end:
                if self._was_rrmc_boundary_admission_skipped(req, segment.end):
                    continue
                boundary_slot = self._consume_rrmc_boundary_slot(req, segment.end)
                self._ensure_mamba_on_node(req, child, boundary_slot)

        self._free_duplicate_ranges(kv_indices, duplicate_ranges, duplicate_free_from)
        last_mamba_node = self._nearest_device_mamba_node(node)
        if last_mamba_node is None:
            return req.cache_protected_len, self.root_node
        self._update_leaf_status(last_mamba_node)
        return int(getattr(last_mamba_node, "rrmc_prefix_tokens", 0)), last_mamba_node

    def _nearest_device_mamba_node(self, node: TreeNode) -> Optional[TreeNode]:
        while node is not None and node is not self.root_node:
            if node.mamba_value is not None and node.value is not None:
                return node
            node = node.parent
        return None

    def _count_cached_blocks(self) -> int:
        return sum(
            1
            for node in self._collect_rrmc_nodes(include_evicted=True)
            if node is not self.root_node
            and getattr(node, "is_checkpoint_state", False)
            and (node.mamba_value is not None or node.mamba_backuped)
        )

    def _collect_rrmc_nodes(self, include_evicted: bool = False) -> list[TreeNode]:
        ret: list[TreeNode] = []
        stack = [self.root_node]
        while stack:
            cur = stack.pop()
            if include_evicted or not cur.evicted:
                ret.append(cur)
            stack.extend(cur.children.values())
        return ret

    def _is_full_device_evictable_node(self, node: TreeNode) -> bool:
        return (
            node.value is not None
            and node.full_lock_ref == 0
            and node.mamba_lock_ref == 0
            and node in self.evictable_full_device_leaves
        )

    def full_evictable_size(self) -> int:
        return sum(
            len(node.value)
            for node in self.full_lru_list.cache.values()
            if node.value is not None and node.full_lock_ref == 0
        )

    def mamba_evictable_size(self) -> int:
        return sum(
            len(node.mamba_value)
            for node in self.mamba_lru_list.cache.values()
            if node.mamba_value is not None and node.mamba_lock_ref == 0
        )

    def full_protected_size(self) -> int:
        return self.full_protected_size_

    def mamba_protected_size(self) -> int:
        return self.mamba_protected_size_

    def available_and_evictable_str(self) -> str:
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = self.full_evictable_size()
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} "
            f"({full_available_size=} + cascade_{full_evictable_size=})\n"
            f"Full residency-leaf evictable nodes: "
            f"{len(self.evictable_full_device_leaves)}\n"
            f"Policy-protected full tokens: {self.full_protected_size_}\n"
        )

    def _remove_rrmc_child_from_parent(self, node: TreeNode) -> None:
        key = self._node_tree_key(node)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have RRMC child key, {key}"

    def _free_device_mamba_for_node(self, node: TreeNode) -> int:
        if node.mamba_value is None:
            return 0
        mamba_num = len(node.mamba_value)
        self.req_to_token_pool.mamba_pool.free(node.mamba_value)
        if node.mamba_lock_ref > 0:
            self.mamba_protected_size_ -= mamba_num
            node.mamba_lock_ref = 0
        else:
            self.mamba_evictable_size_ -= mamba_num
        if self.mamba_lru_list.in_list(node):
            self.mamba_lru_list.remove_node(node)
        self._on_checkpoint_evicted(node)
        node.mamba_value = None
        self._rrmc_invalidate_node(node)
        self._rrmc_mark_node_updated(node)
        return mamba_num

    def _free_host_mamba_for_node(self, node: TreeNode) -> int:
        if node.mamba_host_value is None:
            return 0
        if self.mamba_host_lru_list.in_list(node):
            self.mamba_host_lru_list.remove_node(node)
        count = len(node.mamba_host_value)
        self.mamba_pool_host.free(node.mamba_host_value)
        node.mamba_host_value = None
        self._rrmc_invalidate_node(node)
        self._rrmc_mark_node_updated(node)
        return count

    def _backup_mamba_to_host(self, node: TreeNode) -> int:
        if node.mamba_value is None:
            return 0
        if node.mamba_host_value is not None:
            if self.mamba_host_lru_list.in_list(node):
                self.mamba_host_lru_list.reset_node_mru(node)
                self._rrmc_mark_node_updated(node)
            return len(node.mamba_host_value)
        if node.id in self.ongoing_write_through:
            self.writing_check(write_back=True)

        transfers = self.mamba_backup_transfers(node)
        if not transfers:
            return 0

        self._rrmc_write_back_node_ids.add(node.id)
        empty_kv_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        host_indices = self.cache_controller.write(
            device_indices=empty_kv_indices,
            node_id=node.id,
            extra_pools=transfers,
        )
        if host_indices is None:
            self._rrmc_write_back_node_ids.discard(node.id)
            return 0

        self.mamba_backup_commit(node, transfers)
        self.ongoing_write_through[node.id] = node
        self.writing_check(write_back=True)
        self._rrmc_mark_node_updated(node)
        return len(node.mamba_host_value) if node.mamba_host_value is not None else 0

    def _has_kv_residency(self, node: TreeNode) -> bool:
        return node.value is not None or node.host_value is not None

    def _has_mamba_residency(self, node: TreeNode) -> bool:
        return node.mamba_value is not None or node.mamba_host_value is not None

    def _can_drop_leaf(self, node: TreeNode) -> bool:
        return (
            node is not self.root_node
            and len(node.children) == 0
            and node.full_lock_ref == 0
            and node.mamba_lock_ref == 0
            and node.host_ref_counter == 0
        )

    def _drop_rrmc_leaf(self, node: TreeNode) -> bool:
        if not self._can_drop_leaf(node):
            return False

        if node.value is not None:
            self.cache_controller.evict_device(node.value)
            self.full_evictable_size_ -= len(node.value)
            if self.full_lru_list.in_list(node):
                self.full_lru_list.remove_node(node)
            self._on_token_node_evicted(node)
            node.value = None

        if node.host_value is not None:
            self.cache_controller.evict_host(node.host_value)
            node.host_value = None

        self._free_device_mamba_for_node(node)
        self._free_host_mamba_for_node(node)
        self._discard_from_leaf_sets(node)
        self._remove_rrmc_child_from_parent(node)
        self._update_leaf_status(node.parent)
        self._rrmc_invalidate_node(node)
        self._rrmc_mark_node_updated(node.parent)
        return True

    def _repair_invalid_rrmc_boundary_state(self, node: TreeNode) -> bool:
        """Drop or degrade RRMC boundary entries with only one cache component.

        A document-boundary hit is only executable when both attention KV and the
        recurrent Mamba state exist somewhere in the hierarchy. KV-only internal
        nodes may still be useful as a path to a later valid boundary, so those
        are degraded by not selecting them. Leaf KV-only entries are removed.
        Mamba-only entries are never useful without KV; clear their Mamba state
        and remove the leaf when possible.
        """
        if node is self.root_node or not getattr(node, "rrmc_is_block_end", False):
            return False

        has_kv = self._has_kv_residency(node)
        has_mamba = self._has_mamba_residency(node)
        if has_kv and has_mamba and self._rrmc_boundary_matchable(node):
            return False

        if has_mamba and not has_kv:
            self._free_device_mamba_for_node(node)
            self._free_host_mamba_for_node(node)
            if self._drop_rrmc_leaf(node):
                return True
            self._update_leaf_status(node)
            return False

        if has_kv and not has_mamba:
            if self._drop_rrmc_leaf(node):
                return True
            return False

        if self._drop_rrmc_leaf(node):
            return True
        return False

    def _update_full_host_leaf_status(self, node: TreeNode):
        if node.mamba_lock_ref > 0 or len(node.children) > 0:
            self.evictable_full_host_leaves.discard(node)
            return
        return super()._update_full_host_leaf_status(node)

    def _update_full_device_leaf_status(self, node: TreeNode):
        if node.mamba_lock_ref > 0:
            self.evictable_full_device_leaves.discard(node)
            return
        return super()._update_full_device_leaf_status(node)

    def _evict_device_leaf(self, node: TreeNode) -> tuple[int, int]:
        """Offload a device residency leaf, keeping KV and Mamba together."""
        if not self._is_full_device_evictable_node(node):
            self.evictable_full_device_leaves.discard(node)
            return 0, 0

        if not node.backuped:
            written = self.write_backup(node, write_back=True)
            if written <= 0:
                return 0, 0
            self.writing_check(write_back=True)
        elif node.mamba_value is not None and node.mamba_host_value is None:
            backed_up = self._backup_mamba_to_host(node)
            if backed_up <= 0:
                return 0, 0
        return self._evict_to_host(node)

    def _evict_to_host(self, node: TreeNode) -> tuple[int, int]:
        assert not node.evicted, f"already evicted, {node.id=}"
        assert node.backuped, f"not backuped, {node.id=}"

        num_full = len(node.value)
        self.cache_controller.evict_device(node.value)
        self.full_evictable_size_ -= num_full
        if self.full_lru_list.in_list(node):
            self.full_lru_list.remove_node(node)
        node.value = None
        mamba_num = self._free_device_mamba_for_node(node)

        self._update_leaf_status(node)
        self._update_full_device_leaf_status(node.parent)
        self._rrmc_invalidate_node(node)
        self._rrmc_mark_node_updated(node)
        self._rrmc_mark_node_updated(node.parent)
        return num_full, mamba_num

    def _evict_full_ranked(self, full_num_tokens: int) -> int:
        if self.disable or full_num_tokens <= 0:
            return 0

        full_num_evicted = 0
        skipped_ids: set[int] = set()
        while full_num_evicted < full_num_tokens:
            x = self._pop_rrmc_ranked_candidate("device_full")
            if x is None:
                break
            if x.id in skipped_ids:
                continue

            evicted_full, evicted_mamba = self._evict_device_leaf(x)
            if evicted_full <= 0 and evicted_mamba <= 0:
                skipped_ids.add(x.id)
                continue
            full_num_evicted += evicted_full
            self._ranked_full_mamba_evicted = (
                getattr(self, "_ranked_full_mamba_evicted", 0) + evicted_mamba
            )
            skipped_ids.clear()

        return full_num_evicted

    def _evict_mamba_ranked(self, mamba_num: int) -> int:
        if self.disable or mamba_num <= 0:
            return 0

        mamba_num_evicted = 0
        skipped_ids: set[int] = set()
        while mamba_num_evicted < mamba_num:
            x = self._pop_rrmc_ranked_candidate("device_mamba")
            if x is None:
                break
            if x.id in skipped_ids:
                continue

            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}"
            assert x != self.root_node, f"root node is not evictable, {x.id=}"
            assert x.mamba_lock_ref == 0, f"node is in use, {x.id=}"

            if len(x.children) > 0:
                mamba_num_evicted += self._free_device_mamba_for_node(x)
            else:
                evicted_full, evicted_mamba = self._evict_device_leaf(x)
                if evicted_full <= 0 and evicted_mamba <= 0:
                    skipped_ids.add(x.id)
                    continue
                mamba_num_evicted += evicted_mamba
            skipped_ids.clear()
        return mamba_num_evicted

    def evict_mamba(self, mamba_num: int) -> int:
        if self.rrmc_radix_eviction_policy in RRMC_RANKED_EVICTION_POLICIES:
            return self._evict_mamba_ranked(mamba_num)
        if self.disable or mamba_num <= 0:
            return 0

        x = self.mamba_lru_list.get_lru_no_lock()
        mamba_num_evicted = 0
        while mamba_num_evicted < mamba_num and self.mamba_lru_list.in_list(x):
            assert x is not None
            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}"
            assert x != self.root_node, f"root node is not evictable, {x.id=}"
            assert x.mamba_lock_ref == 0, f"node is in use, {x.id=}"

            x_next = self.mamba_lru_list.get_prev_no_lock(x)
            if len(x.children) > 0:
                mamba_num_evicted += self._free_device_mamba_for_node(x)
            else:
                if not self._is_full_device_evictable_node(x):
                    x = x_next
                    continue
                evicted_full, evicted_mamba = self._evict_device_leaf(x)
                if evicted_full <= 0 and evicted_mamba <= 0:
                    x = x_next
                    continue
                mamba_num_evicted += evicted_mamba

            if not self.mamba_lru_list.in_list(x_next):
                x_next = self.mamba_lru_list.get_lru_no_lock()
            x = x_next

        return mamba_num_evicted

    def _evict_mamba_host_ranked(self, num_mamba_hosts: int) -> int:
        if self.disable or num_mamba_hosts <= 0:
            return 0

        num_evicted = 0
        skipped_ids: set[int] = set()
        while num_evicted < num_mamba_hosts:
            x = self._pop_rrmc_ranked_candidate("host_mamba")
            if x is None:
                break
            if x.id in skipped_ids:
                continue

            if len(x.children) == 0 and x in self.evictable_full_host_leaves:
                if self._evict_host_leaf(x) > 0:
                    num_evicted += 1
                    skipped_ids.clear()
                else:
                    skipped_ids.add(x.id)
            else:
                if self._free_host_mamba_for_node(x) > 0:
                    num_evicted += 1
                    skipped_ids.clear()
                else:
                    skipped_ids.add(x.id)
        return num_evicted

    def evict_mamba_host(self, num_mamba_hosts: int) -> int:
        if self.rrmc_radix_eviction_policy in RRMC_RANKED_EVICTION_POLICIES:
            return self._evict_mamba_host_ranked(num_mamba_hosts)
        if self.disable or num_mamba_hosts <= 0:
            return 0

        x = self.mamba_host_lru_list.get_lru_no_lock()
        num_evicted = 0
        while num_evicted < num_mamba_hosts and self.mamba_host_lru_list.in_list(x):
            x_next = self.mamba_host_lru_list.get_prev_no_lock(x)
            if x.host_ref_counter > 0:
                x = x_next
                continue

            if len(x.children) == 0 and x in self.evictable_full_host_leaves:
                if self._evict_host_leaf(x) > 0:
                    num_evicted += 1
            else:
                if self._free_host_mamba_for_node(x) > 0:
                    num_evicted += 1

            if not self.mamba_host_lru_list.in_list(x_next):
                x_next = self.mamba_host_lru_list.get_lru_no_lock()
            x = x_next
        return num_evicted

    def _evict_host_full_ranked(self, num_tokens: int) -> int:
        if self.disable or num_tokens <= 0:
            return 0

        num_evicted = 0
        skipped_ids: set[int] = set()
        while num_evicted < num_tokens:
            x = self._pop_rrmc_ranked_candidate("host_full")
            if x is None:
                break
            if x.id in skipped_ids:
                continue
            evicted = self._evict_host_leaf(x)
            if evicted <= 0:
                skipped_ids.add(x.id)
                continue
            num_evicted += evicted
            skipped_ids.clear()
        return num_evicted

    def evict_host(self, num_tokens: int):
        if self.rrmc_radix_eviction_policy in RRMC_RANKED_EVICTION_POLICIES:
            self._evict_host_full_ranked(num_tokens)
            return
        return super().evict_host(num_tokens)

    def _evict_regular(self, node: TreeNode) -> tuple[int, int]:
        assert not node.evicted, f"already evicted, {node.id=}"
        assert not node.backuped, f"backuped node, {node.id=}"
        assert len(node.children) == 0, f"non-leaf, {node.id=}"

        full_num_evicted = len(node.value)
        self.cache_controller.evict_device(node.value)
        self.full_evictable_size_ -= full_num_evicted
        if self.full_lru_list.in_list(node):
            self.full_lru_list.remove_node(node)
        self._on_token_node_evicted(node)

        mamba_num_evicted = self._free_device_mamba_for_node(node)
        self._free_host_mamba_for_node(node)

        node.value = None
        self._discard_from_leaf_sets(node)
        self._remove_rrmc_child_from_parent(node)

        parent = node.parent
        self._rrmc_invalidate_node(node)
        self._update_leaf_status(parent)
        _, cascade_full_num_evicted, cascade_mamba_num_evicted = (
            self._iteratively_delete_tombstone_leaf(node)
        )
        self._rrmc_mark_node_updated(parent)
        return (
            full_num_evicted + cascade_full_num_evicted,
            mamba_num_evicted + cascade_mamba_num_evicted,
        )

    def _evict_host_leaf(self, node: TreeNode) -> int:
        assert node.evicted, f"not evicted, {node.id=}"
        assert node.backuped, f"not backuped, {node.id=}"
        assert node.host_ref_counter == 0, (
            f"in use, {node.id=} {node.host_ref_counter=}"
        )
        if len(node.children) > 0:
            self.evictable_full_host_leaves.discard(node)
            return 0

        full_num_evicted = self.cache_controller.evict_host(node.host_value)
        node.host_value = None
        self._free_device_mamba_for_node(node)
        self._free_host_mamba_for_node(node)

        self._discard_from_leaf_sets(node)
        self._remove_rrmc_child_from_parent(node)

        parent = node.parent
        self._rrmc_invalidate_node(node)
        self._update_leaf_status(parent)
        _, cascade_full_num_evicted, _ = self._iteratively_delete_tombstone_leaf(node)
        self._rrmc_mark_node_updated(parent)

        return full_num_evicted + cascade_full_num_evicted

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert node.mamba_value is None, f"has device mamba, {node.id=}"
        assert node.mamba_host_value is None, f"has host mamba, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"

        self._discard_from_leaf_sets(node)
        if node.backuped and node.host_ref_counter == 0:
            self.cache_controller.evict_host(node.host_value)
            node.host_value = None

        self._remove_rrmc_child_from_parent(node)
        self._update_leaf_status(node.parent)
        self._rrmc_invalidate_node(node)
        self._rrmc_mark_node_updated(node.parent)

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> tuple[TreeNode, int, int]:
        full_num_evicted = 0
        mamba_num_evicted = 0

        while len(node.parent.children) == 0:
            if node.parent == self.root_node:
                break
            if node.parent.mamba_value is not None:
                break
            if node.parent.mamba_host_value is not None:
                break
            if node.parent.full_lock_ref > 0 or node.parent.mamba_lock_ref > 0:
                break

            parent = node.parent
            if not parent.evicted:
                full_num_evicted += len(parent.value)
                self.full_evictable_size_ -= len(parent.value)
                self.cache_controller.evict_device(parent.value)
                if self.full_lru_list.in_list(parent):
                    self.full_lru_list.remove_node(parent)
                self._on_token_node_evicted(parent)

            self._discard_from_leaf_sets(parent)
            self._delete_tombstone_leaf(parent)
            self._rrmc_invalidate_node(parent)
            node = parent

        return node, full_num_evicted, mamba_num_evicted

    def _empty_match_result(self) -> MatchResult:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            host_hit_length=0,
        )
