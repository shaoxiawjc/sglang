from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.mamba_radix_cache import (
    MambaRadixCache,
    TreeNode,
    get_last_access_time,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class MarconiForwardBoundary:
    req_index: int
    prefix_end: int
    local_start: int
    local_end: int
    mamba_index: int


@dataclasses.dataclass(frozen=True)
class MarconiForwardBoundaryBatch:
    boundaries: list[MarconiForwardBoundary]

    @property
    def boundaries_by_req(self) -> dict[int, list[tuple[int, int, int]]]:
        by_req: dict[int, list[tuple[int, int, int]]] = {}
        for boundary in self.boundaries:
            by_req.setdefault(int(boundary.req_index), []).append(
                (
                    int(boundary.local_start),
                    int(boundary.local_end),
                    int(boundary.mamba_index),
                )
            )
        for req_boundaries in by_req.values():
            req_boundaries.sort(key=lambda item: (item[0], item[1]))
        return by_req

    @property
    def req_indices(self) -> list[int]:
        return [boundary.req_index for boundary in self.boundaries]

    @property
    def block_ends(self) -> list[int]:
        return [boundary.prefix_end for boundary in self.boundaries]

    @property
    def local_starts(self) -> list[int]:
        return [boundary.local_start for boundary in self.boundaries]

    @property
    def local_ends(self) -> list[int]:
        return [boundary.local_end for boundary in self.boundaries]

    @property
    def mamba_indices(self) -> list[int]:
        return [boundary.mamba_index for boundary in self.boundaries]


class MarconiCache(MambaRadixCache):
    """Input-prefix Marconi cache for hybrid SSM models.

    Marconi keeps the full input KV path in the radix tree, but admits Mamba
    states only for speculative input branch points. Output-continuation states
    and FLOP-aware eviction are intentionally not implemented here.
    """

    def __init__(self, params):
        super().__init__(params)
        self._warned_no_extra_buffer = False
        logger.info("Initialized MarconiCache: input_only=True, eviction=lru")

    def _reset_cache_perf_counters(self) -> None:
        super()._reset_cache_perf_counters()
        self.total_marconi_branch_candidates = 0
        self.total_marconi_created_states = 0
        self.total_marconi_state_capture_failures = 0
        self.total_marconi_skipped_states = 0

    def reset(self) -> None:
        super().reset()
        self.root_node.marconi_prefix_tokens = 0

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = self._match_pre_processor(params)
        if key is None or self._disabled_for_marconi():
            return self._empty_match_result()

        req = params.req
        branch_len = self._find_input_branch_checkpoint(req, key) if req else None
        value, last_node, best_value_len = self._match_kv_path_to_mamba(key)
        result = self._match_post_processor(params, value, last_node, best_value_len)

        if req is not None:
            req._marconi_admission_seqlen = branch_len
            if branch_len is not None:
                self.total_marconi_branch_candidates += 1

        if params.log_stats:
            self._log_cache_stats(hit_tokens=len(result.device_indices))
        return result

    def cache_finished_req(self, req, is_insert: bool = True) -> None:
        try:
            kv_committed_len = req.pop_committed_kv_cache()
            if self.disable:
                self._free_req_kv(req, 0, kv_committed_len)
                self.req_to_token_pool.free_mamba_cache(req)
                return

            input_len = min(kv_committed_len, len(req.origin_input_ids))
            input_len = self._align_down(input_len)
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]

            cached_len = req.cache_protected_len
            new_last_node = req.last_node
            if is_insert and input_len > 0:
                _cached_len, new_last_node = self._cache_input_kv_path(
                    req=req,
                    token_ids=req.origin_input_ids[:input_len],
                    kv_indices=kv_indices[:input_len],
                    duplicate_free_from=req.cache_protected_len,
                )
                cached_len = max(cached_len, _cached_len)

            mamba_attached = self._attach_mamba_to_node(new_last_node, req)

            self._free_req_kv(req, max(cached_len, input_len), kv_committed_len)
            if not mamba_attached:
                self._free_finished_req_mamba(req)
            self.dec_lock_ref(req.last_node)
            req.last_node = new_last_node
        finally:
            self._release_unattached_marconi_slots(req)
            req._marconi_admission_seqlen = None

    def cache_unfinished_req(self, req, chunked: bool = False) -> None:
        del chunked
        try:
            token_ids = req.fill_ids
            if self.disable or not token_ids:
                return self._skip_cache_unfinished_req(req, len(token_ids))

            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            cached_len, new_last_node = self._cache_input_kv_path(
                req=req,
                token_ids=token_ids,
                kv_indices=kv_indices,
                duplicate_free_from=req.cache_protected_len,
            )
            if cached_len <= req.cache_protected_len:
                return self._skip_cache_unfinished_req(req, len(token_ids))

            canonical_indices = self._collect_prefix_indices(new_last_node)
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(req.cache_protected_len, cached_len)),
                canonical_indices[req.cache_protected_len:cached_len],
            )
            self.dec_lock_ref(req.last_node)
            self.inc_lock_ref(new_last_node)

            if cached_len < len(kv_indices):
                req.prefix_indices = torch.cat(
                    [canonical_indices, kv_indices[cached_len:]]
                )
            else:
                req.prefix_indices = canonical_indices
            req.cache_protected_len = cached_len
            req.last_node = new_last_node
        finally:
            self._release_unattached_marconi_slots(req)

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        result = super().evict(params)
        return result

    def get_cache_metrics(self) -> dict:
        metrics = super().get_cache_metrics()
        metrics.update(
            {
                "total_marconi_branch_candidates": int(
                    self.total_marconi_branch_candidates
                ),
                "total_marconi_created_states": int(
                    self.total_marconi_created_states
                ),
                "total_marconi_state_capture_failures": int(
                    self.total_marconi_state_capture_failures
                ),
                "total_marconi_skipped_states": int(
                    self.total_marconi_skipped_states
                ),
            }
        )
        return metrics

    def prepare_mamba_forward_boundaries(
        self,
        reqs: list,
        prefix_lens: list[int],
        extend_lens: list[int],
    ) -> Optional[MarconiForwardBoundaryBatch]:
        """Reuse the operator-boundary capture path for Marconi branch states."""
        for req in reqs:
            self._release_unattached_marconi_slots(req)
            req._marconi_boundary_mamba_indices_by_end = {}
            req._marconi_pending_boundary_mamba_indices = {}

        if self.disable or self._disabled_for_marconi():
            return None

        pending: list[tuple[object, int, int, int, int]] = []
        flat_offset = 0
        for req_index, (req, prefix_len, extend_len) in enumerate(
            zip(reqs, prefix_lens, extend_lens)
        ):
            target_len = getattr(req, "_marconi_admission_seqlen", None)
            prefix_len = int(prefix_len)
            extend_len = int(extend_len)
            extend_end = prefix_len + extend_len
            if (
                target_len is not None
                and prefix_len < int(target_len) <= extend_end
                and not self._has_mamba_at_prefix(req, int(target_len))
            ):
                local_end = flat_offset + int(target_len) - prefix_len
                pending.append(
                    (req, req_index, int(target_len), flat_offset, local_end)
                )
            flat_offset += extend_len

        if not pending:
            return None

        slots = self.req_to_token_pool.mamba_pool.alloc_uninitialized(len(pending))
        if slots is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=len(pending)))
            slots = self.req_to_token_pool.mamba_pool.alloc_uninitialized(len(pending))
        if slots is None:
            self.total_marconi_state_capture_failures += len(pending)
            return None

        slot_ids = [int(slot_id) for slot_id in slots.detach().cpu().tolist()]
        boundaries: list[MarconiForwardBoundary] = []
        for offset, (req, req_index, prefix_end, local_start, local_end) in enumerate(
            pending
        ):
            slot = slots[offset : offset + 1]
            req._marconi_boundary_mamba_indices_by_end[prefix_end] = slot
            req._marconi_pending_boundary_mamba_indices[prefix_end] = slot
            boundaries.append(
                MarconiForwardBoundary(
                    req_index=req_index,
                    prefix_end=prefix_end,
                    local_start=local_start,
                    local_end=local_end,
                    mamba_index=slot_ids[offset],
                )
            )

        return MarconiForwardBoundaryBatch(boundaries=boundaries)

    def prepare_rrmc_forward_boundaries(
        self,
        reqs: list,
        prefix_lens: list[int],
        extend_lens: list[int],
    ) -> Optional[MarconiForwardBoundaryBatch]:
        return self.prepare_mamba_forward_boundaries(reqs, prefix_lens, extend_lens)

    def disable_operator_chunk_state_tracking(self, req) -> bool:
        # Marconi captures admitted branch states explicitly through boundary
        # metadata, so the native every-chunk ping-pong tracking should not write
        # tree states for this request.
        return True

    def rrmc_disable_operator_chunk_state_tracking(self, req) -> bool:
        return self.disable_operator_chunk_state_tracking(req)

    def _disabled_for_marconi(self) -> bool:
        if not self.enable_mamba_extra_buffer:
            if not self._warned_no_extra_buffer:
                logger.warning(
                    "MarconiCache requires mamba extra_buffer for branch capture; "
                    "falling back to empty matches."
                )
                self._warned_no_extra_buffer = True
            return True
        if self.page_size <= 0:
            return True
        return False

    def _empty_match_result(self) -> MatchResult:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
        )

    def _align_down(self, length: int) -> int:
        if length <= 0:
            return 0
        return length // self.page_size * self.page_size

    def _state_align_down(self, length: int) -> int:
        server_args = get_global_server_args()
        align = max(
            1,
            int(getattr(server_args, "mamba_cache_chunk_size", 1)),
            int(self.page_size),
        )
        return length // align * align

    def _find_input_branch_checkpoint(self, req, key: RadixKey) -> Optional[int]:
        if req is None or len(key) == 0:
            return None

        node = self.root_node
        remaining = key
        matched_len = 0
        child_key = self.get_child_key_fn(remaining)

        while len(remaining) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, remaining)
            if prefix_len == 0:
                return None

            if prefix_len < len(child.key):
                return self._admissible_checkpoint_len(req, matched_len + prefix_len)

            matched_len += prefix_len
            node = child
            remaining = remaining[prefix_len:]
            if len(remaining):
                child_key = self.get_child_key_fn(remaining)

        if node is not self.root_node and len(remaining) > 0 and len(node.children) > 0:
            return self._admissible_checkpoint_len(req, matched_len)
        return None

    def _admissible_checkpoint_len(self, req, raw_branch_len: int) -> Optional[int]:
        checkpoint_len = self._state_align_down(raw_branch_len)
        if checkpoint_len <= 0:
            self.total_marconi_skipped_states += 1
            return None
        if checkpoint_len <= len(req.prefix_indices):
            return None
        if checkpoint_len >= len(req.origin_input_ids):
            return None
        if self._has_mamba_at_prefix(req, checkpoint_len):
            return None
        return checkpoint_len

    def _match_kv_path_to_mamba(
        self, key: RadixKey
    ) -> tuple[list[torch.Tensor], TreeNode, int]:
        node = self.root_node
        child_key = self.get_child_key_fn(key) if len(key) else None
        value: list[torch.Tensor] = []
        best_last_node = self.root_node
        best_value_len = 0

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len == 0:
                break
            if prefix_len < len(child.key):
                break

            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if node.mamba_value is not None:
                best_last_node = node
                best_value_len = len(value)
            if len(key):
                child_key = self.get_child_key_fn(key)

        return value, best_last_node, best_value_len

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[torch.Tensor],
        last_node: TreeNode,
        best_value_len: int,
    ) -> MatchResult:
        if params.log_stats and best_value_len > 0:
            self._mark_token_path_shared(last_node)

        if last_node is not self.root_node:
            self.full_lru_list.reset_node_and_parents_mru(last_node, self.root_node)
            self.mamba_lru_list.reset_node_and_parents_mru(last_node, self.root_node)
            cur_time = get_last_access_time()
            node_update = last_node
            while node_update is not None:
                node_update.last_access_time = cur_time
                cur_time -= 0.00001
                node_update = node_update.parent

        if params.cow_mamba and last_node.mamba_value is not None:
            req = params.req
            assert req is not None
            if req.mamba_pool_idx is None:
                dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                if dst_index is None:
                    self.inc_lock_ref(last_node)
                    self.evict(EvictParams(num_tokens=0, mamba_num=1))
                    dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                    self.dec_lock_ref(last_node)
                assert dst_index is not None, "Can not alloc mamba cache"
                req.mamba_pool_idx = dst_index[0]
            else:
                dst_index = req.mamba_pool_idx.unsqueeze(0)
            self.req_to_token_pool.mamba_pool.copy_from(last_node.mamba_value, dst_index)

        matched_values = value[:best_value_len]
        if matched_values:
            indices = torch.cat(matched_values)
        else:
            indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=indices,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def _cache_input_kv_path(
        self,
        req,
        token_ids: list[int],
        kv_indices: torch.Tensor,
        duplicate_free_from: int,
    ) -> tuple[int, TreeNode]:
        target_len = getattr(req, "_marconi_admission_seqlen", None)
        forced_splits = {int(target_len)} if target_len is not None else set()
        key = RadixKey(token_ids, req.extra_key)
        node = self._insert_kv_only(
            key=key,
            value=kv_indices,
            duplicate_free_from=duplicate_free_from,
            forced_splits=forced_splits,
        )
        if target_len is not None:
            target_node = self._lookup_node_by_prefix(
                req.origin_input_ids[: int(target_len)], req.extra_key
            )
            if target_node is not None:
                self._attach_captured_state(req, target_node, int(target_len))

        last_mamba_node = self._nearest_mamba_node(node)
        if last_mamba_node is None:
            return int(getattr(node, "marconi_prefix_tokens", 0)), node
        return int(getattr(last_mamba_node, "marconi_prefix_tokens", 0)), last_mamba_node

    def _insert_kv_only(
        self,
        key: RadixKey,
        value: torch.Tensor,
        duplicate_free_from: int,
        forced_splits: set[int],
    ) -> TreeNode:
        node = self.root_node
        total_prefix_len = 0
        child_key = self.get_child_key_fn(key) if len(key) else None

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            child.last_access_time = get_last_access_time()
            self.full_lru_list.reset_node_mru(child)
            if child.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(child)

            prefix_len = self.key_match_fn(child.key, key)
            split_len = self._split_len_for_forced_checkpoint(
                total_prefix_len=total_prefix_len,
                matched_len=prefix_len,
                child_len=len(child.key),
                forced_splits=forced_splits,
            )
            if split_len is not None:
                child = self._split_node(child.key, child, split_len)
                prefix_len = split_len

            if duplicate_free_from < total_prefix_len + prefix_len:
                start = max(0, duplicate_free_from - total_prefix_len)
                self.token_to_kv_pool_allocator.free(value[start:prefix_len])

            total_prefix_len += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            node = child

            if prefix_len < len(child.key):
                break
            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value.to(dtype=torch.int64, copy=True)
            new_node.mamba_value = None
            self._set_prefix_tokens(new_node, total_prefix_len)
            node.children[self.get_child_key_fn(key)] = new_node
            self.full_lru_list.insert_mru(new_node)
            self.full_evictable_size_ += len(new_node.value)
            self._on_token_node_created(new_node)
            node = new_node
        return node

    def _split_len_for_forced_checkpoint(
        self,
        *,
        total_prefix_len: int,
        matched_len: int,
        child_len: int,
        forced_splits: set[int],
    ) -> Optional[int]:
        if matched_len <= 0:
            return None
        for split_abs in sorted(forced_splits):
            split_len = split_abs - total_prefix_len
            if 0 < split_len < child_len and split_len <= matched_len:
                return split_len
        if matched_len < child_len:
            return matched_len
        return None

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        new_node = super()._split_node(key, child, split_len)
        parent_prefix = int(getattr(new_node.parent, "marconi_prefix_tokens", 0))
        self._set_prefix_tokens(new_node, parent_prefix)
        self._refresh_prefix_tokens(child)
        return new_node

    def _set_prefix_tokens(self, node: TreeNode, parent_prefix_len: int) -> None:
        node.marconi_prefix_tokens = parent_prefix_len + len(node.key)

    def _refresh_prefix_tokens(self, node: TreeNode) -> None:
        parent_prefix = int(getattr(node.parent, "marconi_prefix_tokens", 0))
        self._set_prefix_tokens(node, parent_prefix)
        for child in node.children.values():
            self._refresh_prefix_tokens(child)

    def _lookup_node_by_prefix(
        self, token_ids: list[int], extra_key: Optional[str]
    ) -> Optional[TreeNode]:
        key = RadixKey(token_ids, extra_key)
        node = self.root_node
        child_key = self.get_child_key_fn(key) if len(key) else None
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                return None
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)
        return node if len(key) == 0 else None

    def _has_mamba_at_prefix(self, req, prefix_len: int) -> bool:
        node = self._lookup_node_by_prefix(
            req.origin_input_ids[:prefix_len], req.extra_key
        )
        return node is not None and node.mamba_value is not None

    def _attach_captured_state(self, req, node: TreeNode, prefix_len: int) -> bool:
        if node is self.root_node or node.mamba_value is not None:
            return False
        slot = self._consume_marconi_boundary_slot(req, prefix_len)
        if slot is None:
            self.total_marconi_state_capture_failures += 1
            return False

        node.mamba_value = slot.to(dtype=torch.int64, copy=False)
        node.last_access_time = get_last_access_time()
        self.full_lru_list.reset_node_mru(node)
        self.mamba_lru_list.insert_mru(node)
        self.mamba_evictable_size_ += len(node.mamba_value)
        self._on_checkpoint_created(node)
        self._mark_marconi_boundary_slot_attached(req, prefix_len)
        self.total_marconi_created_states += 1
        return True

    def _attach_mamba_to_node(self, node: TreeNode, req) -> bool:
        if node is self.root_node or node.mamba_value is not None:
            return False
        if req.mamba_pool_idx is None:
            return False

        node.mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()
        node.last_access_time = get_last_access_time()
        self.full_lru_list.reset_node_mru(node)
        self.mamba_lru_list.insert_mru(node)
        if node.full_lock_ref > 0:
            self.mamba_protected_size_ += len(node.mamba_value)
            node.mamba_lock_ref += 1
        else:
            self.mamba_evictable_size_ += len(node.mamba_value)
        self._on_checkpoint_created(node)
        return True

    def _consume_marconi_boundary_slot(self, req, prefix_len: int) -> Optional[torch.Tensor]:
        boundary_map = getattr(req, "_marconi_boundary_mamba_indices_by_end", None)
        if not boundary_map:
            return None
        return boundary_map.get(int(prefix_len))

    def _mark_marconi_boundary_slot_attached(self, req, prefix_len: int) -> None:
        pending_map = getattr(req, "_marconi_pending_boundary_mamba_indices", None)
        if pending_map:
            pending_map.pop(int(prefix_len), None)

    def _release_unattached_marconi_slots(self, req) -> None:
        pending_map = getattr(req, "_marconi_pending_boundary_mamba_indices", None)
        if not pending_map:
            return
        pending_slots = list(pending_map.values())
        if pending_slots:
            self.req_to_token_pool.mamba_pool.free(torch.cat(pending_slots))
        pending_map.clear()

    def _collect_prefix_indices(self, last_node: TreeNode) -> torch.Tensor:
        values: list[torch.Tensor] = []
        node = last_node
        while node is not None and node is not self.root_node:
            values.append(node.value)
            node = node.parent
        if not values:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        values.reverse()
        return torch.cat(values)

    def _nearest_mamba_node(self, node: TreeNode) -> Optional[TreeNode]:
        while node is not None and node is not self.root_node:
            if node.mamba_value is not None:
                return node
            node = node.parent
        return None

    def _skip_cache_unfinished_req(self, req, total_tokens: int) -> None:
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :total_tokens
        ]
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)

    def _free_req_kv(self, req, start: int, end: int) -> None:
        if req.req_pool_idx is None or start >= end:
            return
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, start:end]
        self.token_to_kv_pool_allocator.free(kv_indices)

    def _free_finished_req_mamba(self, req) -> None:
        if req.mamba_pool_idx is not None:
            self.req_to_token_pool.free_mamba_cache(req)

    def _delete_leaf(self, node: TreeNode) -> None:
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = self.get_child_key_fn(node.key)
        removed = node.parent.children.pop(key, None)
        assert removed == node, f"parent does not have child key, {key}"
        self._on_token_node_evicted(node)
        self.full_evictable_size_ -= len(node.value)
        if node.mamba_value is not None:
            self.mamba_evictable_size_ -= len(node.mamba_value)

    def _evict_leaf_node(
        self, x: TreeNode, is_evict_mamba: bool
    ) -> tuple[int, int, TreeNode, TreeNode]:
        assert (
            x.full_lock_ref == 0 and x.mamba_lock_ref == 0
        ), f"evict leaf node invalid with {x.id=} {x.full_lock_ref=} {x.mamba_lock_ref=}"
        assert x.value is not None, f"leaf node has no KV value, {x.id=}"

        has_mamba = x.mamba_value is not None
        if has_mamba:
            self._on_checkpoint_evicted(x)
            self.req_to_token_pool.mamba_pool.free(x.mamba_value)
            mamba_num_evicted = len(x.mamba_value)
        else:
            mamba_num_evicted = 0

        self.token_to_kv_pool_allocator.free(x.value)
        full_num_evicted = len(x.value)

        if is_evict_mamba and has_mamba:
            x_next = self.mamba_lru_list.get_prev_no_lock(x)
        else:
            x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
        self.full_lru_list.remove_node(x)
        if has_mamba:
            self.mamba_lru_list.remove_node(x)

        self._delete_leaf(x)
        x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
        full_num_evicted += leaf_full_num_evicted
        return full_num_evicted, mamba_num_evicted, x, x_next

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        self._delete_leaf(node)

    def _tombstone_internal_node(self, node: TreeNode) -> None:
        assert node.mamba_value is not None
        self._on_checkpoint_evicted(node)
        self.mamba_evictable_size_ -= len(node.mamba_value)
        node.mamba_value = None
