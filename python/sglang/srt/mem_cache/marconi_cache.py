from __future__ import annotations

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


class MarconiCache(MambaRadixCache):
    """Input-only Marconi-style prefix cache for hybrid SSM models.

    This implements Marconi's input-prefix admission idea while deliberately
    omitting output-continuation checkpoints and FLOP-aware eviction:
    - KV is cached for token radix paths.
    - Mamba states are admitted only at detected input branch points.
    - A match is executable only at a node that has both contiguous KV and an
      admitted Mamba state.
    """

    def __init__(self, params):
        super().__init__(params)
        self._warned_page_size = False
        self._warned_no_extra_buffer = False
        logger.info(
            "Initialized MarconiCache: input_only=True, selected_mamba=True, eviction=lru"
        )

    def _reset_cache_perf_counters(self) -> None:
        super()._reset_cache_perf_counters()
        self.total_marconi_branch_candidates = 0
        self.total_marconi_created_states = 0
        self.total_marconi_state_capture_failures = 0
        self.total_marconi_skipped_states = 0

    def reset(self) -> None:
        super().reset()
        self.root_node.marconi_prefix_tokens = 0
        self.root_node.marconi_branch_admitted = False

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = self._match_pre_processor(params)
        if key is None or self._disabled_for_marconi():
            return self._empty_match_result()

        branch_len = (
            self._find_input_branch_checkpoint(params.req, key)
            if params.req is not None
            else None
        )
        value, last_node, best_value_len = self._match_prefix_helper(key)
        result = self._match_post_processor(params, value, last_node, best_value_len)
        if branch_len is not None:
            result = result._replace(mamba_branching_seqlen=branch_len)

        if params.req is not None:
            if branch_len is not None:
                params.req.mamba_branching_seqlen = branch_len
                self.total_marconi_branch_candidates += 1
            else:
                params.req.mamba_branching_seqlen = None

        if params.log_stats:
            self._log_cache_stats(hit_tokens=len(result.device_indices))
        return result

    def cache_finished_req(self, req, is_insert: bool = True) -> None:
        try:
            kv_committed_len = req.pop_committed_kv_cache()
            if self.disable:
                kv_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :kv_committed_len
                ]
                self.token_to_kv_pool_allocator.free(kv_indices)
                self.req_to_token_pool.free_mamba_cache(req)
                return
            if self._disabled_for_marconi():
                kv_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :kv_committed_len
                ]
                self.token_to_kv_pool_allocator.free(kv_indices)
                self._free_finished_req_mamba(req)
                self.dec_lock_ref(req.last_node)
                return

            token_ids = req.origin_input_ids[
                : min(kv_committed_len, len(req.origin_input_ids))
            ]
            kv_indices_all = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            kv_indices = kv_indices_all[: len(token_ids)]

            cached_len = req.cache_protected_len
            new_last_node = req.last_node
            if is_insert and token_ids:
                cached_len, new_last_node = self._cache_kv_path(
                    req=req,
                    token_ids=token_ids,
                    kv_indices=kv_indices,
                    duplicate_free_from=req.cache_protected_len,
                    require_mamba=True,
                )

            free_from = max(req.cache_protected_len, cached_len)
            if free_from < kv_committed_len:
                self.token_to_kv_pool_allocator.free(kv_indices_all[free_from:])
            self._free_finished_req_mamba(req)
            self.dec_lock_ref(req.last_node)
        finally:
            req.mamba_branching_seqlen = None

    def cache_unfinished_req(self, req, chunked: bool = False) -> None:
        del chunked
        token_ids = req.fill_ids
        if self.disable or self._disabled_for_marconi() or not token_ids:
            return self._skip_cache_unfinished_req(req, len(token_ids))

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        canonical_len, new_last_node = self._cache_kv_path(
            req=req,
            token_ids=token_ids,
            kv_indices=kv_indices_orig,
            duplicate_free_from=req.cache_protected_len,
            require_mamba=False,
        )

        if canonical_len <= req.cache_protected_len:
            return self._skip_cache_unfinished_req(req, len(token_ids))

        canonical_indices = self._collect_prefix_indices(new_last_node)
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, canonical_len)),
            canonical_indices[req.cache_protected_len:canonical_len],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        if canonical_len < len(kv_indices_orig):
            req.prefix_indices = torch.cat(
                [canonical_indices, kv_indices_orig[canonical_len:]]
            )
        else:
            req.prefix_indices = canonical_indices

        req.cache_protected_len = canonical_len
        req.mamba_last_track_seqlen = None
        req.mamba_branching_seqlen = None
        req.last_node = new_last_node

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        full_num_evicted = 0
        mamba_num_evicted = 0
        if params.num_tokens > 0:
            full_num_evicted = self.evict_full(params.num_tokens)
        if params.mamba_num > 0:
            mamba_num_evicted = self.evict_mamba(params.mamba_num)

        self._log_cache_stats(
            evicted_tokens=full_num_evicted,
            evicted_mamba_states=mamba_num_evicted,
        )
        return EvictResult(
            num_tokens_evicted=full_num_evicted,
            mamba_num_evicted=mamba_num_evicted,
        )

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

    def _disabled_for_marconi(self) -> bool:
        if self.page_size != 1:
            if not self._warned_page_size:
                logger.warning(
                    "MarconiCache currently requires page_size == 1; falling back to no-op."
                )
                self._warned_page_size = True
            return True
        if not self.enable_mamba_extra_buffer:
            if not self._warned_no_extra_buffer:
                logger.warning(
                    "MarconiCache requires mamba extra_buffer to capture input "
                    "branch states; falling back to no-op."
                )
                self._warned_no_extra_buffer = True
            return True
        return False

    def _empty_match_result(self) -> MatchResult:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
        )

    def _find_input_branch_checkpoint(self, req, key: RadixKey) -> Optional[int]:
        if not self.enable_mamba_extra_buffer or len(key) == 0:
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
                return self._align_branch_len(matched_len + prefix_len, req)

            matched_len += prefix_len
            node = child
            remaining = remaining[prefix_len:]
            if len(remaining):
                child_key = self.get_child_key_fn(remaining)

        if (
            node is not self.root_node
            and len(remaining) > 0
            and len(node.children) > 0
            and node.mamba_value is None
        ):
            return self._align_branch_len(matched_len, req)
        return None

    def _align_branch_len(self, branch_len: int, req) -> Optional[int]:
        if branch_len <= 0:
            return None
        chunk_size = get_global_server_args().mamba_cache_chunk_size
        if branch_len % chunk_size != 0:
            self.total_marconi_skipped_states += 1
            return None
        if branch_len <= len(req.prefix_indices):
            return None
        if branch_len >= len(req.origin_input_ids):
            return None
        node = self._lookup_node_by_prefix(
            req.origin_input_ids[:branch_len], req.extra_key
        )
        if node is not None and node.mamba_value is not None:
            return None
        return branch_len

    def _cache_kv_path(
        self,
        req,
        token_ids: list[int],
        kv_indices: torch.Tensor,
        duplicate_free_from: int,
        require_mamba: bool,
    ) -> tuple[int, TreeNode]:
        key = RadixKey(token_ids, req.extra_key)
        new_last_node = self._insert_kv_only(key, kv_indices, duplicate_free_from)
        self._attach_tracked_branch_state(req)
        if not require_mamba:
            return len(token_ids), new_last_node
        last_mamba_node = self._nearest_mamba_node(new_last_node)
        if last_mamba_node is None:
            return req.cache_protected_len, self.root_node
        return int(getattr(last_mamba_node, "marconi_prefix_tokens", 0)), last_mamba_node

    def _insert_kv_only(
        self, key: RadixKey, value: torch.Tensor, duplicate_free_from: int
    ) -> TreeNode:
        node = self.root_node
        total_prefix_length = 0
        child_key = self.get_child_key_fn(key)

        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            self.full_lru_list.reset_node_mru(node)
            if node.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(node)

            prefix_len = self.key_match_fn(node.key, key)
            if duplicate_free_from < total_prefix_length + prefix_len:
                start = max(0, duplicate_free_from - total_prefix_length)
                self.token_to_kv_pool_allocator.free(value[start:prefix_len])

            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)
                self._set_marconi_node_fields(
                    node, total_prefix_length - len(node.key)
                )

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            new_node.mamba_value = None
            self._set_marconi_node_fields(new_node, total_prefix_length)
            node.children[child_key] = new_node
            self.full_lru_list.insert_mru(new_node)
            self.full_evictable_size_ += len(new_node.value)
            self._on_token_node_created(new_node)
            node = new_node
        return node

    def _set_marconi_node_fields(self, node: TreeNode, parent_prefix_len: int) -> None:
        node.marconi_prefix_tokens = parent_prefix_len + len(node.key)
        node.marconi_branch_admitted = bool(
            getattr(node, "marconi_branch_admitted", False)
        )

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        new_node = super()._split_node(key, child, split_len)
        parent_prefix = int(getattr(new_node.parent, "marconi_prefix_tokens", 0))
        self._set_marconi_node_fields(new_node, parent_prefix)
        self._refresh_prefix_tokens(child)
        return new_node

    def _refresh_prefix_tokens(self, node: TreeNode) -> None:
        parent_prefix = int(getattr(node.parent, "marconi_prefix_tokens", 0))
        self._set_marconi_node_fields(node, parent_prefix)
        for child in node.children.values():
            self._refresh_prefix_tokens(child)

    def _attach_tracked_branch_state(self, req) -> bool:
        branch_len = req.mamba_branching_seqlen
        if branch_len is None:
            return False
        if req.mamba_last_track_seqlen != branch_len:
            self.total_marconi_state_capture_failures += 1
            return False

        node = self._lookup_node_by_prefix(
            req.origin_input_ids[:branch_len], req.extra_key
        )
        if node is None or node is self.root_node:
            self.total_marconi_state_capture_failures += 1
            return False
        if node.mamba_value is not None:
            return False

        mamba_ping_pong_track_buffer_to_keep = (
            self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                req.mamba_next_track_idx
            )
        )
        mamba_value = (
            req.mamba_ping_pong_track_buffer[mamba_ping_pong_track_buffer_to_keep]
            .unsqueeze(-1)
            .clone()
        )
        mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(mamba_value)
        if mamba_value_forked is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(
                mamba_value
            )
        if mamba_value_forked is None:
            self.total_marconi_state_capture_failures += 1
            return False

        node.mamba_value = mamba_value_forked
        node.marconi_branch_admitted = True
        node.last_access_time = get_last_access_time()
        self.mamba_lru_list.insert_mru(node)
        self.mamba_evictable_size_ += len(mamba_value_forked)
        self._on_checkpoint_created(node)
        self.total_marconi_created_states += 1
        return True

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

    def _free_finished_req_mamba(self, req) -> None:
        if req.mamba_pool_idx is not None:
            self.req_to_token_pool.free_mamba_cache(req)

    def _delete_leaf(self, node: TreeNode) -> None:
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"
        self._on_token_node_evicted(node)
        self.full_evictable_size_ -= len(node.key)
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
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}"
        self._on_checkpoint_evicted(node)
        self.mamba_evictable_size_ -= len(node.mamba_value)
        node.mamba_value = None
        node.marconi_branch_admitted = False
