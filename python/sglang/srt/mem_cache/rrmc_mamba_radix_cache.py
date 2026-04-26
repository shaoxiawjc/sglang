from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Optional

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

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

ANSI_RESET = "\x1b[0m"
ANSI_CYAN = "\x1b[36m"
ANSI_GREEN = "\x1b[32m"
ANSI_YELLOW = "\x1b[33m"
ANSI_RED = "\x1b[31m"


@dataclasses.dataclass(frozen=True)
class RRMCBlockSpec:
    block_id: str
    block_type: str
    version: str
    cacheable: bool
    token_count: int
    start: int
    end: int

    @property
    def identity(self) -> tuple[str, str, str, int]:
        return (self.block_type, self.block_id, self.version, self.token_count)


@dataclasses.dataclass(frozen=True)
class RRMCSegmentSpec:
    block_identity: tuple[str, str, str, int]
    block_id: str
    block_type: str
    version: str
    block_token_count: int
    segment_idx: int
    token_count: int
    start: int
    end: int
    is_block_end: bool

    @property
    def identity(self) -> tuple[Any, ...]:
        return (
            self.block_type,
            self.block_id,
            self.version,
            self.block_token_count,
            self.segment_idx,
            self.token_count,
        )


class RRMCMambaRadixCache(MambaRadixCache):
    """Block-aware hybrid prefix cache for RRMC-style RAG requests.

    This cache keeps KV and Mamba states on the same radix tree but uses explicit
    request-provided document blocks instead of raw token prefixes as the tree
    granularity. A request only participates in this cache when it carries
    `custom_params["rrmc"]["blocks"]`.

    Current implementation notes:
    - Only page_size == 1 is supported for RRMC mode.
    - Nodes are created per cacheable segment. Large blocks are split into
      fixed-size segments so chunked prefill boundaries can align with the tree.
    - KV nodes may still be segment-shaped internally, but reusable Mamba state
      is attached only at completed cacheable block ends.
    """

    def __init__(self, params):
        super().__init__(params)
        server_args = get_global_server_args()
        self.rrmc_eviction_policy = getattr(
            server_args, "rrmc_radix_eviction_policy", "value_aware"
        ).lower()
        self.rrmc_marconi_alpha = float(
            getattr(server_args, "rrmc_marconi_alpha", 1.0)
        )
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
        logger.info(
            "Initialized RRMCMambaRadixCache with eviction policy: %s, segment_size=%s, marconi_alpha=%s",
            self.rrmc_eviction_policy,
            self.rrmc_segment_size,
            self.rrmc_marconi_alpha,
        )
        # self.disable = True

    def _reset_cache_perf_counters(self) -> None:
        super()._reset_cache_perf_counters()
        self.total_hit_blocks = 0
        self.total_evicted_blocks = 0

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        req = params.req
        if self.disable or req is None:
            return self._empty_match_result()

        segment_specs = self._get_cacheable_prefix_segments(req, len(params.key))
        if segment_specs is None or len(segment_specs) == 0:
            return self._empty_match_result()

        node = self.root_node
        matched_nodes: list[TreeNode] = []
        matched_values: list[torch.Tensor] = []
        best_last_node = self.root_node
        best_value_len = 0

        extra_key = req.extra_key
        for spec in segment_specs:
            tree_key = self._make_tree_key(node, extra_key, spec)
            child = node.children.get(tree_key)
            if child is None:
                break

            matched_nodes.append(child)
            matched_values.append(child.value)
            node = child

            if child.mamba_value is not None:
                best_last_node = child
                best_value_len = len(matched_values)

        if best_value_len == 0:
            return self._empty_match_result()

        for node in matched_nodes[:best_value_len]:
            node.hit_count += 1

        self._log_rrmc_stats(
            hit_blocks=sum(
                1 for node in matched_nodes[:best_value_len] if node.mamba_value is not None
            ),
            hit_tokens=int(best_last_node.rrmc_prefix_tokens),
            evicted_blocks=0,
            evicted_tokens=0,
        )

        return self._match_post_processor(
            params=params,
            value=matched_values[:best_value_len],
            last_node=best_last_node,
            best_value_len=best_value_len,
        )

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            self._free_req_kv(req, 0, kv_committed_len)
            self._free_finished_req_mamba(req)
            return

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        segment_specs = self._get_cacheable_prefix_segments(
            req, kv_committed_len
        )
        cached_len = req.cache_protected_len
        if is_insert and segment_specs:
            cached_len, _ = self._cache_segment_path(
                req=req,
                segment_specs=segment_specs,
                token_ids=req.origin_input_ids,
                kv_indices=kv_indices,
                duplicate_free_from=req.cache_protected_len,
                chunked=False,
            )

        self._free_req_kv(req, max(req.cache_protected_len, cached_len), kv_committed_len)
        self._free_finished_req_mamba(req)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked: bool = False) -> None:
        token_ids = req.fill_ids
        if self.disable:
            return self._skip_cache_unfinished_req(req, len(token_ids))

        segment_specs = self._get_cacheable_prefix_segments(req, len(token_ids))
        if not segment_specs:
            return self._skip_cache_unfinished_req(req, len(token_ids))

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        canonical_len, new_last_node = self._cache_segment_path(
            req=req,
            segment_specs=segment_specs,
            token_ids=token_ids,
            kv_indices=kv_indices_orig,
            duplicate_free_from=req.cache_protected_len,
            chunked=chunked,
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
        req.last_node = new_last_node

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        before_blocks = self._count_cached_blocks()
        before_tokens, _ = self._total_size_helper()

        if self.rrmc_eviction_policy == "lru":
            result = super().evict(params)
        else:
            full_num_evicted = 0
            mamba_num_evicted = 0
            if params.num_tokens > 0:
                full_num_evicted = self._evict_full_custom(
                    params.num_tokens, self.rrmc_eviction_policy
                )
            if params.mamba_num > 0:
                mamba_num_evicted = self._evict_mamba_custom(
                    params.mamba_num, self.rrmc_eviction_policy
                )
            result = EvictResult(
                num_tokens_evicted=full_num_evicted,
                mamba_num_evicted=mamba_num_evicted,
            )

        after_blocks = self._count_cached_blocks()
        after_tokens, _ = self._total_size_helper()
        evicted_blocks = max(0, before_blocks - after_blocks)
        evicted_tokens = max(0, before_tokens - after_tokens)
        if evicted_blocks > 0 or evicted_tokens > 0:
            self._log_rrmc_stats(
                hit_blocks=0,
                hit_tokens=0,
                evicted_blocks=evicted_blocks,
                evicted_tokens=evicted_tokens,
            )

        return result

    def _delete_leaf(self, node: TreeNode) -> None:
        assert node.mamba_value is not None, (
            f"Invariant violated: RRMC leaf node must carry mamba, {node.id=}"
        )
        assert len(node.children) == 0, f"RRMC leaf node has children, {node.id=}"
        key = self._node_tree_key(node)
        v = node.parent.children.pop(key, None)
        assert v == node, f"Parent does not have RRMC child key, {key}"

        self.full_evictable_size_ -= len(node.key)
        self.mamba_evictable_size_ -= len(node.mamba_value)

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert node.mamba_value is None, (
            f"Deleting unexpected non-tombstone RRMC leaf node, {node.id=}"
        )
        assert len(node.children) == 0, f"RRMC leaf node has children, {node.id=}"
        key = self._node_tree_key(node)
        v = node.parent.children.pop(key, None)
        assert v == node, f"Parent does not have RRMC child key, {key}"
        self.full_evictable_size_ -= len(node.key)

    def _empty_match_result(self) -> MatchResult:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
        )

    def _current_track_limit(self, req: Req, total_tokens: int) -> int:
        track_limit = (
            req.mamba_last_track_seqlen
            if self.enable_mamba_extra_buffer
            else total_tokens
        )
        if track_limit is None:
            return 0
        return min(track_limit, len(req.origin_input_ids))

    def rrmc_next_extend_len(self, req: Req) -> Optional[int]:
        block_specs = self._get_cacheable_blocks(req)
        if not block_specs:
            return None

        prefix_len = len(req.prefix_indices)
        for block in block_specs:
            if block.end > prefix_len:
                return block.end - prefix_len
        return None

    def rrmc_disable_operator_chunk_state_tracking(self, req: Req) -> bool:
        return self._get_cacheable_blocks(req) is not None

    def _get_req_blocks(self, req: Req) -> Optional[list[RRMCBlockSpec]]:
        cached = getattr(req, "_rrmc_block_specs_cache", None)
        if cached is not None:
            return cached

        custom_params = getattr(req.sampling_params, "custom_params", None)
        if not isinstance(custom_params, dict):
            return None

        block_payload = None
        rrmc_payload = custom_params.get("rrmc")
        if isinstance(rrmc_payload, dict):
            block_payload = rrmc_payload.get("blocks")
        elif isinstance(custom_params.get("blocks"), list):
            block_payload = custom_params.get("blocks")

        if not isinstance(block_payload, list):
            return None

        block_specs: list[RRMCBlockSpec] = []
        offset = 0
        prompt_len = len(req.origin_input_ids)
        try:
            for idx, raw_block in enumerate(block_payload):
                if not isinstance(raw_block, dict):
                    raise TypeError(f"block[{idx}] is not a dict")

                token_count = int(raw_block["token_count"])
                if token_count <= 0:
                    raise ValueError(
                        f"block[{idx}] has non-positive token_count={token_count}"
                    )

                block_specs.append(
                    RRMCBlockSpec(
                        block_id=str(raw_block.get("block_id", f"block:{idx}")),
                        block_type=str(raw_block.get("block_type", "document")),
                        version=str(raw_block.get("version", "")),
                        cacheable=bool(raw_block.get("cacheable", True)),
                        token_count=token_count,
                        start=offset,
                        end=offset + token_count,
                    )
                )
                offset += token_count
        except Exception as exc:
            if not self._warned_bad_metadata:
                logger.warning("Invalid RRMC block metadata; disabling cache. %s", exc)
                self._warned_bad_metadata = True
            return None

        if offset > prompt_len:
            if not self._warned_bad_metadata:
                logger.warning(
                    "RRMC block metadata exceeds prompt length: blocks=%s prompt=%s",
                    offset,
                    prompt_len,
                )
                self._warned_bad_metadata = True
            return None

        setattr(req, "_rrmc_block_specs_cache", block_specs)
        return block_specs

    def _get_req_segments(self, req: Req) -> Optional[list[RRMCSegmentSpec]]:
        cached = getattr(req, "_rrmc_segment_specs_cache", None)
        if cached is not None:
            return cached

        block_specs = self._get_cacheable_blocks(req)
        if not block_specs:
            return None

        segment_specs: list[RRMCSegmentSpec] = []
        for block in block_specs:
            segment_idx = 0
            segment_start = block.start
            while segment_start < block.end:
                segment_end = min(segment_start + self.rrmc_segment_size, block.end)
                segment_specs.append(
                    RRMCSegmentSpec(
                        block_identity=block.identity,
                        block_id=block.block_id,
                        block_type=block.block_type,
                        version=block.version,
                        block_token_count=block.token_count,
                        segment_idx=segment_idx,
                        token_count=segment_end - segment_start,
                        start=segment_start,
                        end=segment_end,
                        is_block_end=segment_end == block.end,
                    )
                )
                segment_idx += 1
                segment_start = segment_end

        setattr(req, "_rrmc_segment_specs_cache", segment_specs)
        return segment_specs

    def _get_cacheable_prefix_segments(
        self, req: Req, token_limit: int
    ) -> Optional[list[RRMCSegmentSpec]]:
        if self.page_size != 1:
            if not self._warned_page_size:
                logger.warning(
                    "RRMCMambaRadixCache currently requires page_size == 1; falling back to no-op."
                )
                self._warned_page_size = True
            return None

        segment_specs = self._get_req_segments(req)
        if not segment_specs:
            return None

        token_limit = min(token_limit, len(req.origin_input_ids))
        last_completed_end = self._last_completed_cacheable_block_end(req, token_limit)
        if last_completed_end <= 0:
            return None
        cacheable_segments: list[RRMCSegmentSpec] = []
        for segment in segment_specs:
            if segment.end > last_completed_end:
                break
            cacheable_segments.append(segment)
        return cacheable_segments

    def _get_cacheable_blocks(self, req: Req) -> Optional[list[RRMCBlockSpec]]:
        block_specs = self._get_req_blocks(req)
        if not block_specs:
            return None

        cacheable_blocks: list[RRMCBlockSpec] = []
        for block in block_specs:
            if not block.cacheable:
                break
            cacheable_blocks.append(block)
        return cacheable_blocks or None

    def _last_completed_cacheable_block_end(self, req: Req, token_limit: int) -> int:
        block_specs = self._get_cacheable_blocks(req)
        if not block_specs:
            return 0

        last_completed_end = 0
        for block in block_specs:
            if block.end > token_limit:
                break
            last_completed_end = block.end
        return last_completed_end

    def _make_tree_key(
        self, parent: TreeNode, extra_key: Optional[str], segment: RRMCSegmentSpec
    ) -> tuple[Any, ...]:
        segment_key = segment.identity
        if parent is self.root_node:
            return ("rrmc", extra_key or "", segment_key)
        return ("rrmc", segment_key)

    def _node_tree_key(self, node: TreeNode) -> Any:
        return getattr(node, "rrmc_tree_key")

    def _new_segment_node(
        self,
        parent: TreeNode,
        tree_key: Any,
        segment: RRMCSegmentSpec,
        token_ids: list[int],
        kv_values: torch.Tensor,
        extra_key: Optional[str],
    ) -> TreeNode:
        node = TreeNode()
        node.parent = parent
        node.key = RadixKey(token_ids=token_ids, extra_key=extra_key)
        node.value = kv_values.clone()
        node.mamba_value = None
        node.last_access_time = get_last_access_time()
        node.hit_count = 1
        node.rrmc_tree_key = tree_key
        node.rrmc_block_identity = segment.block_identity
        node.rrmc_block_id = segment.block_id
        node.rrmc_segment_idx = segment.segment_idx
        node.rrmc_block_tokens = segment.token_count
        node.rrmc_prefix_tokens = segment.end
        parent.children[tree_key] = node
        self.full_lru_list.insert_mru(node)
        self.full_evictable_size_ += len(node.value)
        return node

    def _ensure_mamba_on_node(self, req: Req, node: TreeNode) -> bool:
        if node is self.root_node or node.mamba_value is not None:
            return False

        src_index = self._get_req_mamba_source(req)
        if src_index is None:
            return False

        dst_index = self.req_to_token_pool.mamba_pool.fork_from(src_index)
        if dst_index is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            dst_index = self.req_to_token_pool.mamba_pool.fork_from(src_index)
            assert dst_index is not None, "Can not alloc mamba cache for RRMC node"

        node.mamba_value = dst_index
        node.last_access_time = get_last_access_time()
        self.mamba_lru_list.insert_mru(node)
        self.mamba_evictable_size_ += len(dst_index)
        self._on_checkpoint_created(node)
        return True

    def _get_req_mamba_source(self, req: Req) -> Optional[torch.Tensor]:
        if req.req_pool_idx is not None:
            return self.req_to_token_pool.get_mamba_indices(req.req_pool_idx).unsqueeze(
                -1
            )
        if req.mamba_pool_idx is not None:
            return req.mamba_pool_idx.unsqueeze(-1)
        return None

    def _cache_segment_path(
        self,
        req: Req,
        segment_specs: list[RRMCSegmentSpec],
        token_ids: list[int],
        kv_indices: torch.Tensor,
        duplicate_free_from: int,
        chunked: bool,
    ) -> tuple[int, TreeNode]:
        del chunked  # RRMC counts explicit matches, not token-level chunk hits.

        node = self.root_node
        extra_key = req.extra_key
        duplicate_ranges: list[tuple[int, int]] = []
        for segment in segment_specs:
            tree_key = self._make_tree_key(node, extra_key, segment)
            child = node.children.get(tree_key)
            if child is None:
                child = self._new_segment_node(
                    parent=node,
                    tree_key=tree_key,
                    segment=segment,
                    token_ids=token_ids[segment.start : segment.end],
                    kv_values=kv_indices[segment.start : segment.end].to(
                        dtype=torch.int64, copy=True
                    ),
                    extra_key=extra_key,
                )
            else:
                child.last_access_time = get_last_access_time()
                self.full_lru_list.reset_node_mru(child)
                if child.mamba_value is not None:
                    self.mamba_lru_list.reset_node_mru(child)
                if segment.end > duplicate_free_from:
                    duplicate_ranges.append((segment.start, segment.end))
            node = child

        self._ensure_mamba_on_node(req, node)
        self._free_duplicate_ranges(kv_indices, duplicate_ranges, duplicate_free_from)
        last_mamba_node = self._nearest_mamba_node(node)
        if last_mamba_node is None:
            return req.cache_protected_len, self.root_node
        return int(getattr(last_mamba_node, "rrmc_prefix_tokens", 0)), last_mamba_node

    def _free_duplicate_ranges(
        self,
        kv_indices: torch.Tensor,
        duplicate_ranges: list[tuple[int, int]],
        duplicate_free_from: int,
    ) -> None:
        for start, end in duplicate_ranges:
            free_start = max(start, duplicate_free_from)
            if free_start < end:
                self.token_to_kv_pool_allocator.free(kv_indices[free_start:end])

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

    def _skip_cache_unfinished_req(self, req: Req, total_tokens: int) -> None:
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :total_tokens
        ]
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)

    def _nearest_mamba_node(self, node: TreeNode) -> Optional[TreeNode]:
        while node is not None and node is not self.root_node:
            if node.mamba_value is not None:
                return node
            node = node.parent
        return None

    def _free_req_kv(self, req: Req, start: int, end: int) -> None:
        if req.req_pool_idx is None:
            return
        if start >= end:
            return
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, start:end]
        self.token_to_kv_pool_allocator.free(kv_indices)

    def _free_finished_req_mamba(self, req: Req) -> None:
        if req.mamba_pool_idx is not None:
            self.req_to_token_pool.free_mamba_cache(req)

    def _value_aware_priority(self, node: TreeNode) -> tuple[float, int, int, float]:
        return (
            float(getattr(node, "hit_count", 0)),
            int(getattr(node, "rrmc_prefix_tokens", len(node.value))),
            int(getattr(node, "rrmc_block_tokens", len(node.value))),
            float(node.last_access_time),
        )

    def _policy_priority(self, node: TreeNode, policy: str) -> tuple[float, ...]:
        if policy == "value_aware":
            return self._value_aware_priority(node)

        hit_count = float(getattr(node, "hit_count", 0))
        prefix_tokens = float(getattr(node, "rrmc_prefix_tokens", len(node.value)))
        block_tokens = float(getattr(node, "rrmc_block_tokens", len(node.value)))
        last_access = float(node.last_access_time)

        if policy == "lfu":
            return (hit_count, last_access)
        if policy == "gdsf":
            # RRMC-adapted GDSF: use segment recompute cost as the utility term.
            return (hit_count * max(block_tokens, 1.0), last_access)
        if policy == "pgdsf":
            # RRMC-adapted PGDSF: deeper reusable prefixes imply higher future
            # recompute savings, so use prefix depth as the cost term.
            return (hit_count * max(prefix_tokens, 1.0), last_access)

        return self._value_aware_priority(node)

    def _normalized(self, values: list[float]) -> list[float]:
        if not values:
            return []
        lower = min(values)
        upper = max(values)
        if upper <= lower:
            return [1.0 for _ in values]
        scale = upper - lower
        return [(value - lower) / scale for value in values]

    def _marconi_memory_cost(self, node: TreeNode, *, kind: str) -> float:
        if kind == "mamba":
            if node.mamba_value is None:
                return 0.0
            return float(len(node.mamba_value))
        return float(len(node.value)) if node.value is not None else 0.0

    def _marconi_flop_efficiency(self, node: TreeNode, *, kind: str) -> float:
        memory_cost = max(self._marconi_memory_cost(node, kind=kind), 1.0)
        prefix_tokens = float(getattr(node, "rrmc_prefix_tokens", len(node.value)))
        return prefix_tokens / memory_cost

    def _select_marconi_candidate(
        self, candidates: list[TreeNode], *, kind: str
    ) -> Optional[TreeNode]:
        if not candidates:
            return None
        recencies = self._normalized(
            [float(node.last_access_time) for node in candidates]
        )
        efficiencies = self._normalized(
            [self._marconi_flop_efficiency(node, kind=kind) for node in candidates]
        )

        best_node: Optional[TreeNode] = None
        best_score: Optional[tuple[float, float]] = None
        for node, recency, efficiency in zip(candidates, recencies, efficiencies):
            utility = recency + self.rrmc_marconi_alpha * efficiency
            score = (utility, float(node.last_access_time))
            if best_score is None or score < best_score:
                best_score = score
                best_node = node
        return best_node

    def _select_full_candidate(self, policy: str) -> Optional[TreeNode]:
        candidates = [
            node
            for node in self._collect_all_nodes()
            if node is not self.root_node
            and len(node.children) == 0
            and node.full_lock_ref == 0
            and node.value is not None
        ]
        if not candidates:
            return None
        if policy == "marconi":
            # Marconi considers low-fanout entries for eviction, but RRMC's
            # segment KV ownership is leaf-based today, so we keep full-KV
            # eviction leaf-only and only change the scoring function.
            return self._select_marconi_candidate(candidates, kind="full")
        return min(candidates, key=lambda node: self._policy_priority(node, policy))

    def _select_mamba_candidate(self, policy: str) -> Optional[TreeNode]:
        candidates = [
            node
            for node in self._collect_nontombstone_nodes()
            if node is not self.root_node
            and node.mamba_lock_ref == 0
            and node.mamba_value is not None
        ]
        if not candidates:
            return None
        if policy == "marconi":
            marconi_candidates = [
                node for node in candidates if len(node.children) <= 1
            ]
            if marconi_candidates:
                return self._select_marconi_candidate(
                    marconi_candidates, kind="mamba"
                )
        return min(candidates, key=lambda node: self._policy_priority(node, policy))

    def _evict_full_custom(self, num_tokens: int, policy: str) -> int:
        full_num_evicted = 0
        while full_num_evicted < num_tokens:
            node = self._select_full_candidate(policy)
            if node is None:
                break
            full_num_evicted_delta, _, _, _ = self._evict_leaf_node(node, False)
            full_num_evicted += full_num_evicted_delta
        return full_num_evicted

    def _evict_mamba_custom(self, mamba_num: int, policy: str) -> int:
        mamba_num_evicted = 0
        while mamba_num_evicted < mamba_num:
            node = self._select_mamba_candidate(policy)
            if node is None:
                break
            assert node.mamba_value is not None, f"node has no mamba value, {node.id=}"
            if len(node.children) > 0:
                self.req_to_token_pool.mamba_pool.free(node.mamba_value)
                self.mamba_lru_list.remove_node(node)
                self._tombstone_internal_node(node)
                mamba_num_evicted += 1
            else:
                _, delta, _, _ = self._evict_leaf_node(node, True)
                mamba_num_evicted += delta
        return mamba_num_evicted

    def _count_cached_blocks(self) -> int:
        return sum(
            1
            for node in self._collect_all_nodes()
            if node is not self.root_node
            and getattr(node, "is_checkpoint_state", False)
            and node.mamba_value is not None
        )

    def _log_rrmc_stats(
        self,
        *,
        hit_blocks: int,
        hit_tokens: int,
        evicted_blocks: int,
        evicted_tokens: int,
    ) -> None:
        if (
            hit_blocks <= 0
            and hit_tokens <= 0
            and evicted_blocks <= 0
            and evicted_tokens <= 0
        ):
            return
        self.total_hit_blocks += int(hit_blocks)
        self.total_evicted_blocks += int(evicted_blocks)
        self._log_cache_stats(
            hit_tokens=hit_tokens,
            evicted_tokens=evicted_tokens,
            extra_fields={
                "hit_blocks": int(hit_blocks),
                "total_hit_blocks": int(self.total_hit_blocks),
                "evicted_blocks": int(evicted_blocks),
                "total_evicted_blocks": int(self.total_evicted_blocks),
            },
        )

    def get_cache_metrics(self) -> dict[str, Any]:
        metrics = super().get_cache_metrics()
        metrics["total_hit_blocks"] = int(self.total_hit_blocks)
        metrics["total_evicted_blocks"] = int(self.total_evicted_blocks)
        return metrics
