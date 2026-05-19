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


@dataclasses.dataclass(frozen=True)
class RRMCForwardBoundary:
    req_index: int
    block_end: int
    local_start: int
    local_end: int
    mamba_index: int


@dataclasses.dataclass(frozen=True)
class RRMCForwardBoundaryBatch:
    boundaries: list[RRMCForwardBoundary]

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
        return [boundary.block_end for boundary in self.boundaries]

    @property
    def local_starts(self) -> list[int]:
        return [boundary.local_start for boundary in self.boundaries]

    @property
    def local_ends(self) -> list[int]:
        return [boundary.local_end for boundary in self.boundaries]

    @property
    def mamba_indices(self) -> list[int]:
        return [boundary.mamba_index for boundary in self.boundaries]


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
        self.enable_rrmc_admission = bool(
            getattr(server_args, "enable_rrmc_admission", False)
        )
        self.rrmc_admission_min_accesses = max(
            1, int(getattr(server_args, "rrmc_admission_min_accesses", 2))
        )
        self.rrmc_admission_counts: dict[tuple[Any, ...], int] = {}
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
            "Initialized RRMCMambaRadixCache with eviction policy: lru, segment_size=%s, admission=%s, admission_min_accesses=%s",
            self.rrmc_segment_size,
            self.enable_rrmc_admission,
            self.rrmc_admission_min_accesses,
        )
        # self.disable = True

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "rrmc_admission_counts"):
            self.rrmc_admission_counts.clear()

    def _reset_cache_perf_counters(self) -> None:
        super()._reset_cache_perf_counters()
        self.total_hit_blocks = 0
        self.total_evicted_blocks = 0
        self.total_rrmc_forced_chunks = 0
        self.total_rrmc_created_states = 0
        self.total_rrmc_boundary_states_created = 0
        self.total_rrmc_boundary_state_capture_failures = 0
        self.total_rrmc_accepted_state_hits = 0
        self.total_rrmc_skipped_cold_boundaries = 0

    def record_rrmc_forced_chunk(
        self,
        *,
        req: Optional[Req] = None,
        extend_len: int = 0,
        full_extend_len: int = 0,
    ) -> None:
        _ = (req, extend_len, full_extend_len)
        # RRMC no longer changes scheduler granularity. Keep the metric as a
        # compatibility field, but it should remain zero for the operator-boundary
        # implementation.
        return

    def record_accepted_hit_tokens(
        self, hit_tokens: int, req: Optional[Req] = None
    ) -> None:
        super().record_accepted_hit_tokens(hit_tokens, req=req)
        if hit_tokens <= 0 or req is None:
            return
        node = getattr(req, "last_node", None)
        if not isinstance(node, TreeNode) or node is self.root_node:
            return
        if node.mamba_value is not None:
            node.has_been_shared = True
            self.total_rrmc_accepted_state_hits += 1

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

        if params.log_stats:
            self._log_rrmc_stats(
                hit_blocks=sum(
                    1
                    for node in matched_nodes[:best_value_len]
                    if node.mamba_value is not None
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
        try:
            kv_committed_len = req.pop_committed_kv_cache()
            if self.disable:
                self._free_req_kv(req, 0, kv_committed_len)
                self._free_finished_req_mamba(req)
                return

            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]

            segment_specs = self._get_cacheable_prefix_segments(req, kv_committed_len)
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

            self._free_req_kv(
                req, max(req.cache_protected_len, cached_len), kv_committed_len
            )
            self._free_finished_req_mamba(req)
            self.dec_lock_ref(req.last_node)
        finally:
            self._release_unattached_rrmc_boundary_slots(req)

    def cache_unfinished_req(self, req: Req, chunked: bool = False) -> None:
        try:
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
        finally:
            self._release_unattached_rrmc_boundary_slots(req)

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        before_blocks = self._count_cached_blocks()
        before_tokens, _ = self._total_size_helper()
        result = super().evict(params)

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

        self._on_token_node_evicted(node)
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
        self._on_token_node_evicted(node)
        self.full_evictable_size_ -= len(node.key)

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
        else:
            self._delete_tombstone_leaf(x)

        x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
        full_num_evicted += leaf_full_num_evicted
        return full_num_evicted, mamba_num_evicted, x, x_next

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
        # Deprecated compatibility hook. RRMC document boundaries are now handled
        # inside the GDN operator, so scheduler chunk sizing must stay unchanged.
        _ = req
        return None

    def rrmc_disable_operator_chunk_state_tracking(self, req: Req) -> bool:
        return self._get_cacheable_blocks(req) is not None

    def _lookup_block_end_node(self, req: Req, block_end: int) -> Optional[TreeNode]:
        segment_specs = self._get_req_segments(req)
        if not segment_specs:
            return None

        node = self.root_node
        extra_key = req.extra_key
        for segment in segment_specs:
            if segment.end > block_end:
                return None
            tree_key = self._make_tree_key(node, extra_key, segment)
            child = node.children.get(tree_key)
            if child is None:
                return None
            node = child
            if segment.end == block_end:
                return node
        return None

    def _has_cached_mamba_for_block_end(self, req: Req, block_end: int) -> bool:
        node = self._lookup_block_end_node(req, block_end)
        return node is not None and node.mamba_value is not None

    def _cached_mamba_by_block_end(self, req: Req) -> dict[int, bool]:
        segment_specs = self._get_req_segments(req)
        if not segment_specs:
            return {}

        node = self.root_node
        extra_key = req.extra_key
        cached_by_end: dict[int, bool] = {}
        for segment in segment_specs:
            tree_key = self._make_tree_key(node, extra_key, segment)
            child = node.children.get(tree_key)
            if child is None:
                break
            node = child
            if segment.is_block_end:
                cached_by_end[int(segment.end)] = node.mamba_value is not None
        return cached_by_end

    def _rrmc_admission_key(
        self,
        req: Req,
        block_path: list[tuple[str, str, str, int]],
    ) -> tuple[Any, ...]:
        return ("rrmc_path", req.extra_key or "", tuple(block_path))

    def _admit_rrmc_boundary_capture(self, admission_key: tuple[Any, ...]) -> bool:
        if not self.enable_rrmc_admission:
            return True
        if self.rrmc_admission_min_accesses <= 1:
            return True

        access_count = self.rrmc_admission_counts.get(admission_key, 0) + 1
        self.rrmc_admission_counts[admission_key] = access_count
        if access_count < self.rrmc_admission_min_accesses:
            self.total_rrmc_skipped_cold_boundaries += 1
            return False
        return True

    def _was_rrmc_boundary_admission_skipped(self, req: Req, block_end: int) -> bool:
        skipped = getattr(req, "_rrmc_admission_skipped_block_ends", None)
        return isinstance(skipped, set) and int(block_end) in skipped

    def prepare_rrmc_forward_boundaries(
        self,
        reqs: list[Req],
        prefix_lens: list[int],
        extend_lens: list[int],
    ) -> Optional[RRMCForwardBoundaryBatch]:
        """Allocate Mamba slots for cacheable RRMC block ends in this extend.

        The returned local ranges are flattened-token ranges in the current
        prefill batch. They let the linear attention backend run the RRMC request
        as consecutive operator segments and copy the working state at block
        boundaries into the allocated Mamba slots.
        """
        for req in reqs:
            self._release_unattached_rrmc_boundary_slots(req)
            setattr(req, "_rrmc_boundary_mamba_indices_by_end", {})
            setattr(req, "_rrmc_pending_boundary_mamba_indices", {})
            setattr(req, "_rrmc_admission_skipped_block_ends", set())

        if self.disable or self.page_size != 1:
            return None

        pending: list[tuple[Req, int, int, int, int]] = []
        flat_offset = 0
        for req_index, (req, prefix_len, extend_len) in enumerate(
            zip(reqs, prefix_lens, extend_lens)
        ):
            block_specs = self._get_cacheable_blocks(req)
            if not block_specs:
                flat_offset += int(extend_len)
                continue

            cached_mamba_by_end = self._cached_mamba_by_block_end(req)
            prefix_len = int(prefix_len)
            extend_len = int(extend_len)
            extend_end = prefix_len + extend_len
            segment_local_start = flat_offset
            admission_block_path: list[tuple[str, str, str, int]] = []
            for block in block_specs:
                admission_block_path.append(block.identity)
                if block.end <= prefix_len:
                    continue
                if block.end > extend_end:
                    break
                if cached_mamba_by_end.get(int(block.end), False):
                    continue
                admission_key = self._rrmc_admission_key(req, admission_block_path)
                if not self._admit_rrmc_boundary_capture(admission_key):
                    getattr(req, "_rrmc_admission_skipped_block_ends").add(
                        int(block.end)
                    )
                    continue

                local_end = flat_offset + (block.end - prefix_len)
                if local_end <= segment_local_start:
                    continue
                pending.append(
                    (
                        req,
                        req_index,
                        int(block.end),
                        int(segment_local_start),
                        int(local_end),
                    )
                )
                segment_local_start = local_end

            flat_offset += extend_len

        if not pending:
            return None

        slots = self.req_to_token_pool.mamba_pool.alloc_uninitialized(len(pending))
        if slots is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=len(pending)))
            slots = self.req_to_token_pool.mamba_pool.alloc_uninitialized(len(pending))

        if slots is None:
            self.total_rrmc_boundary_state_capture_failures += len(pending)
            return None

        slot_ids = [int(slot_id) for slot_id in slots.detach().cpu().tolist()]
        boundaries: list[RRMCForwardBoundary] = []
        for slot_offset, (
            req,
            req_index,
            block_end,
            local_start,
            local_end,
        ) in enumerate(pending):
            slot = slots[slot_offset : slot_offset + 1]
            slot_id = slot_ids[slot_offset]
            boundary_map = getattr(req, "_rrmc_boundary_mamba_indices_by_end")
            pending_map = getattr(req, "_rrmc_pending_boundary_mamba_indices")
            boundary_map[block_end] = slot
            pending_map[block_end] = slot
            boundaries.append(
                RRMCForwardBoundary(
                    req_index=req_index,
                    block_end=block_end,
                    local_start=local_start,
                    local_end=local_end,
                    mamba_index=slot_id,
                )
            )

        return RRMCForwardBoundaryBatch(boundaries=boundaries)

    def _release_unattached_rrmc_boundary_slots(self, req: Req) -> None:
        pending_map = getattr(req, "_rrmc_pending_boundary_mamba_indices", None)
        if not pending_map:
            return
        pending_slots = list(pending_map.values())
        if pending_slots:
            self.req_to_token_pool.mamba_pool.free(torch.cat(pending_slots))
        pending_map.clear()

    def _consume_rrmc_boundary_slot(self, req: Req, block_end: int) -> Optional[torch.Tensor]:
        boundary_map = getattr(req, "_rrmc_boundary_mamba_indices_by_end", None)
        if not boundary_map:
            return None
        return boundary_map.get(int(block_end))

    def _mark_rrmc_boundary_slot_attached(self, req: Req, block_end: int) -> None:
        pending_map = getattr(req, "_rrmc_pending_boundary_mamba_indices", None)
        if not pending_map:
            return
        pending_map.pop(int(block_end), None)

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
        node.value = kv_values.to(dtype=torch.int64, copy=True)
        node.mamba_value = None
        node.last_access_time = get_last_access_time()
        node.rrmc_tree_key = tree_key
        node.rrmc_block_identity = segment.block_identity
        node.rrmc_block_id = segment.block_id
        node.rrmc_segment_idx = segment.segment_idx
        node.rrmc_block_tokens = segment.token_count
        node.rrmc_total_block_tokens = segment.block_token_count
        node.rrmc_is_block_end = segment.is_block_end
        node.rrmc_prefix_tokens = segment.end
        parent.children[tree_key] = node
        self.full_lru_list.insert_mru(node)
        self.full_evictable_size_ += len(node.value)
        self._on_token_node_created(node)
        return node

    def _ensure_mamba_on_node(
        self,
        req: Req,
        node: TreeNode,
        boundary_slot: Optional[torch.Tensor] = None,
    ) -> bool:
        if node is self.root_node or node.mamba_value is not None:
            return False

        if boundary_slot is None:
            self.total_rrmc_boundary_state_capture_failures += 1
            return False

        node.mamba_value = boundary_slot.to(dtype=torch.int64, copy=False)
        node.last_access_time = get_last_access_time()
        self.mamba_lru_list.insert_mru(node)
        self.mamba_evictable_size_ += len(node.mamba_value)
        self._on_checkpoint_created(node)
        self._mark_rrmc_boundary_slot_attached(
            req, int(getattr(node, "rrmc_prefix_tokens", 0))
        )
        self.total_rrmc_created_states += 1
        self.total_rrmc_boundary_states_created += 1
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
                child.last_access_time = get_last_access_time()
                self.full_lru_list.reset_node_mru(child)
                if child.mamba_value is not None:
                    self.mamba_lru_list.reset_node_mru(child)
                if segment.end > duplicate_free_from:
                    duplicate_ranges.append((segment.start, segment.end))
            node = child
            if segment.is_block_end:
                if self._was_rrmc_boundary_admission_skipped(req, segment.end):
                    continue
                boundary_slot = self._consume_rrmc_boundary_slot(req, segment.end)
                self._ensure_mamba_on_node(req, child, boundary_slot)

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

    def _rrmc_cache_prefix_limit(
        self, req: Req, segment_specs: list[RRMCSegmentSpec]
    ) -> int:
        """Return the largest prefix end that will have a reusable Mamba state.

        Full-attention KV before a skipped RRMC boundary is only useful when a
        later boundary in the same prefix is cached. If admission skips every
        boundary, keeping KV-only tree nodes would leave the allocator and tree
        accounting inconsistent after the request-owned KV is freed.
        """
        limit = 0

        pending_by_end = getattr(req, "_rrmc_boundary_mamba_indices_by_end", None)
        skipped_by_end = getattr(req, "_rrmc_admission_skipped_block_ends", None)
        if not isinstance(pending_by_end, dict):
            pending_by_end = {}
        if not isinstance(skipped_by_end, set):
            skipped_by_end = set()

        for segment in segment_specs:
            if not segment.is_block_end:
                continue
            block_end = int(segment.end)
            if block_end in skipped_by_end:
                continue
            if block_end in pending_by_end:
                limit = block_end

        node = self.root_node
        extra_key = req.extra_key
        for segment in segment_specs:
            tree_key = self._make_tree_key(node, extra_key, segment)
            child = node.children.get(tree_key)
            if child is None:
                break
            node = child
            if segment.is_block_end and child.mamba_value is not None:
                limit = max(limit, int(segment.end))

        return limit

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
        metrics["total_rrmc_forced_chunks"] = int(self.total_rrmc_forced_chunks)
        metrics["total_rrmc_created_states"] = int(self.total_rrmc_created_states)
        metrics["total_rrmc_boundary_states_created"] = int(
            self.total_rrmc_boundary_states_created
        )
        metrics["total_rrmc_boundary_state_capture_failures"] = int(
            self.total_rrmc_boundary_state_capture_failures
        )
        metrics["total_rrmc_accepted_state_hits"] = int(
            self.total_rrmc_accepted_state_hits
        )
        metrics["total_rrmc_skipped_cold_boundaries"] = int(
            self.total_rrmc_skipped_cold_boundaries
        )
        return metrics
