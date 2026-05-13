from __future__ import annotations

import bisect
import dataclasses
import json
import logging
import math
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
        self.rrmc_eviction_policy = getattr(
            server_args, "rrmc_radix_eviction_policy", "value_aware"
        ).lower()
        self.rrmc_marconi_alpha = float(
            getattr(server_args, "rrmc_marconi_alpha", 1.0)
        )
        self.rrmc_pgdsf_cost_profile_path = getattr(
            server_args, "rrmc_pgdsf_cost_profile_path", None
        )
        self.rrmc_pgdsf_cost_profile = self._load_pgdsf_cost_profile(
            self.rrmc_pgdsf_cost_profile_path
        )
        self.rrmc_pgdsf_cost_profile_map = {
            (cached_tokens, non_cached_tokens): cost
            for cached_tokens, non_cached_tokens, cost in self.rrmc_pgdsf_cost_profile
        }
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
            "Initialized RRMCMambaRadixCache with eviction policy: %s, segment_size=%s, marconi_alpha=%s, pgdsf_profile_points=%s, admission=%s, admission_min_accesses=%s",
            self.rrmc_eviction_policy,
            self.rrmc_segment_size,
            self.rrmc_marconi_alpha,
            len(self.rrmc_pgdsf_cost_profile),
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
        self.rrmc_full_gdsf_clock = 0.0
        self.rrmc_mamba_gdsf_clock = 0.0
        self.rrmc_full_pgdsf_clock = 0.0
        self.rrmc_mamba_pgdsf_clock = 0.0

    def _load_pgdsf_cost_profile(
        self, profile_path: Optional[str]
    ) -> list[tuple[float, float, float]]:
        if not profile_path:
            return []
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning(
                "Failed to load RRMC PGDSF cost profile %s: %s", profile_path, exc
            )
            return []

        if isinstance(payload, dict):
            raw_points = payload.get("points", [])
        else:
            raw_points = payload
        if not isinstance(raw_points, list):
            logger.warning(
                "Ignoring RRMC PGDSF cost profile %s: expected a list or a {'points': [...]} object",
                profile_path,
            )
            return []

        points: list[tuple[float, float, float]] = []
        for raw in raw_points:
            if not isinstance(raw, dict):
                continue
            cached_tokens = raw.get("cached_tokens", raw.get("cached"))
            non_cached_tokens = raw.get(
                "non_cached_tokens", raw.get("new_tokens", raw.get("uncached_tokens"))
            )
            cost = raw.get(
                "cost", raw.get("cost_ms", raw.get("latency_ms", raw.get("time_ms")))
            )
            try:
                cached_tokens = float(cached_tokens)
                non_cached_tokens = float(non_cached_tokens)
                cost = float(cost)
            except (TypeError, ValueError):
                continue
            if cached_tokens < 0 or non_cached_tokens <= 0 or cost <= 0:
                continue
            points.append((cached_tokens, non_cached_tokens, cost))

        points.sort()
        if not points:
            logger.warning(
                "RRMC PGDSF cost profile %s has no usable points", profile_path
            )
        return points

    def _lookup_pgdsf_profile_cost(
        self, cached_tokens: float, non_cached_tokens: float
    ) -> Optional[float]:
        profile = getattr(self, "rrmc_pgdsf_cost_profile", [])
        if not profile:
            return None

        cached_axis = sorted({point[0] for point in profile})
        non_cached_axis = sorted({point[1] for point in profile})
        if not cached_axis or not non_cached_axis:
            return None

        def bounds(axis: list[float], value: float) -> tuple[float, float]:
            pos = bisect.bisect_left(axis, value)
            if pos <= 0:
                return axis[0], axis[0]
            if pos >= len(axis):
                return axis[-1], axis[-1]
            return axis[pos - 1], axis[pos]

        x0, x1 = bounds(cached_axis, cached_tokens)
        y0, y1 = bounds(non_cached_axis, non_cached_tokens)
        profile_map = getattr(self, "rrmc_pgdsf_cost_profile_map", {})

        def value_at(x: float, y: float) -> Optional[float]:
            value = profile_map.get((x, y))
            if value is not None:
                return value
            nearest = min(
                profile,
                key=lambda point: abs(point[0] - x) + abs(point[1] - y),
            )
            return nearest[2]

        q00 = value_at(x0, y0)
        q01 = value_at(x0, y1)
        q10 = value_at(x1, y0)
        q11 = value_at(x1, y1)
        if q00 is None or q01 is None or q10 is None or q11 is None:
            return None
        if x0 == x1 and y0 == y1:
            return q00
        if x0 == x1:
            weight = 0.0 if y1 == y0 else (non_cached_tokens - y0) / (y1 - y0)
            return q00 * (1.0 - weight) + q01 * weight
        if y0 == y1:
            weight = 0.0 if x1 == x0 else (cached_tokens - x0) / (x1 - x0)
            return q00 * (1.0 - weight) + q10 * weight

        x_weight = (cached_tokens - x0) / (x1 - x0)
        y_weight = (non_cached_tokens - y0) / (y1 - y0)
        return (
            q00 * (1.0 - x_weight) * (1.0 - y_weight)
            + q10 * x_weight * (1.0 - y_weight)
            + q01 * (1.0 - x_weight) * y_weight
            + q11 * x_weight * y_weight
        )

    def _estimate_rrmc_recompute_cost(
        self, cached_tokens: int, non_cached_tokens: int
    ) -> float:
        non_cached_tokens = max(1, int(non_cached_tokens))
        cached_tokens = max(0, int(cached_tokens))
        profiled_cost = self._lookup_pgdsf_profile_cost(
            float(cached_tokens), float(non_cached_tokens)
        )
        if profiled_cost is not None:
            return max(float(profiled_cost), 1e-6)
        # Fallback approximates the profile-aware PGDSF trend without requiring
        # a profiler: later documents are more expensive because they are
        # computed at a longer prefix.
        return float(non_cached_tokens) * (
            1.0 + math.log2(float(cached_tokens) + 2.0)
        )

    def _rrmc_policy_size(self, node: TreeNode, *, kind: str) -> float:
        if kind == "mamba":
            if node.mamba_value is None:
                return 1.0
            return max(float(len(node.mamba_value)), 1.0)
        if node.value is None:
            return 1.0
        return max(float(len(node.value)), 1.0)

    def _rrmc_gdsf_cost(self, node: TreeNode, *, kind: str) -> float:
        block_tokens = float(
            getattr(
                node,
                "rrmc_block_tokens",
                len(node.value) if node.value is not None else 1,
            )
        )
        prefix_tokens = float(getattr(node, "rrmc_prefix_tokens", block_tokens))
        if kind == "mamba":
            return max(prefix_tokens, 1.0)
        return max(block_tokens, 1.0)

    def _rrmc_frequency(self, node: TreeNode, *, kind: str) -> float:
        if kind == "mamba":
            frequency = float(getattr(node, "rrmc_mamba_frequency", 0))
            if frequency <= 0:
                frequency = float(getattr(node, "accepted_mamba_hit_count", 0))
            return max(frequency, 1.0)
        frequency = float(getattr(node, "rrmc_full_frequency", 0))
        if frequency <= 0:
            frequency = float(getattr(node, "accepted_hit_count", 0))
        return max(frequency, 1.0)

    def _rrmc_clock_attr(self, policy: str, *, kind: str) -> str:
        return f"rrmc_{kind}_{policy}_clock"

    def _rrmc_priority_attr(self, policy: str, *, kind: str) -> str:
        return f"rrmc_{kind}_{policy}_priority"

    def _rrmc_pgdsf_cost_attr(self, suffix: str, *, kind: str) -> str:
        return f"rrmc_{kind}_pgdsf_{suffix}"

    def _record_rrmc_pgdsf_cost_sample(
        self,
        node: TreeNode,
        *,
        kind: str,
        cached_tokens: int,
        non_cached_tokens: int,
    ) -> None:
        cost = self._estimate_rrmc_recompute_cost(cached_tokens, non_cached_tokens)
        cost_per_size = cost / self._rrmc_policy_size(node, kind=kind)
        total_attr = self._rrmc_pgdsf_cost_attr("total_cost_per_size", kind=kind)
        samples_attr = self._rrmc_pgdsf_cost_attr("cost_samples", kind=kind)
        avg_attr = self._rrmc_pgdsf_cost_attr("avg_cost_per_size", kind=kind)
        total = float(getattr(node, total_attr, 0.0)) + float(cost_per_size)
        samples = int(getattr(node, samples_attr, 0)) + 1
        setattr(node, total_attr, total)
        setattr(node, samples_attr, samples)
        setattr(node, avg_attr, total / max(samples, 1))

    def _rrmc_pgdsf_cost_per_size(self, node: TreeNode, *, kind: str) -> float:
        avg_attr = self._rrmc_pgdsf_cost_attr("avg_cost_per_size", kind=kind)
        observed = float(getattr(node, avg_attr, 0.0))
        if observed > 0:
            return observed
        block_tokens = int(
            getattr(
                node,
                "rrmc_block_tokens",
                len(node.value) if node.value is not None else 1,
            )
        )
        prefix_tokens = int(getattr(node, "rrmc_prefix_tokens", block_tokens))
        cached_tokens = max(0, prefix_tokens - block_tokens)
        cost = self._estimate_rrmc_recompute_cost(cached_tokens, max(block_tokens, 1))
        return cost / self._rrmc_policy_size(node, kind=kind)

    def _refresh_rrmc_priority(self, node: TreeNode, *, kind: str) -> None:
        frequency = self._rrmc_frequency(node, kind=kind)
        size = self._rrmc_policy_size(node, kind=kind)
        gdsf_clock = float(
            getattr(self, self._rrmc_clock_attr("gdsf", kind=kind), 0.0)
        )
        pgdsf_clock = float(
            getattr(self, self._rrmc_clock_attr("pgdsf", kind=kind), 0.0)
        )
        setattr(
            node,
            self._rrmc_priority_attr("gdsf", kind=kind),
            gdsf_clock + frequency * self._rrmc_gdsf_cost(node, kind=kind) / size,
        )
        setattr(
            node,
            self._rrmc_priority_attr("pgdsf", kind=kind),
            pgdsf_clock + frequency * self._rrmc_pgdsf_cost_per_size(node, kind=kind),
        )

    def _mark_rrmc_policy_access(
        self,
        req: Req,
        node: TreeNode,
        segment: RRMCSegmentSpec,
        *,
        kind: str,
        duplicate_free_from: int,
    ) -> None:
        touched_attr = f"_rrmc_policy_touched_{kind}_nodes"
        touched = getattr(req, touched_attr, None)
        if touched is None:
            touched = set()
            setattr(req, touched_attr, touched)
        if node.id in touched:
            return
        touched.add(node.id)

        freq_attr = f"rrmc_{kind}_frequency"
        setattr(node, freq_attr, int(getattr(node, freq_attr, 0)) + 1)
        computed_start = max(segment.start, duplicate_free_from)
        if computed_start < segment.end:
            self._record_rrmc_pgdsf_cost_sample(
                node,
                kind=kind,
                cached_tokens=computed_start,
                non_cached_tokens=segment.end - computed_start,
            )
        self._refresh_rrmc_priority(node, kind=kind)

    def _advance_rrmc_clock(
        self, policy: str, *, kind: str, evicted_priority: float
    ) -> None:
        if policy not in ("gdsf", "pgdsf"):
            return
        clock_attr = self._rrmc_clock_attr(policy, kind=kind)
        current = float(getattr(self, clock_attr, 0.0))
        setattr(self, clock_attr, max(current, float(evicted_priority)))

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
        node.accepted_mamba_hit_count = (
            int(getattr(node, "accepted_mamba_hit_count", 0)) + 1
        )
        if node.mamba_value is not None:
            node.has_been_shared = True
            self.total_rrmc_accepted_state_hits += 1
        self._refresh_rrmc_priority(node, kind="mamba")
        while node is not None and node is not self.root_node:
            node.accepted_hit_count = int(getattr(node, "accepted_hit_count", 0)) + 1
            self._refresh_rrmc_priority(node, kind="full")
            node = node.parent

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
        node.hit_count = 1
        node.accepted_hit_count = 0
        node.accepted_mamba_hit_count = 0
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
            self._mark_rrmc_policy_access(
                req,
                child,
                segment,
                kind="full",
                duplicate_free_from=duplicate_free_from,
            )
            if child.mamba_value is not None:
                self._mark_rrmc_policy_access(
                    req,
                    child,
                    segment,
                    kind="mamba",
                    duplicate_free_from=duplicate_free_from,
                )

            if segment.is_block_end:
                if self._was_rrmc_boundary_admission_skipped(req, segment.end):
                    continue
                boundary_slot = self._consume_rrmc_boundary_slot(req, segment.end)
                self._ensure_mamba_on_node(req, child, boundary_slot)
                if child.mamba_value is not None:
                    self._mark_rrmc_policy_access(
                        req,
                        child,
                        segment,
                        kind="mamba",
                        duplicate_free_from=duplicate_free_from,
                    )

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

    def _value_aware_priority(self, node: TreeNode) -> tuple[float, int, int, float]:
        return (
            float(getattr(node, "hit_count", 0)),
            int(getattr(node, "rrmc_prefix_tokens", len(node.value))),
            int(getattr(node, "rrmc_block_tokens", len(node.value))),
            float(node.last_access_time),
        )

    def _policy_priority(
        self, node: TreeNode, policy: str, *, kind: str = "full"
    ) -> tuple[float, ...]:
        if policy == "value_aware":
            return self._value_aware_priority(node)

        accepted_hit_count = float(getattr(node, "accepted_hit_count", 0))
        accepted_mamba_hit_count = float(
            getattr(node, "accepted_mamba_hit_count", 0)
        )
        last_access = float(node.last_access_time)

        if policy == "lfu":
            # RRMC LFU should reflect realized reuse, not lookup attempts.
            if kind == "mamba":
                return (accepted_mamba_hit_count, last_access)
            return (accepted_hit_count, last_access)
        if policy == "gdsf":
            priority = float(
                getattr(node, self._rrmc_priority_attr("gdsf", kind=kind), 0.0)
            )
            if priority <= 0:
                priority = (
                    float(
                        getattr(
                            self, self._rrmc_clock_attr("gdsf", kind=kind), 0.0
                        )
                    )
                    + self._rrmc_frequency(node, kind=kind)
                    * self._rrmc_gdsf_cost(node, kind=kind)
                    / self._rrmc_policy_size(node, kind=kind)
                )
            return (priority, last_access)
        if policy == "pgdsf":
            priority = float(
                getattr(node, self._rrmc_priority_attr("pgdsf", kind=kind), 0.0)
            )
            if priority <= 0:
                priority = (
                    float(
                        getattr(
                            self, self._rrmc_clock_attr("pgdsf", kind=kind), 0.0
                        )
                    )
                    + self._rrmc_frequency(node, kind=kind)
                    * self._rrmc_pgdsf_cost_per_size(node, kind=kind)
                )
            return (priority, last_access)

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
        return min(
            candidates,
            key=lambda node: self._policy_priority(node, policy, kind="full"),
        )

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
        return min(
            candidates,
            key=lambda node: self._policy_priority(node, policy, kind="mamba"),
        )

    def _evict_full_custom(self, num_tokens: int, policy: str) -> int:
        full_num_evicted = 0
        while full_num_evicted < num_tokens:
            node = self._select_full_candidate(policy)
            if node is None:
                break
            full_priority = self._policy_priority(node, policy, kind="full")[0]
            mamba_priority = self._policy_priority(node, policy, kind="mamba")[0]
            full_num_evicted_delta, _, _, _ = self._evict_leaf_node(node, False)
            full_num_evicted += full_num_evicted_delta
            if full_num_evicted_delta > 0:
                self._advance_rrmc_clock(
                    policy, kind="full", evicted_priority=full_priority
                )
                self._advance_rrmc_clock(
                    policy, kind="mamba", evicted_priority=mamba_priority
                )
        return full_num_evicted

    def _evict_mamba_custom(self, mamba_num: int, policy: str) -> int:
        mamba_num_evicted = 0
        while mamba_num_evicted < mamba_num:
            node = self._select_mamba_candidate(policy)
            if node is None:
                break
            assert node.mamba_value is not None, f"node has no mamba value, {node.id=}"
            mamba_priority = self._policy_priority(node, policy, kind="mamba")[0]
            full_priority = self._policy_priority(node, policy, kind="full")[0]
            if len(node.children) > 0:
                self.req_to_token_pool.mamba_pool.free(node.mamba_value)
                self.mamba_lru_list.remove_node(node)
                self._tombstone_internal_node(node)
                mamba_num_evicted += 1
                self._advance_rrmc_clock(
                    policy, kind="mamba", evicted_priority=mamba_priority
                )
            else:
                full_delta, delta, _, _ = self._evict_leaf_node(node, True)
                mamba_num_evicted += delta
                if full_delta > 0:
                    self._advance_rrmc_clock(
                        policy, kind="full", evicted_priority=full_priority
                    )
                if delta > 0:
                    self._advance_rrmc_clock(
                        policy, kind="mamba", evicted_priority=mamba_priority
                    )
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
