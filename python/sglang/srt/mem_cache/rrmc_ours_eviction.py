from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Iterable, Literal, Optional

from sglang.srt.mem_cache.mamba_radix_cache import TreeNode

logger = logging.getLogger(__name__)

OursAction = Literal["full", "mamba", "host_full", "host_mamba"]


@dataclasses.dataclass
class OursNodeMeta:
    node_id: int
    prefix_len: int
    parent_prefix_len: int
    last_access_seq: int = 0


@dataclasses.dataclass
class _ScoredNode:
    node: TreeNode
    recency: float
    raw_efficiency: float
    efficiency: float
    score: float
    saved_flops: float
    memory_bytes: float


class RRMCOursCostEstimator:
    def __init__(self, cache: Any, model_config: Optional[Any]):
        self.cache = cache
        self.config = self._unwrap_config(model_config)

        self.hidden_size = float(self._config_get("hidden_size", 1) or 1)
        self.num_hidden_layers = int(self._config_get("num_hidden_layers", 0) or 0)
        self.num_full_attention_layers = self._count_full_attention_layers()
        self.num_gdn_layers = self._count_gdn_layers()
        if self.num_hidden_layers <= 0:
            self.num_hidden_layers = max(
                self.num_full_attention_layers + self.num_gdn_layers, 1
            )

        self.kv_bytes_per_token = self._estimate_kv_bytes_per_token()
        self.host_kv_bytes_per_token = self._estimate_host_kv_bytes_per_token()
        self.mamba_bytes_per_state = self._estimate_mamba_bytes_per_state()
        self.host_mamba_bytes_per_state = self._estimate_host_mamba_bytes_per_state()

    def saved_flops(self, prefix_len: int) -> float:
        length = max(0, int(prefix_len))
        d = self.hidden_size
        attn = self.num_full_attention_layers * (8 * length * d * d + 4 * length * length * d)
        gdn = self.num_gdn_layers * self._gdn_flops(length)
        mlp = self.num_hidden_layers * (16 * length * d * d)
        return float(attn + gdn + mlp)

    def node_memory_bytes(self, node: TreeNode, action: OursAction) -> float:
        kv_tokens = self._tensor_len(getattr(node, "value", None))
        host_kv_tokens = self._tensor_len(getattr(node, "host_value", None))
        mamba_slots = self._tensor_len(getattr(node, "mamba_value", None))
        host_mamba_slots = self._tensor_len(getattr(node, "mamba_host_value", None))

        if action == "full":
            total = kv_tokens * self.kv_bytes_per_token
            if len(node.children) == 0 and mamba_slots > 0:
                total += mamba_slots * self.mamba_bytes_per_state
            return float(total)

        if action == "mamba":
            if len(node.children) > 0:
                return float(mamba_slots * self.mamba_bytes_per_state)
            return float(
                kv_tokens * self.kv_bytes_per_token
                + mamba_slots * self.mamba_bytes_per_state
            )

        if action == "host_full":
            return float(
                host_kv_tokens * self.host_kv_bytes_per_token
                + host_mamba_slots * self.host_mamba_bytes_per_state
            )

        if action == "host_mamba":
            if len(node.children) > 0:
                return float(host_mamba_slots * self.host_mamba_bytes_per_state)
            return float(
                host_kv_tokens * self.host_kv_bytes_per_token
                + host_mamba_slots * self.host_mamba_bytes_per_state
            )

        return 0.0

    def _unwrap_config(self, model_config: Optional[Any]) -> Optional[Any]:
        config = getattr(model_config, "hf_config", model_config)
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            return get_text_config()
        return config

    def _config_get(self, name: str, default: Any = None) -> Any:
        if self.config is None:
            return default
        if isinstance(self.config, dict):
            return self.config.get(name, default)
        return getattr(self.config, name, default)

    def _count_full_attention_layers(self) -> int:
        layer_ids = self._config_get("full_attention_layer_ids")
        if layer_ids is not None:
            return len(layer_ids)
        block_types = self._config_get("layers_block_type")
        if block_types is not None:
            return sum(1 for block_type in block_types if block_type == "full_attention")
        return int(self.num_hidden_layers)

    def _count_gdn_layers(self) -> int:
        layer_ids = self._config_get("linear_layer_ids")
        if layer_ids is not None:
            return len(layer_ids)
        block_types = self._config_get("layers_block_type")
        if block_types is not None:
            return sum(1 for block_type in block_types if block_type == "linear_attention")
        return 0

    def _gdn_flops(self, length: int) -> float:
        d = self.hidden_size
        linear_num_key_heads = float(self._config_get("linear_num_key_heads", 0) or 0)
        linear_num_value_heads = float(self._config_get("linear_num_value_heads", 0) or 0)
        linear_key_head_dim = float(self._config_get("linear_key_head_dim", 0) or 0)
        linear_value_head_dim = float(self._config_get("linear_value_head_dim", 0) or 0)
        linear_conv_kernel_dim = float(self._config_get("linear_conv_kernel_dim", 0) or 0)

        d_k_total = linear_num_key_heads * linear_key_head_dim
        d_v_total = linear_num_value_heads * linear_value_head_dim
        return float(
            4 * length * d * (d_k_total + d_v_total + linear_num_value_heads)
            + 2 * length * linear_conv_kernel_dim * (2 * d_k_total + d_v_total)
            + 7 * length * linear_num_value_heads * linear_key_head_dim * linear_value_head_dim
            + 2 * length * d_v_total * d
        )

    def _estimate_kv_bytes_per_token(self) -> float:
        pool = self._device_kv_pool()
        size = self._pool_size_per_token(pool)
        if size > 0:
            return size
        host_size = self._estimate_host_kv_bytes_per_token()
        return host_size if host_size > 0 else 1.0

    def _estimate_host_kv_bytes_per_token(self) -> float:
        host_pool = getattr(self.cache, "full_kv_pool_host", None)
        return self._pool_size_per_token(host_pool) or 1.0

    def _estimate_mamba_bytes_per_state(self) -> float:
        mamba_pool = getattr(getattr(self.cache, "req_to_token_pool", None), "mamba_pool", None)
        size = self._mamba_pool_size_per_state(mamba_pool)
        if size > 0:
            return size
        return self._estimate_host_mamba_bytes_per_state() or 1.0

    def _estimate_host_mamba_bytes_per_state(self) -> float:
        host_pool = getattr(self.cache, "mamba_pool_host", None)
        size = self._pool_size_per_token(host_pool)
        return size if size > 0 else 1.0

    def _device_kv_pool(self) -> Optional[Any]:
        allocator = getattr(self.cache, "token_to_kv_pool_allocator", None)
        get_kvcache = getattr(allocator, "get_kvcache", None)
        pool = get_kvcache() if callable(get_kvcache) else getattr(self.cache, "kvcache", None)
        return getattr(pool, "full_kv_pool", pool)

    def _pool_size_per_token(self, pool: Optional[Any]) -> float:
        if pool is None:
            return 0.0
        size_per_token = getattr(pool, "size_per_token", None)
        if size_per_token is not None:
            return float(size_per_token)
        get_size_per_token = getattr(pool, "get_size_per_token", None)
        if callable(get_size_per_token):
            try:
                return float(get_size_per_token())
            except Exception:
                return 0.0
        get_kv_size_bytes = getattr(pool, "get_kv_size_bytes", None)
        size = getattr(pool, "size", None)
        if callable(get_kv_size_bytes) and size:
            try:
                return float(get_kv_size_bytes()) / float(size)
            except Exception:
                return 0.0
        return self._buffer_size_per_index(pool)

    def _buffer_size_per_index(self, pool: Any) -> float:
        total = 0.0
        for attr in ("k_buffer", "v_buffer"):
            buffer = getattr(pool, attr, None)
            if buffer is None:
                continue
            tensors = buffer if isinstance(buffer, (list, tuple)) else [buffer]
            for tensor in tensors:
                try:
                    if tensor.shape[0] > 0:
                        total += float(tensor[0].numel() * tensor.element_size())
                except Exception:
                    continue
        return total

    def _mamba_pool_size_per_state(self, pool: Optional[Any]) -> float:
        mamba_cache = getattr(pool, "mamba_cache", None)
        if mamba_cache is None:
            return 0.0
        total = 0.0
        for conv in getattr(mamba_cache, "conv", []) or []:
            total += self._state_tensor_size_per_index(conv)
        temporal = getattr(mamba_cache, "temporal", None)
        total += self._state_tensor_size_per_index(temporal)
        return total

    def _state_tensor_size_per_index(self, tensor: Any) -> float:
        if tensor is None:
            return 0.0
        try:
            if len(tensor.shape) < 2:
                return 0.0
            per_layer_shape = tensor.shape[2:]
            per_layer = math.prod(per_layer_shape) if per_layer_shape else 1
            return float(tensor.shape[0] * per_layer * tensor.element_size())
        except Exception:
            return 0.0

    def _tensor_len(self, tensor: Any) -> int:
        if tensor is None:
            return 0
        try:
            return len(tensor)
        except TypeError:
            return 1


class RRMCOursEvictionPolicy:
    def __init__(
        self,
        *,
        cache: Any,
        alpha: float,
        debug: bool,
        model_config: Optional[Any],
    ):
        self.cache = cache
        self.alpha = float(alpha)
        self.debug = bool(debug)
        self.cost_estimator = RRMCOursCostEstimator(cache, model_config)
        self.node_meta: dict[int, OursNodeMeta] = {}
        self.nodes: dict[int, TreeNode] = {}
        self.full_candidates: set[int] = set()
        self.mamba_candidates: set[int] = set()
        self.host_full_candidates: set[int] = set()
        self.host_mamba_candidates: set[int] = set()
        self.access_seq = 0

    def reset(self) -> None:
        self.node_meta.clear()
        self.nodes.clear()
        self.full_candidates.clear()
        self.mamba_candidates.clear()
        self.host_full_candidates.clear()
        self.host_mamba_candidates.clear()
        self.access_seq = 0

    def register_node(self, node: TreeNode) -> None:
        if node is getattr(self.cache, "root_node", None):
            return
        parent_prefix_len = self._prefix_len(getattr(node, "parent", None))
        prefix_len = self._prefix_len(node)
        if prefix_len < parent_prefix_len:
            prefix_len = parent_prefix_len + self._node_key_len(node)
        self.nodes[node.id] = node
        self.node_meta[node.id] = OursNodeMeta(
            node_id=node.id,
            prefix_len=prefix_len,
            parent_prefix_len=parent_prefix_len,
        )
        self.touch_node(node)
        self.on_full_candidate_maybe_changed(node)
        self.on_mamba_candidate_maybe_changed(node)
        self.on_host_full_candidate_maybe_changed(node)
        self.on_host_mamba_candidate_maybe_changed(node)

    def unregister_node(self, node: TreeNode) -> None:
        node_id = node.id
        self.node_meta.pop(node_id, None)
        self.nodes.pop(node_id, None)
        self.full_candidates.discard(node_id)
        self.mamba_candidates.discard(node_id)
        self.host_full_candidates.discard(node_id)
        self.host_mamba_candidates.discard(node_id)

    def touch_node(self, node: TreeNode) -> None:
        if node is getattr(self.cache, "root_node", None):
            return
        if node.id not in self.node_meta:
            self.register_node(node)
            return
        self.access_seq += 1
        self.node_meta[node.id].last_access_seq = self.access_seq

    def touch_path(self, node: TreeNode) -> None:
        path = []
        while node is not None and node is not getattr(self.cache, "root_node", None):
            path.append(node)
            node = node.parent
        for path_node in reversed(path):
            self.touch_node(path_node)

    def on_mamba_attached(self, node: TreeNode) -> None:
        self.on_mamba_candidate_maybe_changed(node)

    def on_mamba_detached(self, node: TreeNode) -> None:
        self.on_mamba_candidate_maybe_changed(node)

    def on_full_candidate_maybe_changed(self, node: TreeNode) -> None:
        self._update_candidate_set(
            self.full_candidates, node, getattr(node, "value", None) is not None
        )

    def on_mamba_candidate_maybe_changed(self, node: TreeNode) -> None:
        self._update_candidate_set(
            self.mamba_candidates, node, getattr(node, "mamba_value", None) is not None
        )

    def on_host_full_candidate_maybe_changed(self, node: TreeNode) -> None:
        self._update_candidate_set(
            self.host_full_candidates, node, getattr(node, "host_value", None) is not None
        )

    def on_host_mamba_candidate_maybe_changed(self, node: TreeNode) -> None:
        self._update_candidate_set(
            self.host_mamba_candidates,
            node,
            getattr(node, "mamba_host_value", None) is not None,
        )

    def on_host_mamba_attached(self, node: TreeNode) -> None:
        self.on_host_mamba_candidate_maybe_changed(node)

    def on_host_mamba_detached(self, node: TreeNode) -> None:
        self.on_host_mamba_candidate_maybe_changed(node)

    def select_full_candidate(self) -> Optional[TreeNode]:
        return self._select_candidate(
            self._iter_candidates(self.full_candidates),
            callback_name="_ours_is_full_candidate",
            action="full",
        )

    def select_mamba_candidate(self) -> Optional[TreeNode]:
        return self._select_candidate(
            self._iter_candidates(self.mamba_candidates),
            callback_name="_ours_is_mamba_candidate",
            action="mamba",
        )

    def select_host_full_candidate(self) -> Optional[TreeNode]:
        return self._select_candidate(
            self._iter_candidates(self.host_full_candidates),
            callback_name="_ours_is_host_full_candidate",
            action="host_full",
        )

    def select_host_mamba_candidate(self) -> Optional[TreeNode]:
        return self._select_candidate(
            self._iter_candidates(self.host_mamba_candidates),
            callback_name="_ours_is_host_mamba_candidate",
            action="host_mamba",
        )

    def _select_candidate(
        self,
        nodes: Iterable[TreeNode],
        *,
        callback_name: str,
        action: OursAction,
    ) -> Optional[TreeNode]:
        callback = getattr(self.cache, callback_name)
        candidates = [node for node in nodes if callback(node)]
        scored = self._score_candidates(candidates, action)
        if not scored:
            return None
        selected = min(
            scored,
            key=lambda item: (
                item.score,
                self.node_meta[item.node.id].last_access_seq,
                item.node.id,
            ),
        )
        if self.debug:
            self._log_selection(action, scored, selected)
        return selected.node

    def _score_candidates(
        self, candidates: list[TreeNode], action: OursAction
    ) -> list[_ScoredNode]:
        if not candidates:
            return []

        ordered_by_access = sorted(
            candidates,
            key=lambda node: (self.node_meta[node.id].last_access_seq, node.id),
        )
        denom = len(ordered_by_access) - 1
        recency_by_id = {
            node.id: (1.0 if denom == 0 else rank / denom)
            for rank, node in enumerate(ordered_by_access)
        }

        raw: list[tuple[TreeNode, float, float, float]] = []
        for node in candidates:
            meta = self.node_meta[node.id]
            saved_flops = max(
                0.0,
                self.cost_estimator.saved_flops(meta.prefix_len)
                - self.cost_estimator.saved_flops(meta.parent_prefix_len),
            )
            memory_bytes = self.cost_estimator.node_memory_bytes(node, action)
            raw_efficiency = saved_flops / memory_bytes if memory_bytes > 0 else 0.0
            raw.append((node, raw_efficiency, saved_flops, memory_bytes))

        min_eff = min(item[1] for item in raw)
        max_eff = max(item[1] for item in raw)
        same_eff = max_eff == min_eff
        scored = []
        for node, raw_efficiency, saved_flops, memory_bytes in raw:
            efficiency = 0.0 if same_eff else (raw_efficiency - min_eff) / (max_eff - min_eff)
            recency = recency_by_id[node.id]
            score = recency + self.alpha * efficiency
            scored.append(
                _ScoredNode(
                    node=node,
                    recency=recency,
                    raw_efficiency=raw_efficiency,
                    efficiency=efficiency,
                    score=score,
                    saved_flops=saved_flops,
                    memory_bytes=memory_bytes,
                )
            )
        return scored

    def _iter_candidates(self, candidate_ids: set[int]) -> Iterable[TreeNode]:
        stale_ids = []
        for node_id in list(candidate_ids):
            node = self.nodes.get(node_id)
            if node is None or node_id not in self.node_meta:
                stale_ids.append(node_id)
                continue
            yield node
        for node_id in stale_ids:
            candidate_ids.discard(node_id)

    def _update_candidate_set(
        self, candidate_ids: set[int], node: TreeNode, present: bool
    ) -> None:
        if node is getattr(self.cache, "root_node", None):
            return
        if present:
            if node.id not in self.node_meta:
                self.register_node(node)
            else:
                candidate_ids.add(node.id)
        else:
            candidate_ids.discard(node.id)

    def _prefix_len(self, node: Optional[TreeNode]) -> int:
        if node is None or node is getattr(self.cache, "root_node", None):
            return 0
        prefix_tokens = getattr(node, "rrmc_prefix_tokens", None)
        if prefix_tokens is not None:
            return int(prefix_tokens)
        meta = self.node_meta.get(node.id)
        if meta is not None:
            return meta.prefix_len
        return self._prefix_len(node.parent) + self._node_key_len(node)

    def _node_key_len(self, node: TreeNode) -> int:
        try:
            return len(node.key)
        except Exception:
            return 0

    def _log_selection(
        self, action: OursAction, scored: list[_ScoredNode], selected: _ScoredNode
    ) -> None:
        meta = self.node_meta[selected.node.id]
        logger.debug(
            "RRMC ours eviction action=%s alpha=%s candidates=%s selected=%s kind=%s "
            "prefix_len=%s parent_prefix_len=%s R=%.6f raw_efficiency=%.6f "
            "E=%.6f S=%.6f saved_flops=%.3f memory_bytes=%.3f",
            action,
            self.alpha,
            len(scored),
            selected.node.id,
            "leaf" if len(selected.node.children) == 0 else "internal",
            meta.prefix_len,
            meta.parent_prefix_len,
            selected.recency,
            selected.raw_efficiency,
            selected.efficiency,
            selected.score,
            selected.saved_flops,
            selected.memory_bytes,
        )
