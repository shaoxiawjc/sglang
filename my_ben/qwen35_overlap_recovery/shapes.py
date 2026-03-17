from __future__ import annotations

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig

from my_ben.qwen35_hybrid_recovery.utils import dtype_name


def shape_list(*dims: int) -> list[int]:
    return [int(dim) for dim in dims]


def overlap_shape_info(
    *,
    config: Qwen3_5TextConfig,
    cache_params: Mamba2CacheParams,
    linear_layer_ids: list[int],
    full_layer_ids: list[int],
    batch_size: int,
    seq_len: int,
    prefix_len: int,
) -> dict[str, object]:
    total_len = prefix_len + seq_len
    total_recompute_tokens = batch_size * total_len
    current_tokens = batch_size * seq_len
    prefix_tokens = batch_size * prefix_len
    return {
        "linear_layer_ids": linear_layer_ids,
        "full_layer_ids": full_layer_ids,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "prefix_len": prefix_len,
        "total_len_per_request": total_len,
        "linear_layer_count": len(linear_layer_ids),
        "full_layer_count": len(full_layer_ids),
        "linear_recompute_tokens_total": total_recompute_tokens,
        "linear_current_forward_tokens_total": current_tokens,
        "full_recompute_tokens_total": total_recompute_tokens,
        "full_prefix_prefetch_tokens_total": prefix_tokens * len(full_layer_ids),
        "hidden_shape_recompute": shape_list(total_recompute_tokens, config.hidden_size),
        "hidden_shape_current": shape_list(current_tokens, config.hidden_size),
        "full_qkv_shape_recompute": {
            "q": shape_list(
                total_recompute_tokens, config.num_attention_heads, config.head_dim
            ),
            "k": shape_list(
                total_recompute_tokens, config.num_key_value_heads, config.head_dim
            ),
            "v": shape_list(
                total_recompute_tokens, config.num_key_value_heads, config.head_dim
            ),
        },
        "full_qkv_shape_current": {
            "q": shape_list(current_tokens, config.num_attention_heads, config.head_dim),
            "k": shape_list(current_tokens, config.num_key_value_heads, config.head_dim),
            "v": shape_list(current_tokens, config.num_key_value_heads, config.head_dim),
        },
        "full_prefix_kv_shape_per_layer": shape_list(
            prefix_tokens, config.num_key_value_heads, config.head_dim
        ),
        "state_slot_count": batch_size,
        "state_conv_shapes": [
            shape_list(len(linear_layer_ids), batch_size, *conv_shape)
            for conv_shape in cache_params.shape.conv
        ],
        "state_temporal_shape": shape_list(
            len(linear_layer_ids), batch_size, *cache_params.shape.temporal
        ),
        "state_conv_dtype": dtype_name(cache_params.dtype.conv),
        "state_temporal_dtype": dtype_name(cache_params.dtype.temporal),
        "full_kv_dtype": dtype_name(config.torch_dtype)
        if hasattr(config, "torch_dtype")
        else None,
    }
