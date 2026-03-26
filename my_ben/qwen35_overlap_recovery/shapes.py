from __future__ import annotations

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig

from my_ben.qwen35_hybrid_recovery.utils import dtype_name


def shape_list(*dims: int) -> list[int]:
    return [int(dim) for dim in dims]


def strategy_shape_info(
    *,
    config: Qwen3_5TextConfig,
    cache_params: Mamba2CacheParams,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    linear_layer_count: int,
    linear_recompute_count: int,
) -> dict[str, object]:
    total_len = prefix_len + seq_len
    total_tokens = batch_size * total_len
    current_tokens = batch_size * seq_len
    prefix_tokens = batch_size * prefix_len
    linear_onload_count = linear_layer_count - linear_recompute_count
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "prefix_len": prefix_len,
        "total_len": total_len,
        "linear_layer_count": linear_layer_count,
        "linear_recompute_count": linear_recompute_count,
        "linear_onload_count": linear_onload_count,
        "causal_layer_count": 1,
        "linear_full_hidden_shape": shape_list(total_tokens, config.hidden_size),
        "linear_current_hidden_shape": shape_list(current_tokens, config.hidden_size),
        "linear_state_conv_shapes": [
            shape_list(linear_onload_count, batch_size, *conv_shape)
            for conv_shape in cache_params.shape.conv
        ],
        "linear_state_conv_dtype": dtype_name(cache_params.dtype.conv),
        "linear_state_temporal_shape": shape_list(
            linear_onload_count,
            batch_size,
            *cache_params.shape.temporal,
        ),
        "linear_state_temporal_dtype": dtype_name(cache_params.dtype.temporal),
        "causal_full_hidden_shape": shape_list(total_tokens, config.hidden_size),
        "causal_current_hidden_shape": shape_list(current_tokens, config.hidden_size),
        "causal_prefix_k_shape": shape_list(
            prefix_tokens, config.num_key_value_heads, config.head_dim
        ),
        "causal_prefix_v_shape": shape_list(
            prefix_tokens, config.num_key_value_heads, config.head_dim
        ),
    }
