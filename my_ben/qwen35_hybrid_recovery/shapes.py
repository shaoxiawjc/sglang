from __future__ import annotations

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention

from .utils import dtype_name


def shape_list(*dims: int) -> list[int]:
    return [int(dim) for dim in dims]


def kv_transfer_shape_info(
    token_count: int,
    head_num: int,
    head_dim: int,
    layer_count: int,
) -> dict[str, object]:
    return {
        "token_count": token_count,
        "layer_count": layer_count,
        "head_num": head_num,
        "head_dim": head_dim,
        "per_layer_k_shape": shape_list(token_count, head_num, head_dim),
        "per_layer_v_shape": shape_list(token_count, head_num, head_dim),
    }


def state_shape_info(
    cache_params: Mamba2CacheParams,
    slot_count: int,
    linear_layer_count: int,
) -> dict[str, object]:
    conv_shapes = [
        shape_list(linear_layer_count, slot_count, *conv_shape)
        for conv_shape in cache_params.shape.conv
    ]
    temporal_shape = shape_list(
        linear_layer_count, slot_count, *cache_params.shape.temporal
    )
    return {
        "slot_count": slot_count,
        "linear_layer_count": linear_layer_count,
        "conv_shapes": conv_shapes,
        "temporal_shape": temporal_shape,
        "conv_dtype": dtype_name(cache_params.dtype.conv),
        "temporal_dtype": dtype_name(cache_params.dtype.temporal),
    }


def linear_attention_shape_info(
    layer: RadixLinearAttention,
    batch_size: int,
    seq_len: int,
) -> dict[str, object]:
    total_tokens = batch_size * seq_len
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "mixed_qkv_shape": shape_list(
            total_tokens, layer.q_dim + layer.k_dim + layer.v_dim
        ),
        "a_shape": shape_list(total_tokens, layer.num_v_heads),
        "b_shape": shape_list(total_tokens, layer.num_v_heads),
        "q_shape": shape_list(1, total_tokens, layer.num_q_heads, layer.head_q_dim),
        "k_shape": shape_list(1, total_tokens, layer.num_k_heads, layer.head_k_dim),
        "v_shape": shape_list(1, total_tokens, layer.num_v_heads, layer.head_v_dim),
    }


def full_attention_shape_info(
    config: Qwen3_5TextConfig,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
) -> dict[str, object]:
    total_tokens = batch_size * seq_len
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "prefix_len": prefix_len,
        "q_shape": shape_list(
            total_tokens, config.num_attention_heads, config.head_dim
        ),
        "k_shape": shape_list(
            total_tokens, config.num_key_value_heads, config.head_dim
        ),
        "v_shape": shape_list(
            total_tokens, config.num_key_value_heads, config.head_dim
        ),
        "prefix_kv_shape": shape_list(
            batch_size * prefix_len, config.num_key_value_heads, config.head_dim
        ),
    }
