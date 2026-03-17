from __future__ import annotations

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig


def shape_list(*dims: int) -> list[int]:
    return [int(dim) for dim in dims]


def linear_block_shape_info(
    config: Qwen3_5TextConfig,
    cache_params: Mamba2CacheParams,
    *,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    replay_total_len: int,
) -> dict[str, object]:
    total_tokens = batch_size * seq_len
    replay_tokens = batch_size * replay_total_len
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "prefix_len": prefix_len,
        "replay_total_len": replay_total_len,
        "hidden_shape_current": shape_list(total_tokens, config.hidden_size),
        "hidden_shape_replay": shape_list(replay_tokens, config.hidden_size),
        "mixed_qkv_shape_current": shape_list(total_tokens, key_dim * 2 + value_dim),
        "z_shape_current": shape_list(
            total_tokens, config.linear_num_value_heads, config.linear_value_head_dim
        ),
        "a_shape_current": shape_list(total_tokens, config.linear_num_value_heads),
        "b_shape_current": shape_list(total_tokens, config.linear_num_value_heads),
        "state_conv_shapes": [
            shape_list(1, batch_size, *conv_shape) for conv_shape in cache_params.shape.conv
        ],
        "state_temporal_shape": shape_list(1, batch_size, *cache_params.shape.temporal),
    }


def full_block_shape_info(
    config: Qwen3_5TextConfig,
    *,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    replay_total_len: int,
) -> dict[str, object]:
    total_tokens = batch_size * seq_len
    replay_tokens = batch_size * replay_total_len
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "prefix_len": prefix_len,
        "replay_total_len": replay_total_len,
        "hidden_shape_current": shape_list(total_tokens, config.hidden_size),
        "hidden_shape_replay": shape_list(replay_tokens, config.hidden_size),
        "q_shape_current": shape_list(
            total_tokens, config.num_attention_heads, config.head_dim
        ),
        "k_shape_current": shape_list(
            total_tokens, config.num_key_value_heads, config.head_dim
        ),
        "v_shape_current": shape_list(
            total_tokens, config.num_key_value_heads, config.head_dim
        ),
        "prefix_kv_shape": shape_list(
            batch_size * prefix_len, config.num_key_value_heads, config.head_dim
        ),
    }
