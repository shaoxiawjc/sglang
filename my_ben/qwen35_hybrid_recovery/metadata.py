from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig

from .shapes import state_shape_info
from .utils import dtype_name, git_commit


def build_metadata(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    stamp: str,
    config: Qwen3_5TextConfig,
    model_dtype: torch.dtype,
    raw_config: dict,
    full_layer_ids: list[int],
    linear_layer_ids: list[int],
    mamba_cache_params: Mamba2CacheParams,
) -> dict[str, object]:
    return {
        "timestamp_utc": stamp,
        "model_path": str(args.model_path),
        "device_name": torch.cuda.get_device_name(0),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "git_commit": git_commit(repo_root),
        "model_dtype": dtype_name(model_dtype),
        "config_summary": {
            "num_hidden_layers": config.num_hidden_layers,
            "full_attention_layer_ids": full_layer_ids,
            "linear_layer_ids": linear_layer_ids,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "linear_num_key_heads": config.linear_num_key_heads,
            "linear_num_value_heads": config.linear_num_value_heads,
            "linear_key_head_dim": config.linear_key_head_dim,
            "linear_value_head_dim": config.linear_value_head_dim,
            "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
            "mamba_state_dtype": {
                "conv": dtype_name(mamba_cache_params.dtype.conv),
                "temporal": dtype_name(mamba_cache_params.dtype.temporal),
            },
            "mamba_cache_per_req_bytes": mamba_cache_params.mamba_cache_per_req,
        },
        "shape_summary": {
            "kvcache_per_full_layer": {
                "head_num": config.num_key_value_heads,
                "head_dim": config.head_dim,
                "k_shape_template": [
                    "token_count",
                    config.num_key_value_heads,
                    config.head_dim,
                ],
                "v_shape_template": [
                    "token_count",
                    config.num_key_value_heads,
                    config.head_dim,
                ],
            },
            "state_cache": state_shape_info(
                mamba_cache_params,
                slot_count=1,
                linear_layer_count=1,
            ),
        },
        "raw_config_model_type": raw_config["model_type"],
        "effective_text_config": raw_config["text_config"],
        "benchmark_args": vars(args),
    }
