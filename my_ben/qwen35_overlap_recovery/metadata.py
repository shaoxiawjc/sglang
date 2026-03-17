from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig

from my_ben.qwen35_hybrid_recovery.utils import dtype_name, git_commit


def build_metadata(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    stamp: str,
    config: Qwen3_5TextConfig,
    model_dtype: torch.dtype,
    raw_config: dict,
    target_linear_layer_ids: list[int],
    target_full_layer_ids: list[int],
    recompute_counts: list[int],
    cache_params: Mamba2CacheParams,
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
        "target_group": {
            "group_index": args.group_index,
            "linear_layer_ids": target_linear_layer_ids,
            "full_layer_ids": target_full_layer_ids,
            "linear_layer_count": len(target_linear_layer_ids),
            "full_layer_count": len(target_full_layer_ids),
            "linear_recompute_counts": recompute_counts,
        },
        "config_summary": {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "linear_num_key_heads": config.linear_num_key_heads,
            "linear_num_value_heads": config.linear_num_value_heads,
            "linear_key_head_dim": config.linear_key_head_dim,
            "linear_value_head_dim": config.linear_value_head_dim,
            "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
            "mamba_state_dtype": {
                "conv": dtype_name(cache_params.dtype.conv),
                "temporal": dtype_name(cache_params.dtype.temporal),
            },
        },
        "effective_text_config": raw_config["text_config"],
        "benchmark_args": vars(args),
    }
