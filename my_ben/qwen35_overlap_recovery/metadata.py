from __future__ import annotations

from pathlib import Path

import torch

from my_ben.qwen35_hybrid_recovery.utils import git_commit, make_jsonable


def build_metadata(
    *,
    args,
    repo_root: Path,
    stamp: str,
    config,
    model_dtype: torch.dtype,
    raw_config: dict,
    linear_layer_ids: list[int],
    causal_layer_id: int,
    recompute_counts: list[int],
    cache_params,
):
    return make_jsonable(
        {
            "timestamp": stamp,
            "git_commit": git_commit(repo_root),
            "model_path": args.model_path,
            "device_name": torch.cuda.get_device_name(0),
            "cuda_device_count": torch.cuda.device_count(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "model_dtype": str(model_dtype),
            "group_summary": {
                "assumption": "3 linear-attention blocks : 1 causal-attention block",
                "group_index": args.group_index,
                "linear_layer_ids": linear_layer_ids,
                "causal_layer_id": causal_layer_id,
                "linear_recompute_counts": recompute_counts,
            },
            "args": vars(args),
            "effective_text_config": config.to_dict(),
            "raw_text_config": raw_config["text_config"],
            "mamba_cache_params": {
                "layers": cache_params.layers,
                "conv_shapes": cache_params.shape.conv,
                "temporal_shape": cache_params.shape.temporal,
                "conv_dtype": str(cache_params.dtype.conv),
                "temporal_dtype": str(cache_params.dtype.temporal),
            },
            "strategies": [
                "baseline_recompute",
                "baseline_offload_onload",
                "ours_ca_recompute_overlap_la_state_conv",
                "ours_la_recompute_overlap_ca_kvcache",
            ],
        }
    )
