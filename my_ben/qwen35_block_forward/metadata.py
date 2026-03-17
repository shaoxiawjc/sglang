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
    linear_layer_id: int,
    full_layer_id: int,
    cache_params,
):
    return make_jsonable(
        {
            "timestamp": stamp,
            "git_commit": git_commit(repo_root),
            "model_path": args.model_path,
            "model_dtype": str(model_dtype),
            "linear_layer_id": linear_layer_id,
            "full_layer_id": full_layer_id,
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
        }
    )
