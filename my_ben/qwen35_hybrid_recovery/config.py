from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
    mamba2_state_dtype,
)
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/wjc/resources/models/qwen3_5_9b"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "my_ben" / "results" / "qwen35_hybrid_recovery",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=30)
    parser.add_argument(
        "--kv-token-counts",
        type=int,
        nargs="+",
        default=[64, 256, 1024, 4096, 8192],
    )
    parser.add_argument(
        "--state-slot-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    parser.add_argument(
        "--linear-batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16],
    )
    parser.add_argument(
        "--linear-seq-lens",
        type=int,
        nargs="+",
        default=[128, 512, 2048],
    )
    parser.add_argument(
        "--full-seq-lens",
        type=int,
        nargs="+",
        default=[128, 512, 2048],
    )
    parser.add_argument(
        "--full-prefix-lens",
        type=int,
        nargs="+",
        default=[0, 4096],
    )
    parser.add_argument("--num-hidden-layers", type=int, default=None)
    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--num-key-value-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--full-attention-interval", type=int, default=None)
    parser.add_argument(
        "--layer-types",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit layer types, e.g. linear_attention full_attention ...",
    )
    parser.add_argument("--linear-conv-kernel-dim", type=int, default=None)
    parser.add_argument("--linear-num-key-heads", type=int, default=None)
    parser.add_argument("--linear-num-value-heads", type=int, default=None)
    parser.add_argument("--linear-key-head-dim", type=int, default=None)
    parser.add_argument("--linear-value-head-dim", type=int, default=None)
    parser.add_argument(
        "--model-dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default=None,
    )
    parser.add_argument(
        "--mamba-conv-dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default=None,
    )
    parser.add_argument(
        "--mamba-ssm-dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default=None,
    )
    parser.add_argument(
        "--linear-attn-backend",
        type=str,
        choices=["triton", "cutedsl", "flashinfer"],
        default="flashinfer",
    )
    parser.add_argument(
        "--linear-attn-decode-backend",
        type=str,
        choices=["triton", "cutedsl", "flashinfer"],
        default=None,
    )
    parser.add_argument(
        "--linear-attn-prefill-backend",
        type=str,
        choices=["triton", "cutedsl", "flashinfer"],
        default=None,
    )
    return parser.parse_args()


def load_text_config(model_path: Path) -> tuple[Qwen3_5TextConfig, torch.dtype, dict]:
    config_path = model_path / "config.json"
    config_dict = json.loads(config_path.read_text())
    text_cfg = Qwen3_5TextConfig(**config_dict["text_config"])
    model_dtype = DTYPE_MAP[config_dict["text_config"]["dtype"]]
    return text_cfg, model_dtype, config_dict


def derive_layer_ids(config_dict: dict) -> tuple[list[int], list[int]]:
    layer_types = config_dict["text_config"]["layer_types"]
    linear_layer_ids = [
        idx for idx, layer_type in enumerate(layer_types) if layer_type == "linear_attention"
    ]
    full_layer_ids = [
        idx for idx, layer_type in enumerate(layer_types) if layer_type == "full_attention"
    ]
    return linear_layer_ids, full_layer_ids


def build_local_mamba_cache_params(
    config: Qwen3_5TextConfig,
    linear_layer_ids: list[int],
    dtype_override: Mamba2StateDType | None = None,
) -> Mamba2CacheParams:
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=config.linear_value_head_dim * config.linear_num_value_heads,
        n_groups=config.linear_num_key_heads,
        num_heads=config.linear_num_value_heads,
        head_dim=config.linear_value_head_dim,
        state_size=config.linear_key_head_dim,
        conv_kernel=config.linear_conv_kernel_dim,
    )
    return Mamba2CacheParams(
        shape=shape,
        layers=linear_layer_ids,
        dtype=dtype_override or mamba2_state_dtype(config),
    )


def resolve_mamba_state_dtype(
    args: argparse.Namespace, config: Qwen3_5TextConfig
) -> Mamba2StateDType:
    default_dtype = mamba2_state_dtype(config)
    return Mamba2StateDType(
        conv=DTYPE_MAP[args.mamba_conv_dtype]
        if args.mamba_conv_dtype
        else default_dtype.conv,
        temporal=DTYPE_MAP[args.mamba_ssm_dtype]
        if args.mamba_ssm_dtype
        else default_dtype.temporal,
    )


def resolve_layer_types(
    args: argparse.Namespace, config: Qwen3_5TextConfig, raw_config: dict
) -> list[str]:
    if args.layer_types is not None:
        return args.layer_types

    raw_layer_types = raw_config["text_config"].get("layer_types")
    target_num_layers = args.num_hidden_layers or config.num_hidden_layers
    if raw_layer_types is not None and len(raw_layer_types) == target_num_layers:
        return raw_layer_types

    interval = (
        args.full_attention_interval
        or raw_config["text_config"].get("full_attention_interval")
        or getattr(config, "full_attention_interval", None)
    )
    if interval is None:
        raise ValueError(
            "Unable to resolve layer types. Please pass --layer-types or --full-attention-interval."
        )
    return [
        "full_attention" if (layer_id + 1) % interval == 0 else "linear_attention"
        for layer_id in range(target_num_layers)
    ]


def apply_model_config_overrides(
    args: argparse.Namespace,
    config: Qwen3_5TextConfig,
    model_dtype: torch.dtype,
    raw_config: dict,
) -> tuple[Qwen3_5TextConfig, torch.dtype, dict]:
    override_map = {
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "head_dim": args.head_dim,
        "linear_conv_kernel_dim": args.linear_conv_kernel_dim,
        "linear_num_key_heads": args.linear_num_key_heads,
        "linear_num_value_heads": args.linear_num_value_heads,
        "linear_key_head_dim": args.linear_key_head_dim,
        "linear_value_head_dim": args.linear_value_head_dim,
        "full_attention_interval": args.full_attention_interval,
    }
    for key, value in override_map.items():
        if value is not None:
            setattr(config, key, value)
            raw_config["text_config"][key] = value

    if args.model_dtype is not None:
        model_dtype = DTYPE_MAP[args.model_dtype]
        raw_config["text_config"]["dtype"] = args.model_dtype

    layer_types = resolve_layer_types(args, config, raw_config)
    config.num_hidden_layers = len(layer_types)
    raw_config["text_config"]["layer_types"] = layer_types
    raw_config["text_config"]["num_hidden_layers"] = len(layer_types)

    if args.full_attention_interval is not None:
        config.full_attention_interval = args.full_attention_interval
    elif "full_attention_interval" in raw_config["text_config"]:
        config.full_attention_interval = raw_config["text_config"]["full_attention_interval"]

    return config, model_dtype, raw_config
