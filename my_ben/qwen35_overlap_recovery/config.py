from __future__ import annotations

import argparse
from pathlib import Path

from my_ben.qwen35_hybrid_recovery.config import (
    DTYPE_MAP,
    DEFAULT_QWEN35_FULL_ATTENTION_INTERVAL,
    DEFAULT_QWEN35_HEAD_DIM,
    DEFAULT_QWEN35_LAYER_TYPES,
    DEFAULT_QWEN35_LINEAR_CONV_KERNEL_DIM,
    DEFAULT_QWEN35_LINEAR_KEY_HEAD_DIM,
    DEFAULT_QWEN35_LINEAR_NUM_KEY_HEADS,
    DEFAULT_QWEN35_LINEAR_NUM_VALUE_HEADS,
    DEFAULT_QWEN35_LINEAR_VALUE_HEAD_DIM,
    DEFAULT_QWEN35_MAMBA_SSM_DTYPE,
    DEFAULT_QWEN35_MODEL_DTYPE,
    DEFAULT_QWEN35_NUM_ATTENTION_HEADS,
    DEFAULT_QWEN35_NUM_HIDDEN_LAYERS,
    DEFAULT_QWEN35_NUM_KEY_VALUE_HEADS,
    apply_model_config_overrides,
    build_local_mamba_cache_params,
    derive_layer_ids,
    load_text_config,
    resolve_mamba_state_dtype,
)


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
        default=repo_root / "my_ben" / "results" / "qwen35_overlap_recovery",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=30)
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument(
        "--linear-layer-count",
        type=int,
        default=3,
        help="Number of LA blocks in the tested group. The current design assumes 3.",
    )
    parser.add_argument(
        "--causal-layer-count",
        type=int,
        default=1,
        help="Number of CA blocks in the tested group. The current design assumes 1.",
    )
    parser.add_argument(
        "--linear-recompute-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="LA-first overlap cases: how many LA blocks are fully recomputed.",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--prefix-lens", type=int, nargs="+", default=[0, 4096, 8192])
    parser.add_argument(
        "--num-hidden-layers", type=int, default=DEFAULT_QWEN35_NUM_HIDDEN_LAYERS
    )
    parser.add_argument(
        "--num-attention-heads", type=int, default=DEFAULT_QWEN35_NUM_ATTENTION_HEADS
    )
    parser.add_argument(
        "--num-key-value-heads", type=int, default=DEFAULT_QWEN35_NUM_KEY_VALUE_HEADS
    )
    parser.add_argument("--head-dim", type=int, default=DEFAULT_QWEN35_HEAD_DIM)
    parser.add_argument(
        "--full-attention-interval",
        type=int,
        default=DEFAULT_QWEN35_FULL_ATTENTION_INTERVAL,
    )
    parser.add_argument(
        "--layer-types",
        type=str,
        nargs="+",
        default=DEFAULT_QWEN35_LAYER_TYPES,
    )
    parser.add_argument(
        "--linear-conv-kernel-dim",
        type=int,
        default=DEFAULT_QWEN35_LINEAR_CONV_KERNEL_DIM,
    )
    parser.add_argument(
        "--linear-num-key-heads",
        type=int,
        default=DEFAULT_QWEN35_LINEAR_NUM_KEY_HEADS,
    )
    parser.add_argument(
        "--linear-num-value-heads",
        type=int,
        default=DEFAULT_QWEN35_LINEAR_NUM_VALUE_HEADS,
    )
    parser.add_argument(
        "--linear-key-head-dim",
        type=int,
        default=DEFAULT_QWEN35_LINEAR_KEY_HEAD_DIM,
    )
    parser.add_argument(
        "--linear-value-head-dim",
        type=int,
        default=DEFAULT_QWEN35_LINEAR_VALUE_HEAD_DIM,
    )
    parser.add_argument(
        "--model-dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default=DEFAULT_QWEN35_MODEL_DTYPE,
    )
    parser.add_argument(
        "--mamba-conv-dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default=None,
    )
    parser.add_argument(
        "--mamba-ssm-dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default=DEFAULT_QWEN35_MAMBA_SSM_DTYPE,
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


def resolve_block_group(
    linear_layer_ids: list[int],
    full_layer_ids: list[int],
    *,
    group_index: int,
    linear_layer_count: int,
    causal_layer_count: int,
) -> tuple[list[int], int]:
    if linear_layer_count != 3 or causal_layer_count != 1:
        raise ValueError("The current overlap benchmark requires a fixed 3 LA : 1 CA ratio.")
    if group_index < 0 or group_index >= len(full_layer_ids):
        raise ValueError(
            f"group-index {group_index} exceeds available CA groups ({len(full_layer_ids)})."
        )

    causal_layer_id = full_layer_ids[group_index]
    candidate_linear_layer_ids = [
        layer_id for layer_id in linear_layer_ids if layer_id < causal_layer_id
    ]
    if len(candidate_linear_layer_ids) < linear_layer_count:
        raise ValueError(
            "Not enough LA layers before the selected CA block to form a 3:1 group."
        )
    return candidate_linear_layer_ids[-linear_layer_count:], causal_layer_id


def normalize_recompute_counts(
    linear_recompute_counts: list[int] | None,
    linear_layer_count: int,
) -> list[int]:
    if linear_recompute_counts is None:
        return list(range(1, linear_layer_count + 1))
    counts = sorted(set(linear_recompute_counts))
    for count in counts:
        if count < 1 or count > linear_layer_count:
            raise ValueError(
                f"Invalid recompute count {count}. Valid range is [1, {linear_layer_count}]."
            )
    return counts


def resolve_model_and_group(args: argparse.Namespace):
    config, model_dtype, raw_config = load_text_config(args.model_path)
    config, model_dtype, raw_config = apply_model_config_overrides(
        args, config, model_dtype, raw_config
    )
    linear_layer_ids, full_layer_ids = derive_layer_ids(raw_config)
    target_linear_layer_ids, target_causal_layer_id = resolve_block_group(
        linear_layer_ids,
        full_layer_ids,
        group_index=args.group_index,
        linear_layer_count=args.linear_layer_count,
        causal_layer_count=args.causal_layer_count,
    )
    recompute_counts = normalize_recompute_counts(
        args.linear_recompute_counts,
        args.linear_layer_count,
    )
    cache_params = build_local_mamba_cache_params(
        config,
        target_linear_layer_ids,
        dtype_override=resolve_mamba_state_dtype(args, config),
    )
    return (
        config,
        model_dtype,
        raw_config,
        target_linear_layer_ids,
        target_causal_layer_id,
        recompute_counts,
        cache_params,
    )
