from __future__ import annotations

import argparse
from pathlib import Path

from sglang.srt.configs.mamba_utils import Mamba2CacheParams

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
    parser.add_argument("--linear-layer-count", type=int, default=3)
    parser.add_argument("--full-layer-count", type=int, default=1)
    parser.add_argument(
        "--linear-recompute-counts",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--target-linear-layer-ids",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--target-full-layer-ids",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--prefix-lens", type=int, nargs="+", default=[0, 4096])
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
        help="Optional explicit layer types, e.g. linear_attention full_attention ...",
    )
    parser.add_argument(
        "--linear-conv-kernel-dim", type=int, default=DEFAULT_QWEN35_LINEAR_CONV_KERNEL_DIM
    )
    parser.add_argument(
        "--linear-num-key-heads", type=int, default=DEFAULT_QWEN35_LINEAR_NUM_KEY_HEADS
    )
    parser.add_argument(
        "--linear-num-value-heads", type=int, default=DEFAULT_QWEN35_LINEAR_NUM_VALUE_HEADS
    )
    parser.add_argument(
        "--linear-key-head-dim", type=int, default=DEFAULT_QWEN35_LINEAR_KEY_HEAD_DIM
    )
    parser.add_argument(
        "--linear-value-head-dim", type=int, default=DEFAULT_QWEN35_LINEAR_VALUE_HEAD_DIM
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


def select_recovery_layers(
    linear_layer_ids: list[int],
    full_layer_ids: list[int],
    *,
    group_index: int,
    linear_layer_count: int,
    full_layer_count: int,
    target_linear_layer_ids: list[int] | None,
    target_full_layer_ids: list[int] | None,
) -> tuple[list[int], list[int]]:
    linear_set = set(linear_layer_ids)
    full_set = set(full_layer_ids)

    if target_linear_layer_ids is not None or target_full_layer_ids is not None:
        if target_linear_layer_ids is None or target_full_layer_ids is None:
            raise ValueError(
                "Please pass both --target-linear-layer-ids and --target-full-layer-ids together."
            )
        if not all(layer_id in linear_set for layer_id in target_linear_layer_ids):
            raise ValueError("Some target linear layer ids are not linear-attention layers.")
        if not all(layer_id in full_set for layer_id in target_full_layer_ids):
            raise ValueError("Some target full layer ids are not full-attention layers.")
        return sorted(target_linear_layer_ids), sorted(target_full_layer_ids)

    if group_index < 0 or group_index + full_layer_count > len(full_layer_ids):
        raise ValueError(
            f"group-index {group_index} with full-layer-count {full_layer_count} "
            f"exceeds available full-attention groups ({len(full_layer_ids)} full layers)."
        )

    selected_full_layer_ids = full_layer_ids[
        group_index : group_index + full_layer_count
    ]
    candidate_linear_layer_ids = [
        layer_id for layer_id in linear_layer_ids if layer_id < selected_full_layer_ids[0]
    ]
    if len(candidate_linear_layer_ids) < linear_layer_count:
        raise ValueError(
            "Not enough linear-attention layers before the selected full-attention group."
        )
    selected_linear_layer_ids = candidate_linear_layer_ids[-linear_layer_count:]
    return selected_linear_layer_ids, selected_full_layer_ids


def normalize_recompute_counts(
    linear_recompute_counts: list[int] | None, linear_layer_count: int
) -> list[int]:
    if linear_recompute_counts is None:
        return list(range(linear_layer_count + 1))
    counts = sorted(set(linear_recompute_counts))
    for count in counts:
        if count < 0 or count > linear_layer_count:
            raise ValueError(
                f"Invalid recompute count {count}. Valid range is [0, {linear_layer_count}]."
            )
    return counts


def build_subset_mamba_cache_params(
    base_params: Mamba2CacheParams,
    linear_layer_ids: list[int],
) -> Mamba2CacheParams:
    return Mamba2CacheParams(
        shape=base_params.shape,
        dtype=base_params.dtype,
        layers=linear_layer_ids,
    )
