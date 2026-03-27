from __future__ import annotations

import argparse
from pathlib import Path

from my_ben.qwen35_hybrid_recovery.config import (
    DEFAULT_QWEN35_FIRST_FULL_LAYER_ID,
    DEFAULT_QWEN35_FIRST_LINEAR_LAYER_ID,
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
    parse_args as _unused_hybrid_parse_args,
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
        default=repo_root / "my_ben" / "results" / "qwen35_block_forward",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=30)
    parser.add_argument(
        "--clear-cuda-cache-each-iter",
        action="store_true",
        help="Run gc.collect() and torch.cuda.empty_cache() after each warmup/bench iteration.",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--prefix-lens", type=int, nargs="+", default=[0, 4096])
    parser.add_argument("--linear-layer-id", type=int, default=DEFAULT_QWEN35_FIRST_LINEAR_LAYER_ID)
    parser.add_argument("--full-layer-id", type=int, default=DEFAULT_QWEN35_FIRST_FULL_LAYER_ID)

    hybrid_parser = _unused_hybrid_parse_args  # keep import usage explicit
    del hybrid_parser

    # Model override args kept aligned with qwen35_hybrid_recovery.
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
    parser.add_argument("--layer-types", type=str, nargs="+", default=DEFAULT_QWEN35_LAYER_TYPES)
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
        choices=["float16", "bfloat16", "float32"],
        default=DEFAULT_QWEN35_MODEL_DTYPE,
    )
    parser.add_argument(
        "--mamba-conv-dtype",
        choices=["float16", "bfloat16", "float32"],
        default=None,
    )
    parser.add_argument(
        "--mamba-ssm-dtype",
        choices=["float16", "bfloat16", "float32"],
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


def resolve_model_and_layers(args: argparse.Namespace):
    config, model_dtype, raw_config = load_text_config(args.model_path)
    config, model_dtype, raw_config = apply_model_config_overrides(
        args, config, model_dtype, raw_config
    )
    linear_layer_ids, full_layer_ids = derive_layer_ids(raw_config)
    linear_layer_id = (
        args.linear_layer_id if args.linear_layer_id is not None else linear_layer_ids[0]
    )
    full_layer_id = (
        args.full_layer_id if args.full_layer_id is not None else full_layer_ids[0]
    )
    if linear_layer_id not in linear_layer_ids:
        raise ValueError(f"{linear_layer_id=} is not a linear attention layer.")
    if full_layer_id not in full_layer_ids:
        raise ValueError(f"{full_layer_id=} is not a full attention layer.")
    cache_params = build_local_mamba_cache_params(
        config,
        [linear_layer_id],
        dtype_override=resolve_mamba_state_dtype(args, config),
    )
    return (
        config,
        model_dtype,
        raw_config,
        linear_layer_ids,
        full_layer_ids,
        linear_layer_id,
        full_layer_id,
        cache_params,
    )
