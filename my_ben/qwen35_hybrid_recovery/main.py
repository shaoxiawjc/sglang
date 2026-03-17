from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "FLASHINFER_WORKSPACE_BASE", str(REPO_ROOT / ".flashinfer_workspace")
)
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.srt.layers.attention.linear.utils import initialize_linear_attn_config
from sglang.srt.server_args import ServerArgs

from .benches import (
    run_full_attention_benchmarks,
    run_kv_transfer_benchmarks,
    run_linear_attention_benchmarks,
    run_state_transfer_benchmarks,
)
from .config import (
    apply_model_config_overrides,
    build_local_mamba_cache_params,
    derive_layer_ids,
    load_text_config,
    parse_args,
    resolve_mamba_state_dtype,
)
from .metadata import build_metadata
from .plotting import generate_plots
from .utils import now_stamp, write_outputs


def main() -> None:
    args = parse_args(REPO_ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    server_args = ServerArgs(model_path="dummy")
    server_args.linear_attn_backend = args.linear_attn_backend
    server_args.linear_attn_decode_backend = args.linear_attn_decode_backend
    server_args.linear_attn_prefill_backend = args.linear_attn_prefill_backend
    initialize_linear_attn_config(server_args)

    config, model_dtype, raw_config = load_text_config(args.model_path)
    config, model_dtype, raw_config = apply_model_config_overrides(
        args, config, model_dtype, raw_config
    )
    linear_layer_ids, full_layer_ids = derive_layer_ids(raw_config)
    mamba_cache_params = build_local_mamba_cache_params(
        config,
        linear_layer_ids,
        dtype_override=resolve_mamba_state_dtype(args, config),
    )
    stamp = now_stamp()
    output_dir = args.output_dir / stamp
    rows: list[dict[str, object]] = []

    run_state_transfer_benchmarks(
        rows,
        args=args,
        cache_params=mamba_cache_params,
        linear_layer_ids=linear_layer_ids,
    )
    run_linear_attention_benchmarks(
        rows,
        args=args,
        config=config,
        model_dtype=model_dtype,
        cache_params=mamba_cache_params,
        linear_layer_ids=linear_layer_ids,
    )
    run_full_attention_benchmarks(
        rows,
        args=args,
        config=config,
        model_dtype=model_dtype,
    )
    run_kv_transfer_benchmarks(
        rows,
        args=args,
        config=config,
        model_dtype=model_dtype,
        full_layer_ids=full_layer_ids,
    )

    metadata = build_metadata(
        args=args,
        repo_root=REPO_ROOT,
        stamp=stamp,
        config=config,
        model_dtype=model_dtype,
        raw_config=raw_config,
        full_layer_ids=full_layer_ids,
        linear_layer_ids=linear_layer_ids,
        mamba_cache_params=mamba_cache_params,
    )
    write_outputs(output_dir, rows=rows, metadata=metadata)
    generate_plots(rows, output_dir)

    print(f"Saved benchmark results to {output_dir}")
    for category in sorted({row['category'] for row in rows}):
        category_rows = [row for row in rows if row["category"] == category]
        best = min(category_rows, key=lambda x: x["median_ms"])
        print(f"{category}: best median {best['median_ms']:.3f} ms | {best}")
