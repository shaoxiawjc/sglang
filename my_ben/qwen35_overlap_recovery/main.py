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
sys.path.insert(0, str(REPO_ROOT))

from my_ben.qwen35_hybrid_recovery.utils import now_stamp, write_outputs
from sglang.srt.layers.attention.linear.utils import initialize_linear_attn_config
from sglang.srt.server_args import ServerArgs

from .benches import run_overlap_benchmarks
from .config import (
    apply_model_config_overrides,
    build_local_mamba_cache_params,
    build_subset_mamba_cache_params,
    derive_layer_ids,
    load_text_config,
    normalize_recompute_counts,
    parse_args,
    resolve_mamba_state_dtype,
    select_recovery_layers,
)
from .metadata import build_metadata
from .plotting import generate_plots


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
    target_linear_layer_ids, target_full_layer_ids = select_recovery_layers(
        linear_layer_ids,
        full_layer_ids,
        group_index=args.group_index,
        linear_layer_count=args.linear_layer_count,
        full_layer_count=args.full_layer_count,
        target_linear_layer_ids=args.target_linear_layer_ids,
        target_full_layer_ids=args.target_full_layer_ids,
    )
    recompute_counts = normalize_recompute_counts(
        args.linear_recompute_counts, len(target_linear_layer_ids)
    )
    full_mamba_cache_params = build_local_mamba_cache_params(
        config,
        linear_layer_ids,
        dtype_override=resolve_mamba_state_dtype(args, config),
    )
    target_cache_params = build_subset_mamba_cache_params(
        full_mamba_cache_params, target_linear_layer_ids
    )
    config.torch_dtype = model_dtype

    rows: list[dict[str, object]] = []
    stamp = now_stamp()
    output_dir = args.output_dir / stamp

    run_overlap_benchmarks(
        rows,
        args=args,
        config=config,
        model_dtype=model_dtype,
        linear_cache_params=target_cache_params,
        target_linear_layer_ids=target_linear_layer_ids,
        target_full_layer_ids=target_full_layer_ids,
        recompute_counts=recompute_counts,
    )

    metadata = build_metadata(
        args=args,
        repo_root=REPO_ROOT,
        stamp=stamp,
        config=config,
        model_dtype=model_dtype,
        raw_config=raw_config,
        target_linear_layer_ids=target_linear_layer_ids,
        target_full_layer_ids=target_full_layer_ids,
        recompute_counts=recompute_counts,
        cache_params=target_cache_params,
    )
    write_outputs(output_dir, rows=rows, metadata=metadata)
    generate_plots(rows, output_dir)

    print(f"Saved benchmark results to {output_dir}")
    for category in sorted({row['category'] for row in rows}):
        category_rows = [row for row in rows if row["category"] == category]
        best = min(category_rows, key=lambda x: x["median_ms"])
        print(f"{category}: best median {best['median_ms']:.3f} ms")
