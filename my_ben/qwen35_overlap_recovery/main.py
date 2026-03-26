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

from my_ben.qwen35_hybrid_recovery.utils import now_stamp
from sglang.srt.layers.attention.linear.utils import initialize_linear_attn_config
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

from .benches import run_overlap_benchmarks
from .config import parse_args, resolve_model_and_group
from .metadata import build_metadata
from .plotting import generate_plots
from .utils import write_overlap_outputs


def main() -> None:
    args = parse_args(REPO_ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    server_args = ServerArgs(model_path="dummy")
    server_args.linear_attn_backend = args.linear_attn_backend
    server_args.linear_attn_decode_backend = args.linear_attn_decode_backend
    server_args.linear_attn_prefill_backend = args.linear_attn_prefill_backend
    set_global_server_args_for_scheduler(server_args)
    initialize_linear_attn_config(server_args)

    (
        config,
        model_dtype,
        raw_config,
        linear_layer_ids,
        causal_layer_id,
        recompute_counts,
        cache_params,
    ) = resolve_model_and_group(args)
    config.torch_dtype = model_dtype

    rows: list[dict[str, object]] = []
    stamp = now_stamp()
    output_dir = args.output_dir / stamp

    run_overlap_benchmarks(
        rows,
        args=args,
        config=config,
        model_dtype=model_dtype,
        linear_cache_params=cache_params,
        linear_layer_ids=linear_layer_ids,
        causal_layer_id=causal_layer_id,
        recompute_counts=recompute_counts,
    )

    metadata = build_metadata(
        args=args,
        repo_root=REPO_ROOT,
        stamp=stamp,
        config=config,
        model_dtype=model_dtype,
        raw_config=raw_config,
        linear_layer_ids=linear_layer_ids,
        causal_layer_id=causal_layer_id,
        recompute_counts=recompute_counts,
        cache_params=cache_params,
    )
    write_overlap_outputs(output_dir, rows=rows, metadata=metadata)
    generate_plots(rows, output_dir)

    print(f"Saved benchmark results to {output_dir}")
    for category in sorted({row['category'] for row in rows}):
        category_rows = [row for row in rows if row["category"] == category]
        best = min(category_rows, key=lambda x: x["median_ms"])
        print(f"{category}: best median {best['median_ms']:.3f} ms")
