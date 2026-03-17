from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch


def dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype)


def nbytes_of_shape(shape: Iterable[int], dtype: torch.dtype) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel * torch.tensor([], dtype=dtype).element_size()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def benchmark_cuda_op(
    fn: Callable[[], None],
    *,
    warmup_iters: int,
    bench_iters: int,
) -> dict[str, float]:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(bench_iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

    return {
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "p20_ms": percentile(times_ms, 20),
        "p80_ms": percentile(times_ms, 80),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
    }


def gbps(num_bytes: int, median_ms: float) -> float:
    return num_bytes / median_ms / 1e6


def format_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1e9:.6f} GB"


def git_commit(repo_root: Path) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            )
            .strip()
        )
    except Exception:
        return None


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def add_row(
    rows: list[dict[str, object]],
    *,
    category: str,
    stats: dict[str, float],
    payload: dict[str, object],
) -> None:
    row = {"category": category, **payload, **stats}
    rows.append(row)


def make_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
    return obj


def write_outputs(
    output_dir: Path,
    *,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    metadata_path = output_dir / "metadata.json"

    json_path.write_text(json.dumps(make_jsonable(rows), indent=2, sort_keys=True))
    metadata_path.write_text(
        json.dumps(make_jsonable(metadata), indent=2, sort_keys=True)
    )

    fieldnames: list[str] = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_category_tables(output_dir, rows)


def write_category_tables(output_dir: Path, rows: list[dict[str, object]]) -> None:
    groups = {
        "kv_transfer": [
            "kvcache_h2d_one_full_layer_torch_indexed",
            "kvcache_h2d_one_full_layer_sglang_kernel",
        ],
        "state_transfer": ["state_h2d_conv", "state_h2d_temporal", "state_h2d_total"],
        "linear_attention": ["linear_attention_extend"],
        "full_attention": ["full_attention_extend"],
    }
    for group_name, categories in groups.items():
        group_rows = [row for row in rows if row["category"] in categories]
        json_path = output_dir / f"{group_name}.json"
        csv_path = output_dir / f"{group_name}.csv"
        json_path.write_text(json.dumps(make_jsonable(group_rows), indent=2, sort_keys=True))
        fieldnames: list[str] = sorted({key for row in group_rows for key in row.keys()})
        with csv_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(group_rows)
