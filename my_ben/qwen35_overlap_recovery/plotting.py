from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _subplot_bucket(row: dict[str, object]) -> tuple[int, int]:
    return int(row["batch_size"]), int(row["prefix_len"])


def _subplot_label(row: dict[str, object]) -> str:
    return f"prefix={row['prefix_len']}"


def plot_strategy_comparison(rows: list[dict[str, object]], output_dir: Path) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        print("matplotlib is not installed, skip plot generation.")
        return

    title_fontsize = 22
    axis_title_fontsize = 17
    tick_fontsize = 14
    legend_fontsize = 14

    series_keys = [
        "baseline_recompute",
        "baseline_offload_onload",
        "ours_ca_recompute_overlap_la_state_conv",
        "ours_la_recompute_overlap_ca_kvcache|la=1",
        "ours_la_recompute_overlap_ca_kvcache|la=2",
        "ours_la_recompute_overlap_ca_kvcache|la=3",
    ]
    titles = {
        "baseline_recompute": "Baseline\nRecompute",
        "baseline_offload_onload": "Baseline\nOffload-Onload",
        "ours_ca_recompute_overlap_la_state_conv": "Ours CA->LA\n(CA recompute)",
        "ours_la_recompute_overlap_ca_kvcache|la=1": "Ours LA->CA\n(LA=1)",
        "ours_la_recompute_overlap_ca_kvcache|la=2": "Ours LA->CA\n(LA=2)",
        "ours_la_recompute_overlap_ca_kvcache|la=3": "Ours LA->CA\n(LA=3)",
    }
    color_palette = [
        "#d1495b",
        "#577590",
        "#2a9d8f",
        "#3b82f6",
        "#f4a261",
        "#e76f51",
    ]

    seq_lens = sorted({int(row["seq_len"]) for row in rows})
    if not seq_lens:
        return

    ncols = min(2, len(seq_lens))
    nrows = math.ceil(len(seq_lens) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(12, 7.2 * ncols), max(7, 4.8 * nrows)),
        squeeze=False,
    )
    flat_axes = axes.flatten()
    legend_handles = []

    for ax, seq_len in zip(flat_axes, seq_lens):
        seq_rows = [row for row in rows if int(row["seq_len"]) == seq_len]
        seq_bucket_order: list[tuple[int, int]] = []
        seq_buckets: dict[tuple[int, int], dict[str, object]] = {}
        for row in seq_rows:
            bucket_key = _subplot_bucket(row)
            if bucket_key not in seq_buckets:
                seq_bucket_order.append(bucket_key)
            bucket = seq_buckets.setdefault(bucket_key, {})
            category = row["category"]
            if category == "ours_la_recompute_overlap_ca_kvcache":
                key = f"{category}|la={row['linear_recompute_count']}"
            else:
                key = str(category)
            bucket[key] = row

        seq_bucket_order.sort()
        x = list(range(len(seq_bucket_order)))
        width = 0.8 / len(series_keys)
        center_offset = (len(series_keys) - 1) / 2
        for idx, series_key in enumerate(series_keys):
            offset = (idx - center_offset) * width
            values = [
                seq_buckets[bucket_key].get(series_key, {}).get("median_ms", float("nan"))
                for bucket_key in seq_bucket_order
            ]
            bars = ax.bar(
                [value + offset for value in x],
                values,
                width=width,
                label=titles[series_key],
                color=color_palette[idx % len(color_palette)],
            )
            if len(legend_handles) < len(series_keys):
                legend_handles.append(bars[0])
        ax.set_title(f"seq_len={seq_len}", fontsize=axis_title_fontsize)
        ax.set_ylabel("Median Time (ms)", fontsize=axis_title_fontsize)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"prefix={prefix_len}" for _, prefix_len in seq_bucket_order],
            rotation=20,
            ha="right",
            fontsize=tick_fontsize,
        )
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    for ax in flat_axes[len(seq_lens) :]:
        ax.axis("off")

    if legend_handles:
        fig.legend(
            legend_handles,
            [titles[key] for key in series_keys],
            loc="lower center",
            ncol=2,
            fontsize=legend_fontsize,
            bbox_to_anchor=(0.5, 0.02),
            frameon=False,
        )

    fig.suptitle("3 LA : 1 CA Overlap Recovery", fontsize=title_fontsize)
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.18,
        top=0.90,
        wspace=0.20,
        hspace=0.34,
    )
    fig.savefig(
        output_dir / "plot_strategy_comparison.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(fig)


def generate_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    plot_strategy_comparison(rows, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Benchmark result directory that contains results.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write plots to. Defaults to --input-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else input_dir
    rows = json.loads((input_dir / "results.json").read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_plots(rows, output_dir)
    print(f"Saved plots to {output_dir}")
