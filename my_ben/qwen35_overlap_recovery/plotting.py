from __future__ import annotations

from pathlib import Path


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _config_label(row: dict[str, object]) -> str:
    return (
        f"bs={row['batch_size']}\n"
        f"seq={row['seq_len']}\n"
        f"prefix={row['prefix_len']}"
    )


def plot_strategy_comparison(rows: list[dict[str, object]], output_dir: Path) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        print("matplotlib is not installed, skip plot generation.")
        return

    baseline_categories = [
        "strategy_all_recompute",
        "strategy_all_onload",
    ]
    overlap_category = "strategy_overlap_linear_recompute_then_prefetch"
    label_to_rows: dict[str, dict[str, object]] = {}
    overlap_counts = sorted(
        {
            int(row["linear_recompute_count"])
            for row in rows
            if row["category"] == overlap_category
        }
    )
    for row in rows:
        category = row["category"]
        if category not in baseline_categories and category != overlap_category:
            continue
        label = _config_label(row)
        bucket = label_to_rows.setdefault(label, {})
        if category == overlap_category:
            bucket[f"{category}|a={row['linear_recompute_count']}"] = row
        elif category not in bucket:
            bucket[category] = row

    if not label_to_rows:
        return

    labels = list(label_to_rows.keys())
    x = list(range(len(labels)))
    series_keys = ["strategy_all_recompute"]
    series_keys.extend([f"{overlap_category}|a={count}" for count in overlap_counts])
    series_keys.append("strategy_all_onload")
    width = 0.8 / max(len(series_keys), 1)
    color_palette = [
        "#d1495b",
        "#2a9d8f",
        "#3b82f6",
        "#f4a261",
        "#7c3aed",
        "#e76f51",
        "#577590",
        "#43aa8b",
        "#f8961e",
    ]
    titles = {"strategy_all_recompute": "All Recompute"}
    titles.update(
        {
            f"{overlap_category}|a={count}": f"Overlap a={count}"
            for count in overlap_counts
        }
    )
    titles["strategy_all_onload"] = "All Onload"

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.6), 6))
    center_offset = (len(series_keys) - 1) / 2
    for idx, series_key in enumerate(series_keys):
        offset = (idx - center_offset) * width
        values = [
            label_to_rows[label].get(series_key, {}).get("median_ms", float("nan"))
            for label in labels
        ]
        ax.bar(
            [value + offset for value in x],
            values,
            width=width,
            label=titles[series_key],
            color=color_palette[idx % len(color_palette)],
        )

    ax.set_title("Recovery Strategy Comparison")
    ax.set_ylabel("Median Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_strategy_comparison.png", dpi=200)
    plt.close(fig)


def generate_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    plot_strategy_comparison(rows, output_dir)
