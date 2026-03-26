from __future__ import annotations

from pathlib import Path

from .utils import CATEGORY_DIRS


CATEGORY_PLOT_SPECS = {
    "linear_block_full_forward": {
        "title": "Linear Block Full Forward",
        "x_key": "seq_len",
        "series_keys": ("batch_size", "prefix_len"),
        "xlabel": "seq len",
    },
    "linear_block_reuse_cache_compute": {
        "title": "Linear Block Reuse Cache Compute",
        "x_key": "seq_len",
        "series_keys": ("batch_size", "prefix_len"),
        "xlabel": "seq len",
    },
    "linear_state_cache_h2d": {
        "title": "Linear State Cache H2D",
        "x_key": "batch_size",
        "series_keys": tuple(),
        "xlabel": "batch size",
    },
    "linear_attention_state_update": {
        "title": "Linear Attention State Update",
        "x_key": "seq_len",
        "series_keys": ("batch_size", "prefix_len"),
        "xlabel": "seq len",
    },
    "full_block_full_forward": {
        "title": "Full Block Full Forward",
        "x_key": "seq_len",
        "series_keys": ("batch_size", "prefix_len"),
        "xlabel": "seq len",
    },
    "full_block_reuse_cache_compute": {
        "title": "Full Block Reuse Cache Compute",
        "x_key": "seq_len",
        "series_keys": ("batch_size", "prefix_len"),
        "xlabel": "seq len",
    },
    "full_kvcache_h2d": {
        "title": "Full KVCache H2D",
        "x_key": "prefix_len",
        "series_keys": ("batch_size",),
        "xlabel": "prefix len",
    },
    "full_attention_softmax_qkv": {
        "title": "Full Attention Softmax(QK)V",
        "x_key": "seq_len",
        "series_keys": ("batch_size", "prefix_len"),
        "xlabel": "seq len",
    },
}


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _plot_single_category(rows, output_path: Path, category: str) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        print("matplotlib is not installed, skip plot generation.")
        return

    spec = CATEGORY_PLOT_SPECS[category]
    fig, ax = plt.subplots(figsize=(10, 6))
    x_key = spec["x_key"]
    series_keys = spec["series_keys"]

    if series_keys:
        series_values = sorted(
            {
                tuple(int(row[key]) for key in series_keys)
                for row in rows
            }
        )
    else:
        series_values = [tuple()]

    for series in series_values:
        subset = rows
        label_parts = []
        for key, value in zip(series_keys, series):
            subset = [row for row in subset if int(row[key]) == value]
            label_parts.append(f"{key}={value}")
        subset = sorted(subset, key=lambda row: int(row[x_key]))
        if not subset:
            continue

        x_values = sorted({int(row[x_key]) for row in subset})
        y_values = []
        for x in x_values:
            points = [float(row["median_ms"]) for row in subset if int(row[x_key]) == x]
            y_values.append(sum(points) / len(points))
        label = ", ".join(label_parts) if label_parts else category
        ax.plot(x_values, y_values, marker="o", label=label)

    ax.set_title(spec["title"])
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel("time (ms)")
    ax.grid(True, alpha=0.3)
    if len(series_values) > 1 or series_keys:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_plots(rows, output_dir: Path) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        print("matplotlib is not installed, skip plot generation.")
        return

    experiments_dir = output_dir / "experiments"
    for category, dirname in CATEGORY_DIRS.items():
        category_rows = [row for row in rows if row["category"] == category]
        if not category_rows:
            continue
        _plot_single_category(
            category_rows,
            experiments_dir / dirname / "plot.png",
            category,
        )
