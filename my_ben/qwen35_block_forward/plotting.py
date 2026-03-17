from __future__ import annotations

from pathlib import Path


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _plot_category(rows, output_path: Path, title: str, categories: tuple[str, str]) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        print("matplotlib is not installed, skip plot generation.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for category in categories:
        category_rows = [row for row in rows if row["category"] == category]
        if not category_rows:
            continue
        keys = sorted(
            {(int(row["batch_size"]), int(row["prefix_len"])) for row in category_rows}
        )
        for batch_size, prefix_len in keys:
            subset = sorted(
                [
                    row
                    for row in category_rows
                    if int(row["batch_size"]) == batch_size
                    and int(row["prefix_len"]) == prefix_len
                ],
                key=lambda row: int(row["seq_len"]),
            )
            x = [int(row["seq_len"]) for row in subset]
            y = [float(row["median_ms"]) for row in subset]
            label = f"{category}, bs={batch_size}, prefix={prefix_len}"
            ax.plot(x, y, marker="o", label=label)

    ax.set_title(title)
    ax.set_xlabel("seq len")
    ax.set_ylabel("time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_plots(rows, output_dir: Path) -> None:
    _plot_category(
        rows,
        output_dir / "plot_linear_block.png",
        "Qwen3.5 Linear Block Extend",
        ("linear_block_extend_current", "linear_block_extend_replay"),
    )
    _plot_category(
        rows,
        output_dir / "plot_full_block.png",
        "Qwen3.5 Full Block Extend",
        ("full_block_extend_current", "full_block_extend_replay"),
    )
