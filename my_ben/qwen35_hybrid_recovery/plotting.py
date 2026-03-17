from __future__ import annotations

from pathlib import Path

import numpy as np


def sort_config_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("batch_size", 0)),
            int(row.get("seq_len", 0)),
            int(row.get("prefix_len", 0)),
        ),
    )


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def plot_kv_transfer(rows: list[dict[str, object]], output_dir: Path) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        print("matplotlib is not installed, skip plot generation.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for category, label in (
        ("kvcache_h2d_one_full_layer_torch_indexed", "torch indexed"),
        ("kvcache_h2d_one_full_layer_sglang_kernel", "sglang kernel"),
    ):
        category_rows = sorted(
            [row for row in rows if row["category"] == category],
            key=lambda row: int(row["token_count"]),
        )
        if not category_rows:
            continue
        x = [int(row["token_count"]) for row in category_rows]
        y = [float(row["median_ms"]) for row in category_rows]
        ax.plot(x, y, marker="o", label=label)

    ax.set_title("KV Transfer")
    ax.set_xlabel("token count")
    ax.set_ylabel("time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "plot_kv_transfer.png", dpi=180)
    plt.close(fig)


def plot_state_transfer(rows: list[dict[str, object]], output_dir: Path) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for category, label in (
        ("state_h2d_conv", "conv"),
        ("state_h2d_temporal", "temporal"),
        ("state_h2d_total", "total"),
    ):
        category_rows = sorted(
            [row for row in rows if row["category"] == category],
            key=lambda row: int(row["slot_count"]),
        )
        if not category_rows:
            continue
        x = [int(row["slot_count"]) for row in category_rows]
        y = [float(row["median_ms"]) for row in category_rows]
        ax.plot(x, y, marker="o", label=label)

    ax.set_title("State Transfer")
    ax.set_xlabel("slot count / batch size")
    ax.set_ylabel("time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "plot_state_transfer.png", dpi=180)
    plt.close(fig)


def plot_linear_attention(rows: list[dict[str, object]], output_dir: Path) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    category_rows = sort_config_rows(
        [row for row in rows if row["category"] == "linear_attention_extend"]
    )
    if not category_rows:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    batch_sizes = sorted({int(row["batch_size"]) for row in category_rows})
    for batch_size in batch_sizes:
        batch_rows = sorted(
            [row for row in category_rows if int(row["batch_size"]) == batch_size],
            key=lambda row: int(row["seq_len"]),
        )
        x = [int(row["seq_len"]) for row in batch_rows]
        y = [float(row["median_ms"]) for row in batch_rows]
        ax.plot(x, y, marker="o", label=f"bs={batch_size}")

    ax.set_title("Linear Attention Compute")
    ax.set_xlabel("seq len")
    ax.set_ylabel("time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "plot_linear_attention.png", dpi=180)
    plt.close(fig)


def plot_full_attention(rows: list[dict[str, object]], output_dir: Path) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    category_rows = sort_config_rows(
        [row for row in rows if row["category"] == "full_attention_extend"]
    )
    if not category_rows:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    prefix_lens = sorted({int(row["prefix_len"]) for row in category_rows})
    for prefix_len in prefix_lens:
        prefix_rows = sorted(
            [row for row in category_rows if int(row["prefix_len"]) == prefix_len],
            key=lambda row: int(row["seq_len"]),
        )
        by_seq = {}
        for row in prefix_rows:
            by_seq.setdefault(int(row["seq_len"]), []).append(float(row["median_ms"]))
        x = sorted(by_seq.keys())
        y = [float(np.mean(by_seq[seq_len])) for seq_len in x]
        ax.plot(x, y, marker="o", label=f"prefix={prefix_len}")

    ax.set_title("Full Attention Compute")
    ax.set_xlabel("seq len")
    ax.set_ylabel("time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "plot_full_attention.png", dpi=180)
    plt.close(fig)


def generate_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    plot_kv_transfer(rows, output_dir)
    plot_state_transfer(rows, output_dir)
    plot_linear_attention(rows, output_dir)
    plot_full_attention(rows, output_dir)
