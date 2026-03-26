from __future__ import annotations

import csv
import json
from pathlib import Path

from my_ben.qwen35_hybrid_recovery.utils import make_jsonable


CATEGORY_DIRS = {
    "baseline_recompute": "baseline_recompute",
    "baseline_offload_onload": "baseline_offload_onload",
    "ours_ca_recompute_overlap_la_state_conv": "ours_ca_recompute_overlap_la_state_conv",
    "ours_la_recompute_overlap_ca_kvcache": "ours_la_recompute_overlap_ca_kvcache",
}


def _write_rows_json_csv(output_dir: Path, rows: list[dict[str, object]]) -> None:
    (output_dir / "results.json").write_text(
        json.dumps(make_jsonable(rows), indent=2, sort_keys=True)
    )
    fieldnames: list[str] = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "results.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_overlap_outputs(
    output_dir: Path,
    *,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows_json_csv(output_dir, rows)
    (output_dir / "metadata.json").write_text(
        json.dumps(make_jsonable(metadata), indent=2, sort_keys=True)
    )

    groups = {
        "baseline": ["baseline_recompute", "baseline_offload_onload"],
        "ours": [
            "ours_ca_recompute_overlap_la_state_conv",
            "ours_la_recompute_overlap_ca_kvcache",
        ],
    }
    for group_name, categories in groups.items():
        group_rows = [row for row in rows if row["category"] in categories]
        (output_dir / f"{group_name}.json").write_text(
            json.dumps(make_jsonable(group_rows), indent=2, sort_keys=True)
        )
        if group_rows:
            group_fields: list[str] = sorted(
                {key for row in group_rows for key in row.keys()}
            )
            with (output_dir / f"{group_name}.csv").open("w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=group_fields)
                writer.writeheader()
                writer.writerows(group_rows)

    experiments_dir = output_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)
    for category, dirname in CATEGORY_DIRS.items():
        category_rows = [row for row in rows if row["category"] == category]
        if not category_rows:
            continue
        category_dir = experiments_dir / dirname
        category_dir.mkdir(exist_ok=True)
        _write_rows_json_csv(category_dir, category_rows)
