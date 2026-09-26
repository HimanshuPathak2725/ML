"""Stream a dataset profile without loading the full TSVs into memory."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


POSTAL_RE = re.compile(r"\b\d{4,6}\b")
SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")


def _empty_text_stats() -> dict[str, object]:
    return {
        "rows": 0,
        "missing": 0,
        "characters": 0,
        "max_characters": 0,
        "postal_rows": 0,
    }


def profile_source(path: str | Path, columns: Iterable[str] = SOURCE_COLUMNS) -> dict[str, object]:
    """Profile one source TSV using bounded memory."""
    path = Path(path)
    columns = tuple(columns)
    stats = {column: _empty_text_stats() for column in columns if column != "entity_id"}
    countries: Counter[str] = Counter()
    rows = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != list(columns):
            raise ValueError(
                f"{path}: expected columns {list(columns)}, got {reader.fieldnames}"
            )
        for row in reader:
            rows += 1
            country = (row.get("country") or "").strip()
            countries[country] += 1
            for column, column_stats in stats.items():
                value = (row.get(column) or "").strip()
                length = len(value)
                column_stats["rows"] += 1
                column_stats["characters"] += length
                column_stats["max_characters"] = max(column_stats["max_characters"], length)
                if not value:
                    column_stats["missing"] += 1
                if column in {"business_name", "business_address"} and POSTAL_RE.search(value):
                    column_stats["postal_rows"] += 1

    for column_stats in stats.values():
        column_stats["mean_characters"] = round(
            column_stats["characters"] / rows, 2
        ) if rows else 0.0

    return {
        "path": str(path),
        "rows": rows,
        "countries": dict(countries.most_common()),
        "fields": stats,
    }


def profile_ground_truth(path: str | Path) -> dict[str, object]:
    """Profile match cardinalities without storing the ground truth mapping."""
    counts: Counter[int] = Counter()
    rows = 0
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ["source1_entity_id", "matched_entity_ids"]
        if reader.fieldnames != expected:
            raise ValueError(f"{path}: expected columns {expected}, got {reader.fieldnames}")
        for row in reader:
            raw_ids = (row.get("matched_entity_ids") or "").strip()
            count = len([item for item in raw_ids.split(",") if item]) if raw_ids else 0
            counts[count] += 1
            rows += 1
    return {
        "path": str(path),
        "rows": rows,
        "match_count_distribution": {str(k): v for k, v in sorted(counts.items())},
        "singleton_rows": counts.get(0, 0),
        "matched_rows": rows - counts.get(0, 0),
    }


def build_profile(root: str | Path) -> dict[str, object]:
    root = Path(root)
    train = root / "dataset" / "train"
    test = root / "dataset" / "test"
    result: dict[str, object] = {"sources": {}, "ground_truth": {}}
    for split, directory, prefix in (
        ("train", train, "train"),
        ("test", test, "test"),
    ):
        result["sources"][split] = {
            source: profile_source(directory / f"{prefix}_source{source}.tsv")
            for source in ("1", "2", "3")
        }
    result["ground_truth"] = profile_ground_truth(train / "train_ground_truth.tsv")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="student_resource directory")
    parser.add_argument("--json", dest="json_path", help="optional JSON output path")
    args = parser.parse_args()

    profile = build_profile(args.root)
    rendered = json.dumps(profile, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_path:
        Path(args.json_path).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()