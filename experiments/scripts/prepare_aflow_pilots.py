#!/usr/bin/env python3
"""Prepare AWF pilot datasets from AFlow's published archive.

The AFlow paper reports 617 level-5 MATH problems from four domains.  The
currently published dataset archive contains 119 validation and 486 test
rows (605 total).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

MATH_DOMAINS = (
    "Counting & Probability",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _annotate(
    rows: Iterable[dict[str, Any]],
    source_split: str,
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "_aflow_source_split": source_split,
            "_aflow_source_index": index,
        }
        for index, row in enumerate(rows)
    ]


def _select_math(
    rows: list[dict[str, Any]],
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select *size* rows balanced across domains, 80% validate / 20% test."""
    if size < 20 or size % (len(MATH_DOMAINS) * 5):
        raise ValueError(
            "math pilot size must be a positive multiple of 20 so 80% can "
            "come from AFlow validation and 20% from its held-out test"
        )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("level") != "Level 5" or row.get("type") not in MATH_DOMAINS:
            raise ValueError("AFlow MATH source contains an unexpected level/domain")
        groups[
            (str(row["_aflow_source_split"]), str(row["type"]))
        ].append(row)

    rng = random.Random(seed)
    heldout_per_domain = size // (len(MATH_DOMAINS) * 5)
    development_per_domain = size // len(MATH_DOMAINS) - heldout_per_domain
    selected: list[dict[str, Any]] = []
    for domain in MATH_DOMAINS:
        for source_split, count in (
            ("validate", development_per_domain),
            ("test", heldout_per_domain),
        ):
            candidates = list(groups[(source_split, domain)])
            if len(candidates) < count:
                raise ValueError(
                    f"Not enough {source_split} rows for domain {domain}"
                )
            rng.shuffle(candidates)
            selected.extend(candidates[:count])
    return selected


def _select_humaneval(
    rows: list[dict[str, Any]],
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select *size* rows, 80% validate / 20% test."""
    if size < 5 or size % 5 or size > len(rows):
        raise ValueError("invalid HumanEval pilot size")
    task_ids = [str(row.get("task_id", "")) for row in rows]
    if any(not task_id for task_id in task_ids) or len(task_ids) != len(
        set(task_ids)
    ):
        raise ValueError("HumanEval task_id values must be present and unique")

    rng = random.Random(seed)
    heldout_count = size // 5
    development_count = size - heldout_count
    by_source = {
        source_split: [
            row
            for row in rows
            if row["_aflow_source_split"] == source_split
        ]
        for source_split in ("validate", "test")
    }
    rng.shuffle(by_source["validate"])
    rng.shuffle(by_source["test"])
    return (
        by_source["validate"][:development_count]
        + by_source["test"][:heldout_count]
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def prepare(
    source_dir: Path,
    output_dir: Path,
    *,
    pilot_size: int,
    seed: int = 20260726,
) -> dict[str, Any]:
    """Generate pilot datasets from AFlow source files."""
    source_paths = {
        "math_validate": source_dir / "math_validate.jsonl",
        "math_test": source_dir / "math_test.jsonl",
        "humaneval_validate": source_dir / "humaneval_validate.jsonl",
        "humaneval_test": source_dir / "humaneval_test.jsonl",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing AFlow source files: " + ", ".join(missing))

    math_rows = _annotate(
        _read_jsonl(source_paths["math_validate"]),
        "validate",
    ) + _annotate(
        _read_jsonl(source_paths["math_test"]),
        "test",
    )
    humaneval_rows = _annotate(
        _read_jsonl(source_paths["humaneval_validate"]),
        "validate",
    ) + _annotate(
        _read_jsonl(source_paths["humaneval_test"]),
        "test",
    )

    math_counts = Counter(str(row.get("type")) for row in math_rows)
    expected_math_counts = {
        "Counting & Probability": 123,
        "Number Theory": 154,
        "Prealgebra": 193,
        "Precalculus": 135,
    }
    if len(math_rows) != 605 or dict(math_counts) != expected_math_counts:
        raise ValueError(
            "Published AFlow MATH rows changed; inspect the source archive"
        )
    if len(humaneval_rows) != 164:
        raise ValueError("Published AFlow HumanEval rows changed")

    outputs = {
        "math_full": output_dir / "math_level5_four_domains_aflow_full.jsonl",
        "math_pilot": output_dir / f"math_level5_four_domains_pilot{pilot_size}.jsonl",
        "humaneval_full": output_dir / "humaneval_aflow_full.jsonl",
        "humaneval_pilot": output_dir / f"humaneval_pilot{pilot_size}.jsonl",
    }
    _write_jsonl(outputs["math_full"], math_rows)
    _write_jsonl(outputs["math_pilot"], _select_math(math_rows, pilot_size, seed))
    _write_jsonl(outputs["humaneval_full"], humaneval_rows)
    _write_jsonl(outputs["humaneval_pilot"], _select_humaneval(humaneval_rows, pilot_size, seed))

    pilot_info = {
        "selection": "random shuffle without replacement",
        "selection_seed": seed,
        "size_per_benchmark": pilot_size,
        "math_rows_per_domain": pilot_size // len(MATH_DOMAINS),
        "source_policy": (
            "Optimization and validation use only AFlow validation rows; "
            "held-out test uses only AFlow test rows."
        ),
    }
    manifest = {
        "schema_version": 1,
        "math": {
            "paper_reported_count": 617,
            "published_archive_count": len(math_rows),
            "filter": {
                "level": "Level 5",
                "domains": list(MATH_DOMAINS),
            },
            "published_domain_counts": dict(sorted(math_counts.items())),
        },
        "humaneval": {
            "published_archive_count": len(humaneval_rows),
        },
        "pilot": pilot_info,
        "outputs": {
            name: {
                "path": str(path),
                "rows": _count_jsonl(path),
            }
            for name, path in outputs.items()
        },
    }
    manifest_path = output_dir / "selection_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AWF pilot datasets")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/rongxing/Benchmark/AFlow"),
    )
    parser.add_argument("--pilot-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    manifest = prepare(
        args.source_dir,
        args.output_dir,
        pilot_size=args.pilot_size,
        seed=args.seed,
    )
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    main()
