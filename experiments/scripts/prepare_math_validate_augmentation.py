#!/usr/bin/env python3
"""Add unseen Prealgebra/Precalculus Level-5 validation examples.

The checked-in ``competition_math_level5.jsonl`` used to contain only three
domains.  This script first rebuilds it from the complete competition MATH
JSONL, then selects a deterministic, balanced set that does not occur in the
published 605-row AFlow artifact.  The original AFlow file is deliberately
left unchanged so its published provenance remains meaningful.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TARGET_DOMAINS = ("Prealgebra", "Precalculus")
SOURCE_LABEL = "Benchmark/MATH/competition_math_level5.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            if not all(key in value for key in ("problem", "solution", "type", "level")):
                raise ValueError(f"{path}:{line_number} is missing a MATH field")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _problem_key(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def rebuild_level5_source(
    competition_all_path: Path,
    level5_path: Path,
) -> list[dict[str, Any]]:
    """Rebuild the complete all-domain Level-5 source in original order."""
    all_rows = _read_jsonl(competition_all_path)
    level5_rows = [row for row in all_rows if row.get("level") == "Level 5"]
    counts = Counter(str(row.get("type")) for row in level5_rows)
    missing = [domain for domain in TARGET_DOMAINS if counts[domain] == 0]
    if missing:
        raise ValueError(f"Complete MATH source lacks target domains: {missing}")
    _write_jsonl(level5_path, level5_rows)
    return level5_rows


def prepare(
    *,
    competition_all_path: Path,
    level5_path: Path,
    aflow_full_path: Path,
    output_path: Path,
    manifest_path: Path,
    targeted_output_path: Path | None = None,
    targeted_test_rows_per_domain: int = 4,
    per_domain: int = 15,
    seed: int = 20260804,
) -> dict[str, Any]:
    """Create the augmented dataset and a count/index based manifest."""
    if per_domain <= 0:
        raise ValueError("per_domain must be positive")

    level5_rows = rebuild_level5_source(competition_all_path, level5_path)
    aflow_rows = _read_jsonl(aflow_full_path)
    original_keys = {_problem_key(row["problem"]) for row in aflow_rows}
    if len(original_keys) != len(aflow_rows):
        raise ValueError("Original AFlow dataset contains duplicate problems")

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    selected_indices: dict[str, list[int]] = {}
    unseen_counts: dict[str, int] = {}
    for domain in TARGET_DOMAINS:
        # De-duplicate the complete source before sampling. The index always
        # points to the first occurrence in the rebuilt Level-5 JSONL.
        unseen_by_problem: dict[str, tuple[int, dict[str, Any]]] = {}
        for source_index, row in enumerate(level5_rows):
            if row.get("type") != domain:
                continue
            key = _problem_key(row["problem"])
            if key in original_keys or key in unseen_by_problem:
                continue
            unseen_by_problem[key] = (source_index, row)

        candidates = list(unseen_by_problem.values())
        unseen_counts[domain] = len(candidates)
        if len(candidates) < per_domain:
            raise ValueError(
                f"Only {len(candidates)} unseen {domain} examples are available"
            )
        rng.shuffle(candidates)
        chosen = candidates[:per_domain]
        selected_indices[domain] = [index for index, _ in chosen]
        for source_index, row in chosen:
            selected.append(
                {
                    **row,
                    "source_split": "validate",
                    "source_index": source_index,
                    "_augmentation_source": SOURCE_LABEL,
                    "_augmentation_seed": seed,
                }
            )

    selected_keys = [_problem_key(row["problem"]) for row in selected]
    if original_keys.intersection(selected_keys):
        raise AssertionError("Selected validation rows overlap the AFlow artifact")
    if len(set(selected_keys)) != len(selected_keys):
        raise AssertionError("Selected validation rows contain duplicates")

    augmented_rows = aflow_rows + selected
    _write_jsonl(output_path, augmented_rows)

    targeted_rows: list[dict[str, Any]] = []
    if targeted_output_path is not None:
        heldout: list[dict[str, Any]] = []
        heldout_rng = random.Random(seed + 1)
        for domain in TARGET_DOMAINS:
            candidates = [
                row
                for row in aflow_rows
                if row.get("type") == domain
                and row.get("_aflow_source_split") == "test"
            ]
            heldout_rng.shuffle(candidates)
            if len(candidates) < targeted_test_rows_per_domain:
                raise ValueError(f"Not enough AFlow test rows for {domain}")
            heldout.extend(candidates[:targeted_test_rows_per_domain])
        targeted_rows = selected + heldout
        _write_jsonl(targeted_output_path, targeted_rows)
    manifest = {
        "schema_version": 1,
        "selection": "seeded shuffle without replacement after exact normalized-problem exclusion",
        "seed": seed,
        "source": {
            "competition_all_path": str(competition_all_path),
            "rebuilt_level5_path": str(level5_path),
            "level5_rows": len(level5_rows),
            "level5_domain_counts": dict(
                sorted(Counter(str(row["type"]) for row in level5_rows).items())
            ),
        },
        "base": {
            "path": str(aflow_full_path),
            "rows": len(aflow_rows),
        },
        "augmentation": {
            "source_split": "validate",
            "domains": list(TARGET_DOMAINS),
            "rows_per_domain": per_domain,
            "rows": len(selected),
            "unseen_available_before_selection": unseen_counts,
            "selected_level5_source_indices": selected_indices,
            "overlap_with_base": 0,
        },
        "output": {
            "path": str(output_path),
            "rows": len(augmented_rows),
        },
    }
    if targeted_output_path is not None:
        manifest["targeted_research_subset"] = {
            "path": str(targeted_output_path),
            "rows": len(targeted_rows),
            "validate_rows": len(selected),
            "test_rows": len(targeted_rows) - len(selected),
            "test_rows_per_domain": targeted_test_rows_per_domain,
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    benchmark_root = Path("/home/rongxing/Benchmark")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--competition-all",
        type=Path,
        default=benchmark_root / "MATH" / "competition_math.jsonl",
    )
    parser.add_argument(
        "--level5-source",
        type=Path,
        default=benchmark_root / "MATH" / "competition_math_level5.jsonl",
    )
    parser.add_argument(
        "--aflow-full",
        type=Path,
        default=(
            benchmark_root / "AFlow" / "math_level5_four_domains_aflow_full.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            benchmark_root
            / "AFlow"
            / "math_level5_four_domains_aflow_full_plus30_validate.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=benchmark_root / "AFlow" / "math_validate_augmentation_manifest.json",
    )
    parser.add_argument(
        "--targeted-output",
        type=Path,
        default=(
            benchmark_root / "AFlow" / "math_pre_failure_research38.jsonl"
        ),
    )
    parser.add_argument("--targeted-test-rows-per-domain", type=int, default=4)
    parser.add_argument("--per-domain", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    result = prepare(
        competition_all_path=args.competition_all,
        level5_path=args.level5_source,
        aflow_full_path=args.aflow_full,
        output_path=args.output,
        manifest_path=args.manifest,
        targeted_output_path=args.targeted_output,
        targeted_test_rows_per_domain=args.targeted_test_rows_per_domain,
        per_domain=args.per_domain,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
