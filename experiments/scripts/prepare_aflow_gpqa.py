#!/usr/bin/env python3
"""Prepare the GPQA *optimization-only* file consumed by local AFlow.

The source is the minimized 20-row pilot in ``Benchmark``.  AFlow receives
exactly the same twelve optimization rows used by AWF.  Shared validation and
held-out test labels are never written into the AFlow checkout; they remain
private to the later comparison scorer.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from benchmarks.multiple_choice.dataset import load_gpqa


DEFAULT_PILOT = Path(
    "/home/rongxing/Benchmark/QA/GPQA/gpqa_pilot20.jsonl",
)
DEFAULT_SELECTION_MANIFEST = Path(
    "/home/rongxing/Benchmark/ASpec/selection_manifest.json",
)
DEFAULT_AFLOW_DIR = Path("/home/rongxing/AFlow")
DEFAULT_RESEARCH_MANIFEST = Path(
    "/home/rongxing/Benchmark/AFlow/GPQA/aflow_gpqa_manifest.json",
)
_FORBIDDEN_SOURCE_KEYS = {
    "Explanation",
    "Canary String",
    "Canary",
    "Validator Accuracy",
    "Validator Confidence",
    "Writer's Difficulty Estimate",
    "Writer's Difficulty",
}


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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _assert_source_minimized(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        forbidden = _FORBIDDEN_SOURCE_KEYS.intersection(row)
        if forbidden:
            raise ValueError(
                f"GPQA pilot row {index} contains private fields: "
                + ", ".join(sorted(forbidden))
            )


def _compute_split_indices(
    pilot_rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, list[int]]:
    """Compute optimization/validation/test split from pilot rows and seed.

    Pilot rows from the training source are split into optimization (12)
    and validation (4).  Pilot rows from the test source all go to the
    held-out test split (4).
    """
    train_indices = [
        i for i, row in enumerate(pilot_rows)
        if row.get("_source_split") == "train"
    ]
    test_indices = [
        i for i, row in enumerate(pilot_rows)
        if row.get("_source_split") == "test"
    ]
    if len(train_indices) != 16 or len(test_indices) != 4:
        raise ValueError(
            f"GPQA pilot source split is {len(train_indices)}+"
            f"{len(test_indices)}, expected 16+4"
        )
    rng = random.Random(seed)
    rng.shuffle(train_indices)
    return {
        "optimization": sorted(train_indices[:12]),
        "validation": sorted(train_indices[12:]),
        "test": sorted(test_indices),
    }


def prepare(
    *,
    pilot_path: Path = DEFAULT_PILOT,
    selection_manifest_path: Path = DEFAULT_SELECTION_MANIFEST,
    aflow_dir: Path = DEFAULT_AFLOW_DIR,
    research_manifest_path: Path = DEFAULT_RESEARCH_MANIFEST,
) -> dict[str, Any]:
    raw_rows = _read_jsonl(pilot_path)
    if len(raw_rows) != 20:
        raise ValueError(f"Expected 20 GPQA pilot rows, found {len(raw_rows)}")
    _assert_source_minimized(raw_rows)

    pairs = load_gpqa(pilot_path)
    if len(pairs) != len(raw_rows):
        raise ValueError("GPQA loader changed the pilot row count")
    selection_manifest = json.loads(
        selection_manifest_path.read_text(encoding="utf-8"),
    )
    split_indices = _compute_split_indices(
        raw_rows,
        selection_manifest["selection"]["seed"],
    )

    def convert(index: int) -> dict[str, Any]:
        prompt, target = pairs[index]
        if not prompt.rstrip().endswith(
            "Replace X with A, B, C, or D.",
        ):
            raise ValueError(f"GPQA row {index} lacks the strict public suffix")
        answer = target.get("answer_letter")
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"GPQA row {index} has an invalid answer target")
        return {
            "_pilot_index": index,
            "_source_split": target.get("source_split"),
            "answer": answer,
            "formatter_version": target["formatter_version"],
            "question": prompt,
            "sample_id": target["sample_id"],
        }

    optimization_rows = [
        convert(index)
        for index in split_indices["optimization"]
    ]
    if any(row["_source_split"] == "test" for row in optimization_rows):
        raise ValueError("Source-test GPQA row leaked into AFlow optimization")

    data_dir = aflow_dir / "data" / "datasets"
    outputs = {
        # AFlow's upstream evaluator calls its search/fitness file
        # ``validate``.  In this comparison protocol its scientific role is
        # strictly the shared optimization split.
        "fitness": data_dir / "gpqa_validate.jsonl",
    }
    _write_jsonl(outputs["fitness"], optimization_rows)
    forbidden_test_path = data_dir / "gpqa_test.jsonl"
    if forbidden_test_path.exists():
        raise ValueError(
            "AFlow checkout contains gpqa_test.jsonl; remove this leaked "
            "held-out-label file before search"
        )

    manifest = {
        "schema_version": 1,
        "benchmark": "GPQA main set, ASpec-derived pilot20",
        "protocol": {
            "answer_contract": "terminal `The final answer is: X`, X in A-D",
            "choice_order": "AWF deterministic permutation v1",
            "fitness_rows": (
                "shared optimization only (12 rows; AFlow upstream filename "
                "is gpqa_validate.jsonl)"
            ),
            "validation_rows": (
                "not materialized in AFlow; private comparison scorer only"
            ),
            "heldout_rows": (
                "not materialized in AFlow; private comparison scorer only"
            ),
            "private_field_policy": (
                "No explanation, canary, validator metadata, author metadata, "
                "or answer text is copied."
            ),
        },
        "source": {
            "pilot_path": str(pilot_path.resolve()),
            "selection_manifest_path": str(
                selection_manifest_path.resolve(),
            ),
            "pilot_selection_seed": selection_manifest["selection"]["seed"],
        },
        "split_indices": split_indices,
        "outputs": {
            split: {
                "path": str(path.resolve()),
                "rows": len(optimization_rows),
            }
            for split, path in outputs.items()
        },
    }
    research_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    research_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
    )
    parser.add_argument("--aflow-dir", type=Path, default=DEFAULT_AFLOW_DIR)
    parser.add_argument(
        "--research-manifest",
        type=Path,
        default=DEFAULT_RESEARCH_MANIFEST,
    )
    args = parser.parse_args()
    manifest = prepare(
        pilot_path=args.pilot,
        selection_manifest_path=args.selection_manifest,
        aflow_dir=args.aflow_dir,
        research_manifest_path=args.research_manifest,
    )
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    main()
