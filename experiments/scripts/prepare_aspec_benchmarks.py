#!/usr/bin/env python3
"""Prepare reproducible GPQA, MMLU and SciCode pilot artifacts.

The local ASpec checkout is used only as a convenient data mirror.  Every
output is minimized to the fields needed by AWF, attributed to the official
upstream benchmark, and accompanied by explicit notes about ASpec's custom
splits.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_SELECTION_SEED = 20260726
DEFAULT_PILOT_SIZE = 20
SCICODE_HDF5_FILE_ID = "17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW"
SCICODE_HDF5_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR"
)


def prepare(
    aspec_dir: Path,
    benchmark_dir: Path,
    *,
    seed: int = DEFAULT_SELECTION_SEED,
    pilot_size: int = DEFAULT_PILOT_SIZE,
) -> dict[str, Any]:
    if pilot_size < 5 or pilot_size % 5 != 0:
        raise ValueError("pilot_size must be a positive multiple of 5")
    heldout_size = pilot_size // 5
    development_size = pilot_size - heldout_size
    source_dir = aspec_dir / "data"
    source_paths = {
        "gpqa_train": source_dir / "gpqa_train.jsonl",
        "gpqa_test": source_dir / "gpqa_test.jsonl",
        "mmlu_train": source_dir / "mmlu_train.jsonl",
        "mmlu_test": source_dir / "mmlu_test.jsonl",
        "scicode_train": source_dir / "scicode_train.jsonl",
        "scicode_test": source_dir / "scicode_test.jsonl",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing ASpec benchmark files: " + ", ".join(missing)
        )

    raw = {name: _read_jsonl(path) for name, path in source_paths.items()}
    _validate_source_counts(raw)

    gpqa_train = [
        _minimal_gpqa(row, "train", index)
        for index, row in enumerate(raw["gpqa_train"])
    ]
    gpqa_test = [
        _minimal_gpqa(row, "test", index)
        for index, row in enumerate(raw["gpqa_test"])
    ]
    mmlu_train = [
        _minimal_mmlu(row, "train", index)
        for index, row in enumerate(raw["mmlu_train"])
    ]
    mmlu_test = [
        _minimal_mmlu(row, "test", index)
        for index, row in enumerate(raw["mmlu_test"])
    ]
    scicode_train = [
        _minimal_scicode(row, "train", index)
        for index, row in enumerate(raw["scicode_train"])
    ]
    scicode_test = [
        _minimal_scicode(row, "test", index)
        for index, row in enumerate(raw["scicode_test"])
    ]

    outputs = {
        "gpqa_full": (
            benchmark_dir / "QA" / "GPQA" / "gpqa_aspec_main_448.jsonl"
        ),
        "gpqa_pilot": (
            benchmark_dir / "QA" / "GPQA" / f"gpqa_pilot{pilot_size}.jsonl"
        ),
        "mmlu_full": (
            benchmark_dir
            / "QA"
            / "MMLU"
            / "mmlu_aspec_20subjects_500.jsonl"
        ),
        "mmlu_pilot": (
            benchmark_dir / "QA" / "MMLU" / f"mmlu_pilot{pilot_size}.jsonl"
        ),
        "scicode_full": (
            benchmark_dir
            / "Coding"
            / "SciCode"
            / "scicode_aspec_80.jsonl"
        ),
        "scicode_pilot": (
            benchmark_dir
            / "Coding"
            / "SciCode"
            / f"scicode_first_subproblem_pilot{pilot_size}.jsonl"
        ),
    }

    gpqa_full = gpqa_train + gpqa_test
    mmlu_full = mmlu_train + mmlu_test
    scicode_full = scicode_train + scicode_test
    gpqa_pilot = _select_gpqa(gpqa_train, gpqa_test, seed,
                              dev_size=development_size, heldout_size=heldout_size)
    mmlu_pilot = _select_mmlu(mmlu_train, mmlu_test, seed,
                              dev_size=development_size, heldout_size=heldout_size)
    # All 16 ASpec training problems contribute their first subproblem.
    # Additional source-test problems are selected without looking at model
    # outcomes.
    eligible_scicode_test = [
        row
        for row in scicode_test
        if row.get("sub_steps")
        and str(row["sub_steps"][0].get("step_number"))
        not in {"13.6", "62.1", "76.3"}
        and row["sub_steps"][0].get("test_cases")
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible_scicode_test)
    needed_test = max(0, pilot_size - len(scicode_train))
    if needed_test > len(eligible_scicode_test):
        raise ValueError(
            f"Cannot form SciCode pilot of size {pilot_size}: "
            f"only {len(scicode_train)} train + {len(eligible_scicode_test)} "
            f"eligible test rows available"
        )
    scicode_pilot = scicode_train + eligible_scicode_test[:needed_test]

    for path, rows in (
        (outputs["gpqa_full"], gpqa_full),
        (outputs["gpqa_pilot"], gpqa_pilot),
        (outputs["mmlu_full"], mmlu_full),
        (outputs["mmlu_pilot"], mmlu_pilot),
        (outputs["scicode_full"], scicode_full),
        (outputs["scicode_pilot"], scicode_pilot),
    ):
        _write_jsonl(path, rows)

    manifest = {
        "schema_version": 1,
        "selection": {
            "algorithm": "random shuffle without replacement",
            "seed": seed,
            "pilot_size": pilot_size,
            "split_sizes": {
                k: v
                for k, v in {
                    "optimization": int(pilot_size * 0.6),
                    "validation": int(pilot_size * 0.2),
                    "test": pilot_size
                    - int(pilot_size * 0.6)
                    - int(pilot_size * 0.2),
                }.items()
                if v > 0
            },
            "source_boundary": (
                "optimization/validation use only ASpec train rows; held-out "
                "test uses only ASpec test rows"
            ),
        },
        "source": {
            "local_repository": str(aspec_dir.resolve()),
            "git_commit": _git_commit(aspec_dir),
            "files": {
                name: {
                    "path": str(path.resolve()),
                    "rows": len(raw[name]),
                }
                for name, path in source_paths.items()
            },
        },
        "upstream": {
            "gpqa": {
                "url": "https://github.com/idavidrein/gpqa",
                "license": "MIT repository; dataset card CC-BY-4.0",
                "protocol_note": (
                    "ASpec's 89/359 split is custom; this is the 448-question "
                    "main set, not GPQA Diamond."
                ),
            },
            "mmlu": {
                "url": "https://github.com/hendrycks/test",
                "license": "MIT",
                "protocol_note": (
                    "ASpec contains a custom 20-subject, 500-row subset of "
                    "the official 57-subject benchmark."
                ),
            },
            "scicode": {
                "url": "https://github.com/scicode-bench/SciCode",
                "license": "Apache-2.0",
                "hdf5_folder_url": SCICODE_HDF5_FOLDER_URL,
                "hdf5_file_id": SCICODE_HDF5_FILE_ID,
                "protocol_note": (
                    "ASpec's 16/64 split differs from the current official "
                    "15/65 split. The pilot evaluates only each problem's "
                    "first subproblem, matching the first call of the official "
                    "sequential protocol."
                ),
                "official_exclusions": ["13.6", "62.1", "76.3"],
            },
        },
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "rows": _count_jsonl(path),
            }
            for name, path in outputs.items()
        },
    }
    hdf5_path = (
        benchmark_dir / "Coding" / "SciCode" / "test_data.h5"
    )
    if hdf5_path.is_file():
        manifest["outputs"]["scicode_hdf5"] = {
            "path": str(hdf5_path.resolve()),
            "size_bytes": hdf5_path.stat().st_size,
        }
    manifest_path = benchmark_dir / "ASpec" / "selection_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _validate_source_counts(raw: dict[str, list[dict[str, Any]]]) -> None:
    expected = {
        "gpqa_train": 89,
        "gpqa_test": 359,
        "mmlu_train": 100,
        "mmlu_test": 400,
        "scicode_train": 16,
        "scicode_test": 64,
    }
    actual = {name: len(rows) for name, rows in raw.items()}
    if actual != expected:
        raise ValueError(
            f"ASpec dataset counts changed: expected {expected}, got {actual}"
        )


def _minimal_gpqa(
    row: dict[str, Any],
    source_split: str,
    source_index: int,
) -> dict[str, Any]:
    fields = (
        "Record ID",
        "Question",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
        "High-level domain",
        "Subdomain",
    )
    result = {field: row.get(field) for field in fields}
    result["_source_split"] = source_split
    result["_source_index"] = source_index
    return result


def _minimal_mmlu(
    row: dict[str, Any],
    source_split: str,
    source_index: int,
) -> dict[str, Any]:
    result = {
        field: row.get(field)
        for field in ("type", "subject", "question", "choices", "answer")
    }
    result["sample_id"] = (
        f"{row.get('subject', '')}:{source_split}:{source_index}"
    )
    result["_source_split"] = source_split
    result["_source_index"] = source_index
    return result


def _minimal_scicode(
    row: dict[str, Any],
    source_split: str,
    source_index: int,
) -> dict[str, Any]:
    result = {
        field: row.get(field)
        for field in (
            "problem_id",
            "problem_name",
            "problem_description_main",
            "problem_background_main",
            "problem_io",
            "required_dependencies",
            "sub_steps",
            "general_tests",
        )
        if field in row
    }
    result["_source_split"] = source_split
    result["_source_index"] = source_index
    return result


def _select_gpqa(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    seed: int,
    *,
    dev_size: int = 16,
    heldout_size: int = 4,
) -> list[dict[str, Any]]:
    return _stratified_select(
        train,
        dev_size,
        seed,
        group=lambda row: str(row.get("High-level domain")),
        key=lambda row: str(row.get("Record ID") or row.get("Question")),
    ) + _stratified_select(
        test,
        heldout_size,
        seed + 1,
        group=lambda row: str(row.get("High-level domain")),
        key=lambda row: str(row.get("Record ID") or row.get("Question")),
    )


def _select_mmlu(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    seed: int,
    *,
    dev_size: int = 16,
    heldout_size: int = 4,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    train_subjects = sorted(
        {str(row.get("subject")) for row in train}
    )
    rng.shuffle(train_subjects)
    development_subjects = train_subjects[:dev_size]
    development = []
    for subject in development_subjects:
        candidates = [
            row for row in train if str(row.get("subject")) == subject
        ]
        rng.shuffle(candidates)
        development.append(candidates[0])
    heldout_subjects = train_subjects[:heldout_size]
    heldout = []
    for subject in heldout_subjects:
        candidates = [
            row for row in test if str(row.get("subject")) == subject
        ]
        rng.shuffle(candidates)
        heldout.append(candidates[0])
    return development + heldout


def _stratified_select(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    *,
    group: Callable[[dict[str, Any]], str],
    key: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group(row)].append(row)
    groups = sorted(grouped)
    allocation = {name: count // len(groups) for name in groups}
    remainder = count - sum(allocation.values())
    rng = random.Random(seed)
    ranked_groups = list(groups)
    rng.shuffle(ranked_groups)
    for name in ranked_groups[:remainder]:
        allocation[name] += 1
    selected: list[dict[str, Any]] = []
    for name in groups:
        selected.extend(
            _random_select(
                grouped[name],
                allocation[name],
                seed,
                key=key,
            )
        )
    return selected


def _random_select(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    *,
    key: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    if count < 0 or count > len(rows):
        raise ValueError("Invalid selection count")
    ordered = sorted(rows, key=key)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered[:count]


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
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aspec-dir",
        type=Path,
        default=Path("/home/rongxing/ASpec"),
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("/home/rongxing/Benchmark"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=DEFAULT_PILOT_SIZE,
        help="Number of rows in the pilot subset (positive multiple of 5)",
    )
    args = parser.parse_args()
    manifest = prepare(
        args.aspec_dir,
        args.benchmark_dir,
        seed=args.seed,
        pilot_size=args.pilot_size,
    )
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    main()
