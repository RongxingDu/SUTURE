from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from experiments.scripts.prepare_math_validate_augmentation import prepare


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(problem: str, domain: str, level: str = "Level 5") -> dict[str, str]:
    return {
        "problem": problem,
        "solution": rf"Work. \boxed{{{len(problem)}}}",
        "type": domain,
        "level": level,
    }


def test_prepare_adds_only_unseen_balanced_validate_rows(tmp_path: Path) -> None:
    competition = tmp_path / "competition.jsonl"
    level5 = tmp_path / "level5.jsonl"
    base = tmp_path / "base.jsonl"
    output = tmp_path / "augmented.jsonl"
    manifest = tmp_path / "manifest.json"
    targeted = tmp_path / "targeted.jsonl"
    base_rows = [
        {**_row("shared prealgebra", "Prealgebra"), "_aflow_source_split": "test"},
        {**_row("shared precalculus", "Precalculus"), "_aflow_source_split": "validate"},
        {**_row("test prealgebra", "Prealgebra"), "_aflow_source_split": "test"},
        {**_row("test precalculus", "Precalculus"), "_aflow_source_split": "test"},
    ]
    complete_rows = [
        _row("not level five", "Prealgebra", "Level 4"),
        *[_row(f"prealgebra {index}", "Prealgebra") for index in range(4)],
        *[_row(f"precalculus {index}", "Precalculus") for index in range(4)],
        _row("shared prealgebra", "Prealgebra"),
        _row("shared precalculus", "Precalculus"),
    ]
    _write(competition, complete_rows)
    _write(base, base_rows)

    result = prepare(
        competition_all_path=competition,
        level5_path=level5,
        aflow_full_path=base,
        output_path=output,
        manifest_path=manifest,
        targeted_output_path=targeted,
        targeted_test_rows_per_domain=1,
        per_domain=3,
        seed=7,
    )

    rebuilt = [json.loads(line) for line in level5.read_text().splitlines()]
    augmented = [json.loads(line) for line in output.read_text().splitlines()]
    added = augmented[len(base_rows):]
    targeted_rows = [json.loads(line) for line in targeted.read_text().splitlines()]
    assert len(rebuilt) == 10
    assert len(augmented) == 10
    assert Counter(row["type"] for row in added) == {
        "Prealgebra": 3,
        "Precalculus": 3,
    }
    assert all(row["source_split"] == "validate" for row in added)
    assert not {row["problem"] for row in base_rows}.intersection(
        row["problem"] for row in added
    )
    assert result["augmentation"]["overlap_with_base"] == 0
    assert len(targeted_rows) == 8
    assert sum(row.get("source_split") == "validate" for row in targeted_rows) == 6
    assert sum(row.get("_aflow_source_split") == "test" for row in targeted_rows) == 2
    assert json.loads(manifest.read_text())["output"]["rows"] == 10
