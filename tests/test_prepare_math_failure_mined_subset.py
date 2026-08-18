from __future__ import annotations

import json
from pathlib import Path

from experiments.scripts.prepare_math_failure_mined_subset import build_subset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_failure_mining_keeps_roles_and_test_separate(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    traces = tmp_path / "traces.jsonl"
    _write_jsonl(
        source,
        [
            {"question": "hard", "domain": "Precalculus", "source_split": "validate"},
            {"question": "stable", "domain": "Prealgebra", "source_split": "validate"},
            {"question": "test-a", "domain": "Prealgebra", "source_split": "test"},
            {"question": "test-b", "domain": "Precalculus", "source_split": "test"},
        ],
    )
    _write_jsonl(
        traces,
        [
            {"query_text": "hard", "workflow_version": "1.0", "hard_reward": 0.0},
            {"query_text": "hard", "workflow_version": "1.0", "hard_reward": 0.0},
            {"query_text": "stable", "workflow_version": "1.0", "hard_reward": 1.0},
            {"query_text": "hard", "workflow_version": "1.1", "hard_reward": 1.0},
        ],
    )

    rows, manifest = build_subset(
        source,
        [traces],
        workflow_version="1.0",
        failure_rate=2 / 3,
        max_failures=4,
        stable_guards=1,
        test_rows=2,
    )

    assert [row["research_role"] for row in rows] == [
        "mined_failure",
        "stable_success_guard",
        "untouched_test",
        "untouched_test",
    ]
    assert rows[0]["failure_mining"]["observed_failures"] == 2
    assert manifest["counts"] == {
        "mined_failures": 1,
        "stable_success_guards": 1,
        "validate_total": 2,
        "untouched_test": 2,
        "total": 4,
    }
