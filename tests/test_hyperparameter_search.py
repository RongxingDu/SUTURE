import json

import pytest

from experiments.scripts.search_hyperparameters import (
    SearchPoint,
    _candidate_records,
    run_search,
)


def _candidate(candidate_id, observations, edit_distance=0.1):
    return {
        "id": candidate_id,
        "scope": "prompt",
        "node_id": "solve",
        "description": candidate_id,
        "edit_distance": edit_distance,
        "observed_selected": False,
        "observations": observations,
    }


def _observation(
    *,
    original_hard,
    candidate_hard,
    original_process=0.5,
    candidate_process=0.5,
    original_tokens=1000,
    candidate_tokens=1000,
    original_latency=5.0,
    candidate_latency=5.0,
):
    return {
        "original_hard": original_hard,
        "candidate_hard": candidate_hard,
        "original_process": original_process,
        "candidate_process": candidate_process,
        "original_total_tokens": original_tokens,
        "candidate_total_tokens": candidate_tokens,
        "original_llm_latency_seconds": original_latency,
        "candidate_llm_latency_seconds": candidate_latency,
    }


def test_search_hard_guard_rejects_success_regression():
    records = [
        _candidate(
            "regression",
            [
                _observation(original_hard=1.0, candidate_hard=0.0),
                _observation(original_hard=0.0, candidate_hard=1.0),
            ],
        )
    ]
    result = run_search(
        records,
        [SearchPoint(0.4, 0.05, 0.01, 0.1, 0.02)],
    )
    decision = result["recommended"]["candidate_decisions"][0]
    assert decision["hard_guard_passed"] is False
    assert decision["screen_accepted"] is False
    assert decision["confirm_accepted"] is False


def test_search_screen_can_explore_while_noisy_lcb_rejects():
    records = [
        _candidate(
            "high-variance",
            [
                _observation(
                    original_hard=0.0,
                    candidate_hard=1.0,
                    candidate_tokens=2500,
                    candidate_latency=15.0,
                ),
                _observation(
                    original_hard=1.0,
                    candidate_hard=1.0,
                    candidate_tokens=900,
                    candidate_latency=4.0,
                ),
            ],
            edit_distance=0.3,
        )
    ]
    result = run_search(
        records,
        [SearchPoint(0.4, 0.05, 0.01, 0.1, 0.02)],
        lcb_z=1.0,
    )
    decision = result["recommended"]["candidate_decisions"][0]
    assert decision["screen_accepted"] is True
    assert decision["confirm_accepted"] is False
    assert "No candidate survives" in result["warning"]


def test_search_accepts_consistent_repeated_improvement():
    records = [
        _candidate(
            "stable",
            [
                _observation(
                    original_hard=0.0,
                    candidate_hard=1.0,
                    candidate_tokens=1100,
                    candidate_latency=5.2,
                )
                for _ in range(3)
            ],
        )
    ]
    result = run_search(
        records,
        [SearchPoint(0.4, 0.05, 0.01, 0.1, 0.02)],
        lcb_z=1.0,
    )
    decision = result["recommended"]["candidate_decisions"][0]
    assert decision["screen_accepted"] is True
    assert decision["confirm_accepted"] is True
    assert decision["leave_one_out_confirm_rate"] == 1.0


def test_search_refuses_run_marked_invalid(tmp_path):
    run_dir = tmp_path / "bad-run"
    run_dir.mkdir()
    results = run_dir / "results.json"
    results.write_text(json.dumps({"rounds": []}))
    (run_dir / "INVALID_RUN.md").write_text("invalid evaluator")

    with pytest.raises(ValueError, match="explicitly marked invalid"):
        _candidate_records([results])
