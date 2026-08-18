from __future__ import annotations

from experiments.scripts.run_analysis import render_report


def test_analysis_labels_validation_and_test_metrics_by_their_real_objectives():
    report = render_report(
        {
            "config": {"name": "example"},
            "rounds": [],
            "best_val_score": 0.4,
            "checkpoint_summary": {
                "best_round": 0,
                "num_checkpoints": 1,
                "num_checkpoint_updates": 1,
                "num_evaluations": 2,
            },
            "official_test_metrics": {
                "composite_reward": 0.8,
                "runtime_utility": 0.7,
            },
        }
    )

    assert "Best runtime utility: 0.4" in report
    assert "Composite reward: 0.8" in report
