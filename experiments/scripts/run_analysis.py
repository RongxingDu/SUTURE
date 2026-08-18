#!/usr/bin/env python3
"""Render a concise report from an AWF ``results.json`` artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--results",
        "-r",
        required=True,
        help="Path to results.json file",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path for the rendered text report",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path) as handle:
        results = json.load(handle)
    report = render_report(results)
    print(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n")


def render_report(results: dict[str, Any]) -> str:
    """Return a stable human-readable experiment report."""
    lines = [
        "=" * 60,
        "Experiment Analysis",
        "=" * 60,
    ]
    config = results.get("config", {})
    lines.extend(
        [
            "",
            f"Experiment: {config.get('name', 'unknown')}",
            f"Start time: {results.get('start_time', 'unknown')}",
            f"End time: {results.get('end_time', 'unknown')}",
        ]
    )

    rounds = results.get("rounds", [])
    lines.extend(["", f"Optimization rounds: {len(rounds)}"])
    if rounds:
        lines.extend(
            [
                "",
                "Round-by-round validation utility:",
                f"{'Round':>6} {'Accepted':>10} {'Gain':>10} "
                f"{'Val utility':>12} {'Best':>6}",
                "-" * 54,
            ]
        )
        for round_result in rounds:
            accepted = "Yes" if round_result.get("accepted") else "No"
            best = "*" if round_result.get("is_best") else ""
            lines.append(
                f"{round_result['round']:>6} {accepted:>10} "
                f"{round_result.get('gain', 0):>10.4f} "
                f"{round_result.get('val_score', 0):>12.4f} {best:>6}"
            )

    checkpoint = results.get("checkpoint_summary", {})
    lines.extend(
        [
            "",
            "Validation checkpoint:",
            f"  Best runtime utility: {results.get('best_val_score')}",
            f"  Best round: {checkpoint.get('best_round')}",
            f"  Current checkpoint files: {checkpoint.get('num_checkpoints')}",
            "  Checkpoint improvements: "
            f"{checkpoint.get('num_checkpoint_updates')}",
            f"  Validation evaluations: {checkpoint.get('num_evaluations')}",
        ]
    )

    test_metrics = (
        results.get("official_test_metrics")
        or results.get("test_metrics")
        or {}
    )
    lines.extend(["", "Official held-out test:"])
    if not test_metrics:
        lines.append("  Not evaluated")
    else:
        lines.append(
            "  Composite reward: "
            f"{test_metrics.get('composite_reward')}"
        )
        for key in (
            "hard_success_rate",
            "hard_reward",
            "process_reward",
            "runtime_utility",
            "total_tokens",
            "llm_call_count",
            "tool_call_count",
            "latency_seconds",
            "estimated_api_cost_usd",
            "api_cost_estimate_complete",
            "repair_count",
            "reroute_count",
            "fallback_count",
            "loop_count",
        ):
            if key in test_metrics:
                lines.append(f"  {key}: {test_metrics[key]}")

    accepted_rounds = [
        round_result
        for round_result in rounds
        if round_result.get("accepted")
    ]
    lines.extend(
        [
            "",
            f"Accepted edits: {len(accepted_rounds)}/{len(rounds)}",
        ]
    )
    for round_result in accepted_rounds:
        lines.extend(
            [
                f"  Round {round_result['round']}: "
                f"{round_result.get('candidate_description', 'N/A')}",
                f"    Scope: {round_result.get('candidate_scope', 'N/A')}, "
                f"Node: {round_result.get('candidate_node_id', 'N/A')}, "
                f"Gain: {round_result.get('gain', 0):.4f}",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
