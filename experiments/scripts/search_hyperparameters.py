#!/usr/bin/env python3
"""Offline stability/exploration search over persisted counterfactual telemetry.

The script never calls an LLM and never reads a benchmark split. It recomputes
candidate effects from the per-query hard/token deltas already stored in one or
more optimization ``results.json`` artifacts.  Process and latency columns
are retained in persisted telemetry for reporting, but are not part of gain.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SearchPoint:
    alpha_process: float
    token_penalty_per_1k: float
    latency_penalty_per_second: float
    mu_edit: float
    epsilon: float

    # ``alpha_process`` and ``latency_penalty_per_second`` remain in the
    # serialized grid schema so old pilot files can still be read.  They are
    # intentionally ignored by the current reward-only utility/gain policy.


def _float_grid(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("grid cannot be empty")
    return parsed


def _candidate_records(
    result_paths: Iterable[Path],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result_path in result_paths:
        invalid_marker = result_path.parent / "INVALID_RUN.md"
        if invalid_marker.exists():
            raise ValueError(
                "refusing candidate telemetry from a run explicitly marked "
                f"invalid: {invalid_marker}"
            )
        artifact = json.loads(result_path.read_text())
        for round_result in artifact.get("rounds", []):
            for candidate in round_result.get("candidate_evaluations", []):
                observations = candidate.get("per_query_results") or []
                if not observations:
                    continue
                records.append(
                    {
                        "id": (
                            f"{result_path}:{round_result.get('round')}:"
                            f"{candidate.get('candidate_index')}"
                        ),
                        "source": str(result_path),
                        "round": round_result.get("round"),
                        "candidate_index": candidate.get("candidate_index"),
                        "scope": candidate.get("scope"),
                        "node_id": candidate.get("node_id"),
                        "description": candidate.get("description", ""),
                        "edit_distance": float(
                            candidate.get("edit_distance", 0.0)
                        ),
                        "observed_selected": bool(
                            candidate.get("final_selected", False)
                        ),
                        "observations": observations,
                    }
                )
    return records


def _observation_effect(
    observation: dict[str, Any],
    point: SearchPoint,
) -> float:
    hard_delta = float(observation["candidate_hard"]) - float(
        observation["original_hard"]
    )
    token_delta = float(observation["candidate_total_tokens"]) - float(
        observation["original_total_tokens"]
    )
    return (
        hard_delta
        - point.token_penalty_per_1k * token_delta / 1000.0
    )


def _hard_guard_passes(
    observations: list[dict[str, Any]],
    tolerance: float,
) -> bool:
    for observation in observations:
        original = float(observation["original_hard"])
        candidate = float(observation["candidate_hard"])
        if original >= 0.5 and candidate < original - tolerance:
            return False
    mean_delta = statistics.fmean(
        float(item["candidate_hard"]) - float(item["original_hard"])
        for item in observations
    )
    return mean_delta >= -tolerance


def _evaluate_candidate(
    candidate: dict[str, Any],
    point: SearchPoint,
    *,
    lcb_z: float,
    hard_tolerance: float,
    min_confirm_queries: int,
) -> dict[str, Any]:
    observations = candidate["observations"]
    effects = [
        _observation_effect(observation, point)
        for observation in observations
    ]
    mean_effect = statistics.fmean(effects)
    standard_error = (
        statistics.stdev(effects) / math.sqrt(len(effects))
        if len(effects) > 1
        else math.inf
    )
    edit_penalty = point.mu_edit * candidate["edit_distance"]
    screen_score = mean_effect - edit_penalty
    confirm_score = (
        mean_effect - lcb_z * standard_error - edit_penalty
        if math.isfinite(standard_error)
        else -math.inf
    )
    hard_guard = _hard_guard_passes(observations, hard_tolerance)
    screen_accepted = hard_guard and screen_score > point.epsilon
    confirm_accepted = (
        screen_accepted
        and len(effects) >= min_confirm_queries
        and confirm_score > point.epsilon
    )

    leave_one_out: list[bool] = []
    if len(effects) > 2:
        for excluded in range(len(effects)):
            subset = effects[:excluded] + effects[excluded + 1 :]
            subset_mean = statistics.fmean(subset)
            subset_se = statistics.stdev(subset) / math.sqrt(len(subset))
            subset_score = subset_mean - lcb_z * subset_se - edit_penalty
            leave_one_out.append(subset_score > point.epsilon)

    return {
        "candidate_id": candidate["id"],
        "scope": candidate["scope"],
        "node_id": candidate["node_id"],
        "description": candidate["description"],
        "num_queries": len(effects),
        "hard_guard_passed": hard_guard,
        "mean_effect": mean_effect,
        "standard_error": (
            standard_error if math.isfinite(standard_error) else None
        ),
        "edit_penalty": edit_penalty,
        "screen_score": screen_score,
        "confirm_lcb_score": (
            confirm_score if math.isfinite(confirm_score) else None
        ),
        "screen_accepted": screen_accepted,
        "confirm_accepted": confirm_accepted,
        "leave_one_out_confirm_rate": (
            statistics.fmean(leave_one_out) if leave_one_out else None
        ),
        "observed_selected": candidate["observed_selected"],
    }


def _point_dict(point: SearchPoint) -> dict[str, float]:
    return {
        "alpha_process": point.alpha_process,
        "token_penalty_per_1k": point.token_penalty_per_1k,
        "latency_penalty_per_second": point.latency_penalty_per_second,
        "mu_edit": point.mu_edit,
        "epsilon": point.epsilon,
    }


def run_search(
    records: list[dict[str, Any]],
    points: Iterable[SearchPoint],
    *,
    lcb_z: float = 1.0,
    hard_tolerance: float = 0.0,
    min_confirm_queries: int = 3,
    target_screen_rate: float = 0.25,
) -> dict[str, Any]:
    """Evaluate and rank grid points without making causal claims."""
    if not records:
        raise ValueError("no persisted candidate observations were found")

    ranked: list[dict[str, Any]] = []
    for point in points:
        decisions = [
            _evaluate_candidate(
                candidate,
                point,
                lcb_z=lcb_z,
                hard_tolerance=hard_tolerance,
                min_confirm_queries=min_confirm_queries,
            )
            for candidate in records
        ]
        count = len(decisions)
        screen_count = sum(item["screen_accepted"] for item in decisions)
        confirm_count = sum(item["confirm_accepted"] for item in decisions)
        guard_rejections = sum(
            not item["hard_guard_passed"] for item in decisions
        )
        screen_rate = screen_count / count
        confirm_rate = confirm_count / count
        stable_confirm_count = sum(
            item["confirm_accepted"]
            and (
                item["leave_one_out_confirm_rate"] is None
                or item["leave_one_out_confirm_rate"] >= 0.8
            )
            for item in decisions
        )
        # The screen should explore a bounded fraction of candidates, while
        # deployment confirmation remains conservative and guard-safe.
        balance_score = (
            -abs(screen_rate - target_screen_rate)
            + 0.30 * stable_confirm_count
            + 0.05 * confirm_count
            - 0.50 * guard_rejections
        )
        ranked.append(
            {
                "parameters": _point_dict(point),
                "screen_accept_count": screen_count,
                "screen_accept_rate": screen_rate,
                "confirm_accept_count": confirm_count,
                "confirm_accept_rate": confirm_rate,
                "stable_confirm_count": stable_confirm_count,
                "hard_guard_rejection_count": guard_rejections,
                "balance_score": balance_score,
                "candidate_decisions": decisions,
            }
        )

    # Prefer central values when empirical scores tie. This makes the result
    # deterministic without pretending the tiny pilot identified fine-grained
    # coefficient differences.
    prior = SearchPoint(0.4, 0.05, 0.01, 0.1, 0.02)

    def central_distance(item: dict[str, Any]) -> float:
        values = item["parameters"]
        return sum(
            abs(values[key] - reference)
            for key, reference in _point_dict(prior).items()
        )

    ranked.sort(
        key=lambda item: (
            -item["balance_score"],
            central_distance(item),
            json.dumps(item["parameters"], sort_keys=True),
        )
    )
    recommendation = ranked[0]
    all_confirm_zero = all(
        item["confirm_accept_count"] == 0 for item in ranked
    )
    return {
        "schema_version": 1,
        "candidate_record_count": len(records),
        "grid_point_count": len(ranked),
        "policy": {
            "effect": (
                "delta_hard - token_penalty_per_1k*delta_tokens/1000"
            ),
            "utility": "hard_reward",
            "legacy_ignored_parameters": [
                "alpha_process",
                "latency_penalty_per_second",
            ],
            "screen": "mean(effect) - mu_edit*edit_distance > epsilon",
            "confirm": (
                "mean(effect) - lcb_z*SE(effect) "
                "- mu_edit*edit_distance > epsilon"
            ),
            "hard_non_regression_guard": True,
            "lcb_z": lcb_z,
            "hard_tolerance": hard_tolerance,
            "min_confirm_queries": min_confirm_queries,
            "target_screen_rate": target_screen_rate,
        },
        "recommended": recommendation,
        "warning": (
            "No candidate survives the confirmation LCB anywhere in the "
            "searched grid; collect repeated/full-opt confirmation data "
            "before treating coefficients as identified."
            if all_confirm_zero
            else (
                "This is an offline sensitivity result from persisted pilot "
                "telemetry, not a held-out benchmark estimate."
            )
        ),
        "top_grid_points": ranked[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search stability/exploration coefficients using persisted "
            "counterfactual telemetry only"
        )
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        type=Path,
        help="One or more optimization results.json artifacts",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--alpha-process",
        type=_float_grid,
        default=[0.2, 0.4, 0.6],
        help="Legacy grid field retained for artifact compatibility (ignored)",
    )
    parser.add_argument(
        "--token-penalty-per-1k",
        type=_float_grid,
        default=[0.025, 0.05, 0.1],
    )
    parser.add_argument(
        "--latency-penalty-per-second",
        type=_float_grid,
        default=[0.005, 0.01, 0.02],
        help="Legacy grid field retained for artifact compatibility (ignored)",
    )
    parser.add_argument(
        "--mu-edit",
        type=_float_grid,
        # The active failure-repair policy treats edit distance as a light
        # preference after suffix-replay correctness, so search the lower
        # coefficient regime by default.
        default=[0.01, 0.02, 0.05],
    )
    parser.add_argument(
        "--epsilon",
        type=_float_grid,
        default=[0.01, 0.02, 0.05],
    )
    parser.add_argument("--lcb-z", type=float, default=1.0)
    parser.add_argument("--min-confirm-queries", type=int, default=3)
    parser.add_argument("--target-screen-rate", type=float, default=0.25)
    args = parser.parse_args()

    records = _candidate_records(args.results)
    points = (
        SearchPoint(*values)
        for values in itertools.product(
            args.alpha_process,
            args.token_penalty_per_1k,
            args.latency_penalty_per_second,
            args.mu_edit,
            args.epsilon,
        )
    )
    result = run_search(
        records,
        points,
        lcb_z=args.lcb_z,
        min_confirm_queries=args.min_confirm_queries,
        target_screen_rate=args.target_screen_rate,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
