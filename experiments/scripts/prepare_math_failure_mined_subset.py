#!/usr/bin/env python3
"""Build a research-only MATH subset from observed v1.0 failures.

The output deliberately separates three roles:

* mined failures drive counterfactual workflow updates;
* stable successes provide a small local non-regression guard set;
* untouched source-test rows support an independent smoke test.

This is an optimizer-development set, not an unbiased benchmark estimate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(
    "/home/rongxing/Benchmark/AFlow/"
    "math_level5_four_domains_aflow_full_plus30_validate.jsonl"
)
DEFAULT_TRACES = (
    Path(
        "experiments/results/aliyun_qwen35/math_aliyun_qwen35_full/"
        "traces/optimization.jsonl"
    ),
    Path(
        "experiments/results/aliyun_qwen35/"
        "math_aliyun_qwen35_pre_failure_self_iterate/"
        "traces/optimization.jsonl"
    ),
)
DEFAULT_OUTPUT = Path(
    "/home/rongxing/Benchmark/AFlow/math_failure_mined_research28.jsonl"
)
DEFAULT_MANIFEST = Path(
    "/home/rongxing/Benchmark/AFlow/"
    "math_failure_mined_research28_manifest.json"
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _question(row: dict[str, Any]) -> str:
    return str(row.get("problem", row.get("question", row.get("query", ""))))


def _domain(row: dict[str, Any]) -> Any:
    return row.get("domain", row.get("type"))


def _source_split(row: dict[str, Any]) -> Any:
    return row.get("source_split", row.get("_aflow_source_split"))


def build_subset(
    source_path: Path,
    trace_paths: list[Path],
    *,
    workflow_version: str,
    failure_rate: float,
    max_failures: int,
    stable_guards: int,
    test_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = _read_jsonl(source_path)
    source_by_query = {
        _question(row): row
        for row in source_rows
    }
    evidence: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"runs": 0, "failures": 0, "trace_files": set()}
    )
    for trace_path in trace_paths:
        for trace in _read_jsonl(trace_path):
            if str(trace.get("workflow_version")) != workflow_version:
                continue
            query = str(trace.get("query_text", ""))
            if not query or query not in source_by_query:
                continue
            item = evidence[query]
            item["runs"] += 1
            try:
                failed = float(trace.get("hard_reward", 0.0)) < 1.0
            except (TypeError, ValueError, OverflowError):
                failed = True
            item["failures"] += int(failed)
            item["trace_files"].add(str(trace_path))

    ranked_failures = []
    stable = []
    for query, item in evidence.items():
        runs = int(item["runs"])
        failures = int(item["failures"])
        rate = failures / runs if runs else 0.0
        record = (query, runs, failures, rate, item)
        if failures and rate >= failure_rate:
            ranked_failures.append(record)
        elif runs and failures == 0:
            stable.append(record)
    ranked_failures.sort(key=lambda item: (-item[3], -item[1], item[0]))
    stable.sort(key=lambda item: (-item[1], item[0]))
    ranked_failures = ranked_failures[:max_failures]
    stable = stable[:stable_guards]

    selected_queries = {item[0] for item in ranked_failures + stable}
    output: list[dict[str, Any]] = []
    for role, selected in (
        ("mined_failure", ranked_failures),
        ("stable_success_guard", stable),
    ):
        for query, runs, failures, rate, item in selected:
            row = dict(source_by_query[query])
            row["source_split"] = "validate"
            row["_aflow_source_split"] = "validate"
            row["research_role"] = role
            row["failure_mining"] = {
                "workflow_version": workflow_version,
                "observed_runs": runs,
                "observed_failures": failures,
                "observed_failure_rate": rate,
                "trace_files": sorted(item["trace_files"]),
            }
            output.append(row)

    test_candidates = [
        row
        for row in source_rows
        if _source_split(row) == "test"
        and _question(row) not in selected_queries
    ]
    # Prefer the two domains explicitly targeted by this iteration and then
    # fill deterministically from any remaining Level-5 source-test rows.
    preferred_domains = ("Prealgebra", "Precalculus")
    selected_test: list[dict[str, Any]] = []
    per_domain_target = test_rows // len(preferred_domains)
    for domain in preferred_domains:
        domain_rows = [
            row
            for row in test_candidates
            if _domain(row) == domain
        ]
        selected_test.extend(domain_rows[:per_domain_target])
    selected_test_queries = {
        _question(row)
        for row in selected_test
    }
    if len(selected_test) < test_rows:
        selected_test.extend(
            row
            for row in test_candidates
            if _question(row) not in selected_test_queries
        )
    for source_row in selected_test[:test_rows]:
        row = dict(source_row)
        row["source_split"] = "test"
        row["_aflow_source_split"] = "test"
        row["research_role"] = "untouched_test"
        output.append(row)

    manifest = {
        "purpose": "research-only failure-mined workflow self-iteration",
        "unbiased_benchmark": False,
        "source": str(source_path),
        "trace_sources": [str(path) for path in trace_paths],
        "workflow_version": workflow_version,
        "failure_rate_threshold": failure_rate,
        "counts": {
            "mined_failures": len(ranked_failures),
            "stable_success_guards": len(stable),
            "validate_total": len(ranked_failures) + len(stable),
            "untouched_test": min(test_rows, len(selected_test)),
            "total": len(output),
        },
        "mined_failure_evidence": [
            {
                "question": query,
                "domain": _domain(source_by_query[query]),
                "runs": runs,
                "failures": failures,
                "failure_rate": rate,
            }
            for query, runs, failures, rate, _ in ranked_failures
        ],
    }
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--trace", type=Path, action="append", dest="traces")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workflow-version", default="1.0")
    parser.add_argument("--failure-rate", type=float, default=2 / 3)
    parser.add_argument("--max-failures", type=int, default=16)
    parser.add_argument("--stable-guards", type=int, default=4)
    parser.add_argument("--test-rows", type=int, default=8)
    args = parser.parse_args()
    trace_paths = args.traces or list(DEFAULT_TRACES)
    rows, manifest = build_subset(
        args.source,
        trace_paths,
        workflow_version=args.workflow_version,
        failure_rate=args.failure_rate,
        max_failures=args.max_failures,
        stable_guards=args.stable_guards,
        test_rows=args.test_rows,
    )
    _write_jsonl(args.output, rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
