#!/usr/bin/env python3
"""Compare Vanilla / AWF optimized / AFlow methods on the same held-out test split."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from awf.config.loader import load_config
from awf.config.schema import RewardConfig, ExecutorConfig
from awf.executor.runtime import RuntimeExecutor
from awf.llm.client import AsyncLLMClient
from awf.reward.base import RewardEvaluator
from awf.scheduler.graph_scheduler import GraphScheduler
from awf.workflow.serializer import load_workflow
from benchmarks.multiple_choice import MultipleChoiceReward


def _load_test_queries(results_path: Path, data_path: Path) -> list[tuple[str, dict]]:
    """Load the held-out test queries from the results manifest."""
    with open(results_path) as f:
        results = json.load(f)
    test_indices = set(results["manifest"]["split"]["indices"]["test"])

    with open(data_path, encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]

    return [all_rows[i] for i in sorted(test_indices)]


def _build_gpqa_prompt(row: dict) -> str:
    """Build the GPQA multiple-choice prompt from a data row."""
    question = row["Question"]
    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    import random
    # Seed directly from the question text so prompts remain reproducible
    # across processes.
    rng = random.Random(question)
    rng.shuffle(choices)
    correct = row["Correct Answer"]
    answer_idx = choices.index(correct)
    letter = chr(ord("A") + answer_idx)

    letters = ["A", "B", "C", "D"]
    choice_lines = "\n".join(
        f"{letters[i]}. {choice}" for i, choice in enumerate(choices)
    )
    return (
        f"{question}\n\n{choice_lines}\n\n"
        "After reasoning, output exactly one line:\n"
        f"The final answer is: {letter}\n"
    ), letter


# ---------------------------------------------------------------
# AFlow prompt (from AFlow workspace GPQA round_2/prompt.py)
# ---------------------------------------------------------------
AFLOW_PROMPT = """
Solve the following graduate-level multiple-choice question carefully.
You must reason step by step and then choose the correct answer among A, B, C, D.
After your reasoning, re-verify your chosen option: double-check that it matches the problem constraints and that the other options are indeed incorrect. Avoid making subtle misinterpretations.
Finally, output exactly one terminal line in the following format (no extra text after it):

The final answer is: X

Replace X with A, B, C, or D.
"""


async def _evaluate_vanilla(
    queries: list[dict],
    llm: AsyncLLMClient,
) -> dict[str, Any]:
    """Evaluate the original (pre-optimization) AWF workflow."""
    workflow = load_workflow(
        PROJECT_ROOT / "experiments/workflows/multiple_choice/default_workflow.yaml"
    )
    reward = MultipleChoiceReward()
    scheduler = GraphScheduler(llm.config)
    executor = RuntimeExecutor(ExecutorConfig(max_steps=8, trace_enabled=True))

    correct = 0
    total = 0
    total_tokens = 0

    for row in queries:
        query_text, answer_letter = _build_gpqa_prompt(row)
        output, context, recorder = await executor.execute(
            workflow, scheduler, query_text, llm
        )
        trace = recorder.trace
        ground_truth = {"dataset": "gpqa", "answer": answer_letter}
        hard = reward.hard_reward(query_text, ground_truth, output, trace)
        correct += int(hard)
        total += 1
        total_tokens += (
            trace.total_prompt_tokens + trace.total_completion_tokens
        )

    return {
        "method": "Vanilla (AWF baseline)",
        "num_questions": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "total_tokens": total_tokens,
        "avg_tokens_per_query": total_tokens / total if total else 0,
    }


async def _evaluate_aflow(
    queries: list[dict],
    llm: AsyncLLMClient,
    ensemble_size: int = 3,
) -> dict[str, Any]:
    """Evaluate using AFlow's prompt + self-consistency ensemble.

    AFlow generates N independent solutions then uses a final LLM call to
    ensemble them and pick the most reliable answer.
    """
    reward = MultipleChoiceReward()
    scheduler = GraphScheduler(llm.config)
    executor = RuntimeExecutor(ExecutorConfig(max_steps=8, trace_enabled=True))
    workflow = load_workflow(
        PROJECT_ROOT / "experiments/workflows/multiple_choice/default_workflow.yaml"
    )

    correct = 0
    total = 0
    total_tokens = 0

    for row in queries:
        question, answer_letter = _build_gpqa_prompt(row)
        # Replace default instruction suffix with AFlow prompt
        # The default prompt ends with "After reasoning..."; swap for AFlow's version
        query_prefix = question.rsplit("\n\nAfter reasoning", 1)[0]
        aflow_query = query_prefix + AFLOW_PROMPT

        # Phase 1: Independent solutions with AFlow prompt
        solutions = []
        for _ in range(ensemble_size):
            output, _, recorder = await executor.execute(
                workflow, scheduler, aflow_query, llm,
            )
            solutions.append(output)
            total_tokens += (
                recorder.trace.total_prompt_tokens + recorder.trace.total_completion_tokens
            )

        # Phase 2: Ensemble vote
        ensemble_query = (
            "We asked multiple independent experts the same multiple-choice question. "
            "Their answers are below. Identify the most reliable answer (A, B, C, or D) "
            "after comparing their reasoning. Output exactly:\nThe final answer is: X\n\n"
            "Experts' answers:\n" +
            "\n---\n".join(f"Expert {i+1}:\n{s}" for i, s in enumerate(solutions))
        )
        final_output, _, ensemble_recorder = await executor.execute(
            workflow, scheduler, ensemble_query, llm,
        )
        total_tokens += (
            ensemble_recorder.trace.total_prompt_tokens
            + ensemble_recorder.trace.total_completion_tokens
        )

        ground_truth = {"dataset": "gpqa", "answer": answer_letter}
        hard = reward.hard_reward(aflow_query, ground_truth, final_output, ensemble_recorder.trace)
        correct += int(hard)
        total += 1

    return {
        "method": "AFlow (self-consistency ensemble)",
        "ensemble_size": ensemble_size,
        "num_questions": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "total_tokens": total_tokens,
        "avg_tokens_per_query": total_tokens / total if total else 0,
        "llm_calls_per_query": ensemble_size + 1,
    }


async def main() -> None:
    results_path = Path(
        "experiments/results/deepseek_v4_flash/gpqa_deepseek_v4_flash_pilot40/results.json"
    )
    data_path = Path("/home/rongxing/Benchmark/QA/GPQA/gpqa_pilot40.jsonl")
    config_path = Path("experiments/configs/gpqa_deepseek_pilot40.yaml")
    config = load_config(config_path)

    queries = _load_test_queries(results_path, data_path)
    print(f"Loaded {len(queries)} test queries from {data_path}")

    llm = AsyncLLMClient(config.scheduler.llm)

    # -----------------------------------------------------------
    # Read AWF optimized test results (must be run first via run_test.py)
    # -----------------------------------------------------------
    with open(results_path) as f:
        results_data = json.load(f)

    optimized_metrics = results_data.get("official_test_metrics")
    if not optimized_metrics:
        print()
        print("WARNING: Held-out test not yet evaluated for AWF optimized workflow.")
        print("Run:  DEEPSEEK_API_KEY=... python3 -m experiments.scripts.run_test ...")
        print("Continuing with vanilla + AFlow only...\n")
    else:
        print(f"AWF Optimized test metrics: {json.dumps(optimized_metrics, indent=2)}")

    # -----------------------------------------------------------
    # Evaluate vanilla baseline
    # -----------------------------------------------------------
    print("\n=== Evaluating Vanilla Baseline ===")
    vanilla = await _evaluate_vanilla(queries, llm)
    print(json.dumps(vanilla, indent=2))

    # -----------------------------------------------------------
    # Evaluate AFlow
    # -----------------------------------------------------------
    print("\n=== Evaluating AFlow (self-consistency ensemble) ===")
    aflow = await _evaluate_aflow(queries, llm)
    print(json.dumps(aflow, indent=2))

    # -----------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------
    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<40} {'Acc':>8} {'Tokens':>10} {'Calls/Q':>10}")
    print("-" * 70)
    print(f"{'Vanilla (AWF baseline)':<40} {vanilla['accuracy']:>7.1%} {vanilla['total_tokens']:>10} {'1':>10}")
    if optimized_metrics:
        print(f"{'AWF Optimized (selective gate)':<40} {optimized_metrics['hard_success_rate']:>7.1%} {optimized_metrics['total_tokens']:>10} {'~1':>10}")
    print(f"{'AFlow (SC ensemble)':<40} {aflow['accuracy']:>7.1%} {aflow['total_tokens']:>10} {aflow['llm_calls_per_query']:>10}")
    print("-" * 70)

    # Write comparison artifact
    comparison = {
        "test_split_indices": sorted(
            results_data["manifest"]["split"]["indices"]["test"]
        ),
        "vanilla": vanilla,
        "awf_optimized": optimized_metrics,
        "aflow": aflow,
    }
    output_path = Path(
        "experiments/results/deepseek_v4_flash/gpqa_deepseek_v4_flash_pilot40/comparison.json"
    )
    output_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nComparison written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
