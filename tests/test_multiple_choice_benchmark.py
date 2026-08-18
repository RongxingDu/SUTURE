"""Tests for deterministic GPQA/MMLU benchmark semantics."""

from __future__ import annotations

import json

import pytest

from awf.trace.schema import ExecutionTrace, TraceStep
from benchmarks.multiple_choice.dataset import (
    GPQADataset,
    MMLUDataset,
    format_multiple_choice_prompt,
    load_gpqa,
    load_mmlu,
)
from benchmarks.multiple_choice.evaluator import MultipleChoiceEvaluator
from benchmarks.multiple_choice.reward import MultipleChoiceReward


def _trace(*, clean: bool = True) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id="multiple-choice",
        steps=[
            TraceStep(
                step_id="answer",
                step_index=0,
                node_id="answer",
                node_type="llm",
                action="execute",
                success=clean,
            ),
        ],
    )


def _write_jsonl(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_gpqa_loader_is_deterministic_and_drops_private_metadata(tmp_path):
    row = {
        "Record ID": "gpqa-1",
        "Question": "Which planet is closest to the Sun?",
        "Correct Answer": "Mercury",
        "Incorrect Answer 1": "Venus",
        "Incorrect Answer 2": "Earth",
        "Incorrect Answer 3": "Mars",
        "Explanation": "SECRET GOLD RATIONALE",
        "Question Writer": "private-author-id",
        "High-level domain": "Physics",
        "Subdomain": "Astronomy",
    }
    path = _write_jsonl(tmp_path, "gpqa.jsonl", [row])

    first = load_gpqa(path, source_split="train")
    second = load_gpqa(path, source_split="train")

    assert first == second
    prompt, target = first[0]
    assert "SECRET GOLD RATIONALE" not in prompt
    assert "private-author-id" not in prompt
    assert "Correct Answer" not in prompt
    assert prompt.count("A. ") == 1
    assert prompt.count("B. ") == 1
    assert prompt.count("C. ") == 1
    assert prompt.count("D. ") == 1
    assert set(target) == {
        "dataset",
        "answer_index",
        "answer_letter",
        "sample_id",
        "source_index",
        "formatter_version",
        "source_split",
        "domain",
        "subdomain",
    }
    assert target["sample_id"] == "gpqa-1"
    assert target["source_split"] == "train"
    assert "Explanation" not in target
    displayed = [
        line[3:]
        for line in prompt.splitlines()
        if len(line) > 3 and line[:3] in {"A. ", "B. ", "C. ", "D. "}
    ]
    assert displayed[target["answer_index"]] == "Mercury"


def test_mmlu_loader_preserves_choices_and_normalizes_label(tmp_path):
    row = {
        "type": "mmlu",
        "subject": "abstract_algebra",
        "question": "What is the identity element under addition?",
        "choices": ["0", "1", "-1", "None"],
        "answer": 0,
    }
    path = _write_jsonl(tmp_path, "mmlu.jsonl", [row])

    prompt, target = load_mmlu(path, source_split="test")[0]

    assert "A. 0\nB. 1\nC. -1\nD. None" in prompt
    assert target["answer_index"] == 0
    assert target["answer_letter"] == "A"
    assert target["subject"] == "abstract_algebra"
    assert target["source_split"] == "test"


def test_dataset_instances_support_deferred_path(tmp_path):
    gpqa_path = _write_jsonl(
        tmp_path,
        "gpqa.jsonl",
        [{
            "Question": "Q?",
            "Correct Answer": "one",
            "Incorrect Answer 1": "two",
            "Incorrect Answer 2": "three",
            "Incorrect Answer 3": "four",
        }],
    )
    mmlu_path = _write_jsonl(
        tmp_path,
        "mmlu.jsonl",
        [{
            "question": "Q?",
            "choices": ["one", "two", "three", "four"],
            "answer": "B",
        }],
    )

    gpqa = GPQADataset()
    gpqa.load(gpqa_path)
    mmlu = MMLUDataset()
    mmlu.load(mmlu_path)

    assert len(gpqa.to_pairs()) == 1
    assert mmlu.to_pairs()[0][1]["answer_index"] == 1


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {
                "question": "Q?",
                "choices": ["a", "b", "c"],
                "answer": 0,
            },
            "exactly four",
        ),
        (
            {
                "question": "Q?",
                "choices": ["a", "a", "c", "d"],
                "answer": 0,
            },
            "distinct",
        ),
        (
            {
                "question": "Q?",
                "choices": ["a", "b", "c", "d"],
                "answer": 4,
            },
            "0 to 3",
        ),
    ],
)
def test_mmlu_loader_rejects_ambiguous_rows(tmp_path, row, message):
    path = _write_jsonl(tmp_path, "invalid.jsonl", [row])
    dataset = MMLUDataset(path)
    dataset.load()

    with pytest.raises(ValueError, match=message):
        dataset.to_pairs()


def test_prompt_builder_has_no_answer_parameter_or_label_claim():
    prompt = format_multiple_choice_prompt(
        "Pick one.",
        ["red", "green", "blue", "yellow"],
    )

    assert "correct" not in prompt.casefold()
    assert "The final answer is: X" in prompt


@pytest.mark.parametrize(
    ("output", "letter", "mode"),
    [
        (
            "Reasoning may mention A or B.\nThe final answer is: C",
            "C",
            "strict_final_marker",
        ),
        ("the final answer is: d.\n", "D", "strict_final_marker"),
        ("A", None, "bare_choice_rejected"),
        (
            "The final answer is: A\nAdditional prose",
            None,
            "nonterminal_final_marker",
        ),
        (
            "Option B seems strongest.",
            None,
            "missing_final_marker",
        ),
        (
            "The final answer is: E",
            None,
            "invalid_final_choice",
        ),
        ("", None, "missing_output"),
    ],
)
def test_strict_parser_reports_parse_mode(output, letter, mode):
    parsed = MultipleChoiceEvaluator.parse_answer(output)

    assert parsed.answer_letter == letter
    assert parsed.parse_mode == mode


def test_hard_reward_requires_strict_terminal_marker():
    reward = MultipleChoiceReward()
    target = {"answer_index": 1}

    assert reward.hard_reward(
        "query",
        target,
        "Reasoning.\nThe final answer is: B",
        _trace(),
    ) == 1.0
    assert reward.hard_reward(
        "query",
        target,
        "B",
        _trace(),
    ) == 0.0
    assert reward.hard_reward(
        "query",
        target,
        "The final answer is: B\nActually A",
        _trace(),
    ) == 0.0


def test_process_reward_is_correctness_grounded_and_non_leaking():
    reward = MultipleChoiceReward()
    target = {
        "answer_letter": "B",
        "sample_id": "secret-sample-id",
    }
    wrong_trace = _trace()

    wrong_score = reward.process_reward(
        "query",
        target,
        "Plausible explanation.\nThe final answer is: A",
        wrong_trace,
    )

    assert wrong_score == 0.20
    diagnostic = wrong_trace.metadata[
        "reward_diagnostics"
    ]["multiple_choice"]
    assert diagnostic["parse_mode"] == "strict_final_marker"
    assert diagnostic["final_answer_correct"] is False
    assert diagnostic["execution_clean"] is True
    assert "answer_letter" not in diagnostic
    assert "answer_index" not in diagnostic
    assert "expected" not in diagnostic
    assert "predicted" not in diagnostic
    assert "secret-sample-id" not in json.dumps(diagnostic)

    correct_trace = _trace()
    assert reward.process_reward(
        "query",
        target,
        "Reasoning.\nThe final answer is: B",
        correct_trace,
    ) == 1.0


def test_failed_execution_loses_clean_process_component():
    reward = MultipleChoiceReward()
    score = reward.process_reward(
        "query",
        {"answer_letter": "D"},
        "The final answer is: D",
        _trace(clean=False),
    )

    assert score == 0.9
