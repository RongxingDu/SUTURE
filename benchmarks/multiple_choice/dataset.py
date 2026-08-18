"""Deterministic GPQA and MMLU JSONL loaders."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any


_CHOICE_LABELS = ("A", "B", "C", "D")
_GPQA_PERMUTATIONS = tuple(itertools.permutations(range(4)))
_PROMPT_SUFFIX = (
    "Reason about the question, then end your response with exactly one line "
    "in this format:\nThe final answer is: X\n"
    "Replace X with A, B, C, or D."
)


def format_multiple_choice_prompt(
    question: str,
    choices: list[str] | tuple[str, ...],
) -> str:
    """Build the one canonical prompt shared by every evaluated method.

    The function accepts only the public question and displayed choices.  It
    deliberately has no answer argument, which prevents accidental label or
    rationale interpolation into the model-facing prompt.
    """
    normalized_question = _require_text(question, "question")
    normalized_choices = _validate_choices(choices)
    rendered_choices = "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(_CHOICE_LABELS, normalized_choices)
    )
    return f"{normalized_question}\n\n{rendered_choices}\n\n{_PROMPT_SUFFIX}"


class GPQADataset:
    """Load the ASpec-style GPQA main-set split.

    Required source fields are ``Question``, ``Correct Answer`` and
    ``Incorrect Answer 1`` through ``Incorrect Answer 3``.  Choice placement
    uses a compact deterministic question identifier to select one of all 24
    permutations, avoiding process-dependent identifiers.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        *,
        source_split: str | None = None,
    ) -> None:
        self.data_path = Path(data_path) if data_path is not None else None
        self.source_split = source_split
        self._problems: list[dict[str, Any]] = []

    def load(
        self,
        path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        self._problems = _load_jsonl(path, self.data_path)
        return self._problems

    def to_pairs(self) -> list[tuple[str, dict[str, Any]]]:
        pairs: list[tuple[str, dict[str, Any]]] = []
        for source_index, row in enumerate(self._problems):
            question = _require_text(row.get("Question"), "Question")
            source_choices = _validate_choices([
                row.get("Correct Answer"),
                row.get("Incorrect Answer 1"),
                row.get("Incorrect Answer 2"),
                row.get("Incorrect Answer 3"),
            ])
            permutation = _gpqa_permutation(question)
            displayed_choices = [
                source_choices[index]
                for index in permutation
            ]
            answer_index = permutation.index(0)
            sample_id = _sample_id(
                "gpqa",
                row.get("Record ID"),
                question,
            )
            ground_truth = _ground_truth(
                dataset="gpqa",
                answer_index=answer_index,
                sample_id=sample_id,
                source_index=source_index,
                source_split=(
                    row.get("source_split")
                    or row.get("_source_split")
                    or self.source_split
                ),
                metadata={
                    "domain": row.get("High-level domain"),
                    "subdomain": row.get("Subdomain"),
                },
            )
            pairs.append(
                (
                    format_multiple_choice_prompt(
                        question,
                        displayed_choices,
                    ),
                    ground_truth,
                )
            )
        return pairs


class MMLUDataset:
    """Load the compact ASpec-style MMLU subset."""

    def __init__(
        self,
        data_path: str | Path | None = None,
        *,
        source_split: str | None = None,
    ) -> None:
        self.data_path = Path(data_path) if data_path is not None else None
        self.source_split = source_split
        self._problems: list[dict[str, Any]] = []

    def load(
        self,
        path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        self._problems = _load_jsonl(path, self.data_path)
        return self._problems

    def to_pairs(self) -> list[tuple[str, dict[str, Any]]]:
        pairs: list[tuple[str, dict[str, Any]]] = []
        for source_index, row in enumerate(self._problems):
            question = _require_text(row.get("question"), "question")
            choices = _validate_choices(row.get("choices"))
            answer_index = _normalize_answer_index(row.get("answer"))
            sample_id = _sample_id(
                "mmlu",
                row.get("sample_id"),
                f"{row.get('subject', '')}\n{question}",
            )
            ground_truth = _ground_truth(
                dataset="mmlu",
                answer_index=answer_index,
                sample_id=sample_id,
                source_index=source_index,
                source_split=(
                    row.get("source_split")
                    or row.get("_source_split")
                    or self.source_split
                ),
                metadata={"subject": row.get("subject")},
            )
            pairs.append(
                (
                    format_multiple_choice_prompt(question, choices),
                    ground_truth,
                )
            )
        return pairs


def load_gpqa(
    path: str | Path,
    *,
    source_split: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    dataset = GPQADataset(path, source_split=source_split)
    dataset.load()
    return dataset.to_pairs()


def load_mmlu(
    path: str | Path,
    *,
    source_split: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    dataset = MMLUDataset(path, source_split=source_split)
    dataset.load()
    return dataset.to_pairs()


def _load_jsonl(
    path: str | Path | None,
    default_path: Path | None,
) -> list[dict[str, Any]]:
    filepath = Path(path) if path is not None else default_path
    if filepath is None:
        raise ValueError("No data path provided")

    rows: list[dict[str, Any]] = []
    with filepath.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {filepath} at line {line_number}",
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected an object in {filepath} at line {line_number}",
                )
            rows.append(row)
    return rows


def _gpqa_permutation(question: str) -> tuple[int, ...]:
    h = 0
    for c in question:
        h = (h * 31 + ord(c)) & 0xFFFFFFFFFFFFFFFF
    return _GPQA_PERMUTATIONS[h % len(_GPQA_PERMUTATIONS)]


def _ground_truth(
    *,
    dataset: str,
    answer_index: int,
    sample_id: str,
    source_index: int,
    source_split: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    # Do not retain source choices, answer text, explanations, or GPQA
    # validator metadata in the evaluator-facing target.
    target: dict[str, Any] = {
        "dataset": dataset,
        "answer_index": answer_index,
        "answer_letter": _CHOICE_LABELS[answer_index],
        "sample_id": sample_id,
        "source_index": source_index,
        "formatter_version": "multiple_choice_v1",
    }
    if source_split is not None:
        target["source_split"] = str(source_split)
    target.update(
        {
            key: value
            for key, value in metadata.items()
            if value is not None
        }
    )
    return target


def _validate_choices(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("choices must contain exactly four items")
    choices = [
        _require_text(choice, f"choice {index}")
        for index, choice in enumerate(value)
    ]
    if len(set(choices)) != 4:
        raise ValueError("choices must be distinct")
    return choices


def _normalize_answer_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("answer must be an index from 0 to 3")
    if isinstance(value, int) and 0 <= value < 4:
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in _CHOICE_LABELS:
            return _CHOICE_LABELS.index(normalized)
        if normalized in {"0", "1", "2", "3"}:
            return int(normalized)
    raise ValueError("answer must be an index from 0 to 3 or A-D")


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _sample_id(dataset: str, explicit: Any, identity: str) -> str:
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    h = 0
    for c in identity:
        h = (h * 31 + ord(c)) & 0xFFFFFFFFFFFFFFFF
    return f"{dataset}:{h:016x}"
