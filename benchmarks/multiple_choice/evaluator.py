"""Strict answer parsing and scoring for multiple-choice tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_CHOICE_LABELS = ("A", "B", "C", "D")
_STRICT_FINAL_PATTERN = re.compile(
    r"(?:^|\n)[ \t]*The[ \t]+final[ \t]+answer[ \t]+is"
    r"[ \t]*:[ \t]*([A-D])[ \t]*[.]?[ \t]*\Z",
    re.IGNORECASE,
)
_ANY_FINAL_MARKER = re.compile(
    r"The\s+final\s+answer\s+is\s*:\s*([A-Za-z])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerParseResult:
    """A parsed option plus a non-sensitive diagnostic mode."""

    answer_letter: str | None
    answer_index: int | None
    parse_mode: str

    @property
    def valid(self) -> bool:
        return self.answer_letter is not None


class MultipleChoiceEvaluator:
    """Evaluate only an explicit, terminal ``The final answer is: X`` line."""

    @staticmethod
    def parse_answer(output: Any) -> AnswerParseResult:
        if output is None:
            return AnswerParseResult(None, None, "missing_output")
        text = str(output)
        if not text.strip():
            return AnswerParseResult(None, None, "missing_output")
        terminal_text = text.rstrip()

        strict_match = _STRICT_FINAL_PATTERN.search(terminal_text)
        if strict_match is not None:
            letter = strict_match.group(1).upper()
            return AnswerParseResult(
                letter,
                _CHOICE_LABELS.index(letter),
                "strict_final_marker",
            )

        markers = list(_ANY_FINAL_MARKER.finditer(text))
        if markers:
            last_letter = markers[-1].group(1).upper()
            if last_letter not in _CHOICE_LABELS:
                mode = "invalid_final_choice"
            else:
                mode = "nonterminal_final_marker"
            return AnswerParseResult(None, None, mode)

        if re.fullmatch(r"\s*[A-D]\s*[.]?\s*", text, re.IGNORECASE):
            return AnswerParseResult(None, None, "bare_choice_rejected")
        return AnswerParseResult(None, None, "missing_final_marker")

    @staticmethod
    def normalize_ground_truth(ground_truth: Any) -> str | None:
        value = ground_truth
        if isinstance(value, dict):
            if "answer_letter" in value:
                value = value["answer_letter"]
            elif "answer_index" in value:
                value = value["answer_index"]
            elif "answer" in value:
                value = value["answer"]
            else:
                return None

        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return _CHOICE_LABELS[value] if 0 <= value < 4 else None
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in _CHOICE_LABELS:
                return normalized
            if normalized in {"0", "1", "2", "3"}:
                return _CHOICE_LABELS[int(normalized)]
        return None

    def evaluate_detailed(
        self,
        output: Any,
        ground_truth: Any,
    ) -> tuple[bool, AnswerParseResult]:
        parsed = self.parse_answer(output)
        expected = self.normalize_ground_truth(ground_truth)
        return bool(
            parsed.valid
            and expected is not None
            and parsed.answer_letter == expected
        ), parsed

    def evaluate(self, output: Any, ground_truth: Any) -> bool:
        correct, _ = self.evaluate_detailed(output, ground_truth)
        return correct
