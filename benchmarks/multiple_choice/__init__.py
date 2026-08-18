"""Shared multiple-choice benchmarks for GPQA and MMLU."""

from benchmarks.multiple_choice.dataset import (
    GPQADataset,
    MMLUDataset,
    format_multiple_choice_prompt,
    load_gpqa,
    load_mmlu,
)
from benchmarks.multiple_choice.evaluator import (
    AnswerParseResult,
    MultipleChoiceEvaluator,
)
from benchmarks.multiple_choice.reward import MultipleChoiceReward

__all__ = [
    "AnswerParseResult",
    "GPQADataset",
    "MMLUDataset",
    "MultipleChoiceEvaluator",
    "MultipleChoiceReward",
    "format_multiple_choice_prompt",
    "load_gpqa",
    "load_mmlu",
]
