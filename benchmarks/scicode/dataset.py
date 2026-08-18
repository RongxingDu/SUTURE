"""Leakage-resistant SciCode JSONL loader.

SciCode's official evaluator calls a model sequentially for every subproblem.
The generic AWF runner evaluates one query at a time, so this module exposes
two explicit protocols:

``first_subproblem``
    Evaluate only the first subproblem of each main problem.  This is the
    protocol used by the small mechanism study because it is exactly the
    first call of the official sequential protocol and needs no fabricated
    previous model output.

``independent_subproblems``
    Expose every non-excluded subproblem independently.  This is useful for
    unit and component studies, but it is deliberately labelled non-official
    because later subproblems do not receive previous model-generated code.

Hidden assertions stay in the in-memory ``private_test_specs`` mapping.  The
ground truth placed in traces contains only identifiers and deterministic
non-cryptographic fingerprints.
"""

from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path
from typing import Any, Literal



SciCodeProtocol = Literal[
    "first_subproblem",
    "independent_subproblems",
]

# These are the three steps excluded by the official 338-subproblem protocol.
# The upstream JSON currently contains 341 steps; the official evaluator
# supplies fixed carry-over code for these entries and does not score them.
OFFICIAL_EXCLUDED_STEPS = frozenset({"13.6", "62.1", "76.3"})

_DEFAULT_TEMPLATE = """PROBLEM DESCRIPTION:
You are solving one scientific-programming subproblem from SciCode.

{previous_section}

NEXT SUBPROBLEM:
{step_description}

REQUIRED FUNCTION HEADER:
{function_header}

EXPECTED RETURN:
{return_line}

DEPENDENCIES:
Use only the following dependencies. Do not repeat imports in your answer.
{dependencies}

RESPONSE CONTRACT:
Return the complete implementation for this subproblem in exactly one
```python``` block. Do not include tests, example usage, or prose outside the
code block.
"""


def format_scicode_prompt(
    problem: dict[str, Any],
    step_index: int,
) -> str:
    """Construct a public prompt without hidden tests or reference code."""
    sub_steps = problem.get("sub_steps")
    if not isinstance(sub_steps, list) or not 0 <= step_index < len(sub_steps):
        raise ValueError("step_index is outside the problem's sub_steps")
    step = sub_steps[step_index]
    if not isinstance(step, dict):
        raise ValueError("sub_steps entries must be objects")

    previous: list[str] = []
    for prior in sub_steps[:step_index]:
        if not isinstance(prior, dict):
            raise ValueError("sub_steps entries must be objects")
        previous.append(
            "\n".join(
                (
                    _require_text(
                        prior.get("step_description_prompt"),
                        "step_description_prompt",
                    ),
                    _require_text(
                        prior.get("function_header"),
                        "function_header",
                    ),
                )
            )
        )
    previous_section = (
        "PREVIOUS SUBPROBLEM DESCRIPTIONS AND FUNCTION HEADERS:\n"
        + "\n\n---\n\n".join(previous)
        if previous
        else "PREVIOUS SUBPROBLEMS:\nNone. This is the first subproblem."
    )
    return _DEFAULT_TEMPLATE.format(
        previous_section=previous_section,
        step_description=_require_text(
            step.get("step_description_prompt"),
            "step_description_prompt",
        ),
        function_header=_require_text(
            step.get("function_header"),
            "function_header",
        ),
        return_line=(
            str(step.get("return_line") or "").strip()
            or "Use the return behavior specified by the function header."
        ),
        dependencies=(
            str(problem.get("required_dependencies") or "Python standard library")
        ),
    )


class SciCodeDataset:
    """Load SciCode rows while retaining hidden tests outside trace targets."""

    def __init__(
        self,
        data_path: str | Path | None = None,
        *,
        protocol: SciCodeProtocol = "first_subproblem",
        source_split: str | None = None,
    ) -> None:
        if protocol not in {
            "first_subproblem",
            "independent_subproblems",
        }:
            raise ValueError(f"Unsupported SciCode protocol: {protocol}")
        self.data_path = Path(data_path) if data_path is not None else None
        self.protocol = protocol
        self.source_split = source_split
        self._problems: list[dict[str, Any]] = []
        self.private_test_specs: dict[str, dict[str, Any]] = {}

    def load(
        self,
        path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        filepath = Path(path) if path is not None else self.data_path
        if filepath is None:
            raise ValueError("No SciCode data path provided")

        rows: list[dict[str, Any]] = []
        with filepath.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {filepath} at line {line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Expected an object in {filepath} at line {line_number}"
                    )
                _validate_problem(row)
                rows.append(row)
        self._problems = rows
        self.private_test_specs.clear()
        return rows

    def to_pairs(self) -> list[tuple[str, dict[str, Any]]]:
        """Return public query/target pairs and retain tests privately."""
        pairs: list[tuple[str, dict[str, Any]]] = []
        self.private_test_specs.clear()
        for source_index, problem in enumerate(self._problems):
            sub_steps = problem["sub_steps"]
            indices = (
                range(1)
                if self.protocol == "first_subproblem"
                else range(len(sub_steps))
            )
            for step_index in indices:
                step = sub_steps[step_index]
                step_id = str(step["step_number"]).strip()
                if step_id in OFFICIAL_EXCLUDED_STEPS:
                    continue
                task_id = f"SciCode/{step_id}"
                tests = _validate_tests(step.get("test_cases"))
                dependencies = str(
                    problem.get("required_dependencies") or ""
                ).strip()
                function_header = _require_text(
                    step.get("function_header"),
                    "function_header",
                )
                entry_point = _extract_entry_point(function_header)
                test_spec = {
                    "step_id": step_id,
                    "tests": tests,
                    "dependencies": dependencies,
                    "entry_point": entry_point,
                }
                test_spec_hash = _sha256_json(test_spec)
                if task_id in self.private_test_specs:
                    raise ValueError(f"Duplicate SciCode task id: {task_id}")
                self.private_test_specs[task_id] = test_spec

                source_split = (
                    problem.get("_source_split")
                    or problem.get("source_split")
                    or self.source_split
                )
                target: dict[str, Any] = {
                    "dataset": "scicode",
                    "task_id": task_id,
                    "problem_id": str(problem["problem_id"]),
                    "step_id": step_id,
                    "step_index": step_index,
                    "entry_point": entry_point,
                    "protocol": self.protocol,
                    "test_spec_sha256": test_spec_hash,
                    "hdf5_required": True,
                    "source_index": source_index,
                }
                if source_split is not None:
                    target["source_split"] = str(source_split)
                pairs.append(
                    (
                        format_scicode_prompt(problem, step_index),
                        target,
                    )
                )
        return pairs


def load_scicode(
    path: str | Path,
    *,
    protocol: SciCodeProtocol = "first_subproblem",
    source_split: str | None = None,
) -> tuple[
    list[tuple[str, dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    """Load pairs together with their private test-spec registry."""
    dataset = SciCodeDataset(
        path,
        protocol=protocol,
        source_split=source_split,
    )
    dataset.load()
    pairs = dataset.to_pairs()
    return pairs, dict(dataset.private_test_specs)


def _validate_problem(problem: dict[str, Any]) -> None:
    _require_text(str(problem.get("problem_id", "")), "problem_id")
    sub_steps = problem.get("sub_steps")
    if not isinstance(sub_steps, list) or not sub_steps:
        raise ValueError("SciCode problem must contain non-empty sub_steps")
    for step in sub_steps:
        if not isinstance(step, dict):
            raise ValueError("sub_steps entries must be objects")
        _require_text(step.get("step_number"), "step_number")
        _require_text(
            step.get("step_description_prompt"),
            "step_description_prompt",
        )
        _require_text(step.get("function_header"), "function_header")
        if step.get("return_line") is not None and not isinstance(
            step.get("return_line"),
            str,
        ):
            raise ValueError("return_line must be text when provided")
        step_id = str(step.get("step_number") or "").strip()
        if step_id not in OFFICIAL_EXCLUDED_STEPS:
            _validate_tests(step.get("test_cases"))


def _validate_tests(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("SciCode test_cases must be a non-empty list")
    return [
        _require_text(test, f"test_cases[{index}]")
        for index, test in enumerate(value)
    ]


def _extract_entry_point(function_header: str) -> str:
    try:
        # A few published headers contain docstrings with legacy invalid
        # escapes such as ``\o``.  They are valid source today but Python 3.14
        # emits a SyntaxWarning while parsing them; suppress only that warning
        # for this trusted dataset field.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(function_header)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Invalid SciCode function_header") from exc
    for node in tree.body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            return node.name
    raise ValueError("SciCode function_header has no top-level callable")


def _sha256_json(value: Any) -> str:
    """Simple JSON identity for test-spec comparison."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value.strip()
