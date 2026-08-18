"""Best-effort policy for LLM-generated AFlow Python.

This is an integrity guard for the research pilot, not a security sandbox.
It rejects direct filesystem, environment, process, network, dynamic-code,
and dunder-introspection surfaces before a generated module is imported.
"""

from __future__ import annotations

import ast
from pathlib import Path


POLICY_VERSION = "aflow-generated-source-v1"

_ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "asyncio",
        "collections",
        "functools",
        "itertools",
        "math",
        "random",
        "scripts",
        "statistics",
        "typing",
        "workspace",
    }
)
_FORBIDDEN_CALLS = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "globals",
        "input",
        "locals",
        "open",
        "vars",
    }
)
_FORBIDDEN_TEXT = (
    "/home/",
    "benchmark/",
    "deepseek_api_key",
    "gpqa_test",
    "http://",
    "https://",
    "os.environ",
)


def validate_generated_aflow_directory(path: Path) -> None:
    """Reject obvious data/secret access before importing an AFlow artifact."""
    directory = path.resolve()
    if not directory.is_dir():
        raise ValueError(f"AFlow generated artifact is not a directory: {path}")
    sources = sorted(directory.glob("*.py"))
    required = {directory / "graph.py", directory / "prompt.py"}
    if not required.issubset(set(sources)):
        raise ValueError("AFlow artifact lacks graph.py or prompt.py")
    for source in sources:
        _validate_source(source)


def _validate_source(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    for marker in _FORBIDDEN_TEXT:
        if marker in lowered:
            raise ValueError(
                f"Generated AFlow source violates {POLICY_VERSION}: "
                f"{path.name} contains forbidden marker {marker!r}"
            )
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"Generated AFlow source is invalid: {path}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            forbidden = roots - _ALLOWED_IMPORT_ROOTS
            if forbidden:
                raise ValueError(
                    "Generated AFlow source imports forbidden modules: "
                    + ", ".join(sorted(forbidden))
                )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                raise ValueError(
                    "Generated AFlow source imports forbidden module: "
                    f"{node.module!r}"
                )
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in _FORBIDDEN_CALLS
            ):
                raise ValueError(
                    "Generated AFlow source calls forbidden builtin: "
                    f"{node.func.id}"
                )
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(
                "Generated AFlow source uses dunder introspection"
            )
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError(
                "Generated AFlow source uses dunder introspection"
            )
