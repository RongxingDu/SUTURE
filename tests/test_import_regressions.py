"""Public modules must be importable in a clean Python process."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "awf.scheduler",
        "awf.scheduler.base",
        "awf.executor",
        "awf.optimizer",
        "awf.protocol",
    ],
)
def test_public_module_imports_do_not_depend_on_import_order(module):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
