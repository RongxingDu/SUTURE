"""Code extraction and configurable test execution for code-generation tasks.

Generated code is untrusted.  The default evaluator therefore refuses to
execute it.  Production callers should provide an external ``CodeRunner``
backed by a real sandbox.  On Linux, ``BubblewrapCodeRunner`` provides a
minimal, networkless mount/user/PID namespace. ``LocalSubprocessRunner`` is
available only as an explicit opt-in for controlled tests; its resource limits
reduce accidents but do not provide filesystem or network isolation.
"""

from __future__ import annotations

import math
import os
import platform
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

try:  # ``resource`` is Unix-only.
    import resource
except ImportError:  # pragma: no cover - exercised on non-Unix platforms
    resource = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RunnerResult:
    """Raw result returned by a code runner."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error_message: Optional[str] = None


class CodeRunner(Protocol):
    """Execution boundary for an external sandbox service or local runner."""

    def run(self, script: str, timeout_seconds: float) -> RunnerResult:
        """Execute ``script`` and return a bounded result."""
        ...


@dataclass(frozen=True)
class CodeEvaluationResult:
    """Detailed result supporting safe binary/partial-test diagnostics.

    ``outcome_codes`` contains only evaluator-defined categories. It never
    contains hidden test source, assertion payloads, or traceback lines.
    """

    passed_tests: int
    total_tests: int
    tests_executed: int
    messages: tuple[str, ...] = ()
    outcome_codes: tuple[str, ...] = ()

    @property
    def pass_rate(self) -> float:
        if self.total_tests <= 0:
            return 0.0
        return self.passed_tests / self.total_tests

    @property
    def all_passed(self) -> bool:
        return (
            self.total_tests > 0
            and self.tests_executed == self.total_tests
            and self.passed_tests == self.total_tests
        )

    @property
    def execution_rate(self) -> float:
        if self.total_tests <= 0:
            return 0.0
        return min(self.tests_executed, self.total_tests) / self.total_tests

    @property
    def outcome_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(self.outcome_codes).items()))

    def summary(self) -> str:
        prefix = (
            f"{self.passed_tests}/{self.total_tests} tests passed "
            f"({self.tests_executed} executed)"
        )
        outcomes = ", ".join(
            f"{name}={count}"
            for name, count in self.outcome_counts.items()
        )
        details = "; ".join(message for message in self.messages if message)
        suffix = "; ".join(
            item
            for item in (
                f"outcomes: {outcomes}" if outcomes else "",
                details,
            )
            if item
        )
        return f"{prefix}: {suffix}" if suffix else prefix


class LocalSubprocessRunner:
    """Explicitly opt-in, resource-limited local execution.

    This is *not* a security sandbox: code can still access files and the
    network with the current user's permissions.  It is intended for trusted
    regression fixtures only.  Use an external container/VM sandbox runner for
    model-generated code.
    """

    def __init__(
        self,
        python_executable: str = "python3",
        memory_limit_mb: int = 256,
        file_size_limit_mb: int = 1,
        max_output_chars: int = 20_000,
    ):
        self.python = python_executable
        self.memory_limit_mb = memory_limit_mb
        self.file_size_limit_mb = file_size_limit_mb
        self.max_output_chars = max_output_chars

    def run(self, script: str, timeout_seconds: float) -> RunnerResult:
        env = {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
        }
        limiter = self._make_resource_limiter(timeout_seconds)

        try:
            with tempfile.TemporaryDirectory(prefix="awf-code-eval-") as workdir:
                process = subprocess.Popen(
                    [self.python, "-I", "-S", "-c", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=workdir,
                    env=env,
                    start_new_session=True,
                    preexec_fn=limiter,
                )
                try:
                    stdout, stderr = process.communicate(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    self._terminate_process_group(process)
                    stdout, stderr = process.communicate()
                    return RunnerResult(
                        returncode=process.returncode or -1,
                        stdout=self._bounded(stdout),
                        stderr=self._bounded(stderr),
                        timed_out=True,
                        error_message=f"Execution timed out after {timeout_seconds}s",
                    )

            return RunnerResult(
                returncode=process.returncode,
                stdout=self._bounded(stdout),
                stderr=self._bounded(stderr),
            )
        except Exception as exc:
            return RunnerResult(
                returncode=-1,
                error_message=f"Local runner failed: {exc}",
            )

    def _make_resource_limiter(self, timeout_seconds: float):
        if resource is None:
            return None

        memory_bytes = max(self.memory_limit_mb, 1) * 1024 * 1024
        file_bytes = max(self.file_size_limit_mb, 1) * 1024 * 1024
        cpu_seconds = max(1, int(math.ceil(timeout_seconds)))

        def _limit_resources() -> None:
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (cpu_seconds, cpu_seconds + 1),
            )
            resource.setrlimit(
                resource.RLIMIT_AS,
                (memory_bytes, memory_bytes),
            )
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (file_bytes, file_bytes),
            )
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
            if hasattr(resource, "RLIMIT_CORE"):
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        return _limit_resources

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()

    def _bounded(self, value: str) -> str:
        return (value or "")[: self.max_output_chars]


class BubblewrapCodeRunner(LocalSubprocessRunner):
    """Run generated Python inside a minimal Linux bubblewrap sandbox.

    The sandbox has fresh user, PID, IPC, UTS, cgroup, and network namespaces;
    an empty environment; a private ``/tmp``; and only the host runtime trees
    ``/usr``, ``/lib``, and ``/lib64`` mounted read-only when present.  It does
    not bind the project, home directory, current directory, or ``/etc``.

    Bubblewrap still depends on the host kernel permitting its namespace setup.
    Construction validates the binaries, while :meth:`run` reports namespace
    policy failures as an ordinary failed ``RunnerResult``.
    """

    def __init__(
        self,
        python_executable: str = "python3",
        bubblewrap_executable: str = "bwrap",
        memory_limit_mb: int = 256,
        file_size_limit_mb: int = 1,
        max_output_chars: int = 20_000,
    ):
        if platform.system() != "Linux":
            raise RuntimeError("BubblewrapCodeRunner is supported only on Linux")

        resolved_bwrap = shutil.which(bubblewrap_executable)
        if resolved_bwrap is None:
            raise FileNotFoundError(
                f"Bubblewrap executable not found: {bubblewrap_executable}"
            )
        resolved_python = shutil.which(python_executable)
        if resolved_python is None:
            raise FileNotFoundError(
                f"Python executable not found: {python_executable}"
            )

        python_path = Path(resolved_python).resolve()
        runtime_roots = [
            Path(root)
            for root in ("/usr", "/lib", "/lib64")
            if Path(root).exists()
        ]
        if not any(
            python_path == root or root in python_path.parents
            for root in runtime_roots
        ):
            raise ValueError(
                "Bubblewrap Python must reside under /usr, /lib, or /lib64 "
                f"so it is available through a read-only runtime bind: "
                f"{python_path}"
            )

        super().__init__(
            python_executable=str(python_path),
            memory_limit_mb=memory_limit_mb,
            file_size_limit_mb=file_size_limit_mb,
            max_output_chars=max_output_chars,
        )
        self.bubblewrap = str(Path(resolved_bwrap).resolve())
        self.runtime_roots = tuple(runtime_roots)

    @classmethod
    def is_available(cls) -> bool:
        """Return whether this host has Linux, bubblewrap, and a bindable Python."""
        if platform.system() != "Linux" or shutil.which("bwrap") is None:
            return False
        python = shutil.which("python3")
        if python is None:
            return False
        resolved = Path(python).resolve()
        return any(
            resolved == root or root in resolved.parents
            for root in map(Path, ("/usr", "/lib", "/lib64"))
            if root.exists()
        )

    def run(self, script: str, timeout_seconds: float) -> RunnerResult:
        limiter = self._make_resource_limiter(timeout_seconds)
        command = self._build_command(script)

        try:
            # Temporary files keep a malicious program from exhausting parent
            # memory by streaming unbounded output through PIPE.
            with (
                tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file,
                tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file,
            ):
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    cwd="/",
                    env={},
                    start_new_session=True,
                    preexec_fn=limiter,
                )
                try:
                    process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    self._terminate_process_group(process)
                    process.wait()
                    stdout = self._read_bounded(stdout_file)
                    stderr = self._read_bounded(stderr_file)
                    return RunnerResult(
                        returncode=(
                            process.returncode
                            if process.returncode is not None
                            else -1
                        ),
                        stdout=stdout,
                        stderr=stderr,
                        timed_out=True,
                        error_message=(
                            f"Execution timed out after {timeout_seconds}s"
                        ),
                    )

                stdout = self._read_bounded(stdout_file)
                stderr = self._read_bounded(stderr_file)
                return RunnerResult(
                    returncode=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
        except Exception as exc:
            return RunnerResult(
                returncode=-1,
                error_message=f"Bubblewrap runner failed: {exc}",
            )

    def _build_command(self, script: str) -> list[str]:
        """Build the namespace/mount policy without consulting user input."""
        command = [
            self.bubblewrap,
            "--unshare-all",
            # These explicit flags document and satisfy bubblewrap's policy
            # checks even though --unshare-all already requests both.
            "--unshare-user",
            "--unshare-net",
            "--disable-userns",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--cap-drop",
            "ALL",
            "--hostname",
            "awf-code-sandbox",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--size",
            str(max(self.memory_limit_mb, 1) * 1024 * 1024),
            "--tmpfs",
            "/tmp",
        ]
        for root in self.runtime_roots:
            command.extend(("--ro-bind", str(root), str(root)))
        command.extend(
            [
                "--setenv",
                "PATH",
                "/usr/local/bin:/usr/bin",
                "--setenv",
                "LANG",
                "C.UTF-8",
                "--setenv",
                "LC_ALL",
                "C.UTF-8",
                "--setenv",
                "PYTHONHASHSEED",
                "0",
                "--setenv",
                "PYTHONIOENCODING",
                "utf-8",
                "--chdir",
                "/tmp",
                "--",
                self.python,
                "-I",
                "-S",
                "-c",
                script,
            ]
        )
        return command

    def _read_bounded(self, handle) -> str:
        handle.flush()
        handle.seek(0)
        return handle.read(self.max_output_chars)


class CodeEvaluator:
    """Extract code and evaluate it through an explicit execution boundary.

    Args:
        timeout_seconds: Per-test wall-clock timeout.
        python_executable: Python used by the opt-in local runner.
        sandbox_runner: Preferred external runner implementing ``CodeRunner``.
        allow_local_execution: Explicitly enable the limited local runner.
        use_bubblewrap: Explicitly enable the Linux bubblewrap sandbox.

    By default neither runner is enabled, so untrusted code is not executed.
    """

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        python_executable: str = "python3",
        sandbox_runner: Optional[CodeRunner] = None,
        allow_local_execution: bool = False,
        use_bubblewrap: bool = False,
    ):
        if sum(
            (
                sandbox_runner is not None,
                allow_local_execution,
                use_bubblewrap,
            )
        ) > 1:
            raise ValueError(
                "Provide exactly one of sandbox_runner, allow_local_execution, "
                "or use_bubblewrap"
            )
        self.timeout = timeout_seconds
        if sandbox_runner is not None:
            self.runner: Optional[CodeRunner] = sandbox_runner
        elif use_bubblewrap:
            self.runner = BubblewrapCodeRunner(
                python_executable=python_executable,
            )
        elif allow_local_execution:
            self.runner = LocalSubprocessRunner(
                python_executable=python_executable,
            )
        else:
            self.runner = None

    def evaluate(
        self,
        code: str,
        test_code: str | Sequence[str],
        entry_point: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Compatibility API returning whether every test passed."""
        result = self.evaluate_detailed(code, test_code, entry_point)
        return result.all_passed, result.summary()

    def evaluate_detailed(
        self,
        code: str,
        test_code: str | Sequence[str],
        entry_point: Optional[str] = None,
    ) -> CodeEvaluationResult:
        """Run one HumanEval suite or individual MBPP tests.

        A string is treated as one suite.  A sequence is evaluated one item at
        a time, enabling partial pass-rate rewards.
        """
        tests = (
            [test_code]
            if isinstance(test_code, str)
            else list(test_code)
        )
        if not tests:
            return CodeEvaluationResult(
                0,
                0,
                0,
                ("No tests provided",),
            )
        if not code.strip():
            return CodeEvaluationResult(
                0,
                len(tests),
                0,
                ("No generated code provided",),
                tuple("not_run_no_code" for _ in tests),
            )
        try:
            compile(code, "<generated>", "exec")
        except (SyntaxError, TypeError, ValueError):
            return CodeEvaluationResult(
                0,
                len(tests),
                0,
                ("Generated code did not compile",),
                tuple("compile_error" for _ in tests),
            )
        if self.runner is None:
            return CodeEvaluationResult(
                0,
                len(tests),
                0,
                (
                    "Execution disabled for untrusted code; configure an "
                    "external sandbox_runner, explicitly enable bubblewrap, "
                    "or opt in to limited local execution",
                ),
                tuple("execution_disabled" for _ in tests),
            )

        passed = 0
        executed = 0
        messages: list[str] = []
        outcome_codes: list[str] = []
        for index, test in enumerate(tests):
            if not isinstance(test, str) or not test.strip():
                messages.append(f"test {index + 1}: empty or invalid test")
                outcome_codes.append("invalid_test")
                continue
            try:
                script, completion_marker = self._build_script(
                    code,
                    test,
                    entry_point,
                )
            except ValueError as exc:
                del exc
                messages.append(
                    f"test {index + 1}: test harness could not be built"
                )
                outcome_codes.append("harness_build_error")
                continue

            run_result = self.runner.run(script, self.timeout)
            executed += 1
            completed = completion_marker in run_result.stdout
            if (
                run_result.returncode == 0
                and not run_result.timed_out
                and completed
            ):
                passed += 1
                outcome_code = "passed"
                detail = "passed"
            else:
                if (
                    run_result.returncode == 0
                    and not run_result.timed_out
                    and not completed
                ):
                    outcome_code = "harness_incomplete"
                    detail = (
                        "test harness did not reach its completion marker "
                        "(premature exit or harness tampering)"
                    )
                else:
                    outcome_code, detail = (
                        self._safe_failure_diagnostic(run_result)
                    )
            outcome_codes.append(outcome_code)
            messages.append(f"test {index + 1}: {detail[:500]}")

        return CodeEvaluationResult(
            passed_tests=passed,
            total_tests=len(tests),
            tests_executed=executed,
            messages=tuple(messages),
            outcome_codes=tuple(outcome_codes),
        )

    @staticmethod
    def _safe_failure_diagnostic(
        result: RunnerResult,
    ) -> tuple[str, str]:
        """Classify a failed run without retaining hidden-test traceback text."""
        if result.timed_out:
            return "timeout", "execution timed out"

        combined = "\n".join(
            value
            for value in (result.stderr, result.stdout, result.error_message)
            if value
        )
        if "AssertionError" in combined:
            return "assertion_failure", "hidden assertion failed"
        if re.search(
            r"\b(?:SyntaxError|IndentationError|TabError)\b",
            combined,
        ):
            return "compile_error", "generated program or harness did not compile"

        exception_types = re.findall(
            r"(?m)^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))"
            r"(?:\s*:|$)",
            combined,
        )
        if exception_types:
            exception_type = exception_types[-1]
            return (
                "runtime_error",
                f"runtime error ({exception_type})",
            )
        if result.error_message:
            return "runner_error", "execution runner reported an error"
        return (
            "runtime_error",
            f"runner exited unsuccessfully ({result.returncode})",
        )

    def _build_script(
        self,
        code: str,
        test_code: str,
        entry_point: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build a test script plus an unpredictable completion marker.

        A zero exit status alone is not evidence that the hidden tests ran:
        generated code can call ``sys.exit(0)`` or ``os._exit(0)`` before the
        assertions.  The marker is generated after model inference and is
        emitted only after the trusted harness returns, so the parent evaluator
        can distinguish a completed suite from a premature successful exit.
        """
        code = textwrap.dedent(code)
        test_code = textwrap.dedent(test_code)
        completion_marker = (
            "__AWF_TEST_COMPLETE__" + secrets.token_hex(32)
        )

        parts = [
            "# Generated code",
            code,
            "",
            "# Test code",
            test_code,
        ]

        if entry_point:
            if not re.fullmatch(r"[A-Za-z_]\w*", entry_point):
                raise ValueError(f"Invalid entry point: {entry_point!r}")
            parts.extend(
                [
                    "",
                    "# HumanEval harness",
                    f"assert callable({entry_point}), "
                    f"'Entry point {entry_point} is not callable'",
                    "assert callable(check), "
                    "'HumanEval test code must define check(candidate)'",
                    f"check({entry_point})",
                ]
            )

        parts.extend(
            [
                "",
                "# Trusted parent-observed completion signal",
                f"print({completion_marker!r}, flush=True)",
            ]
        )
        return "\n".join(parts), completion_marker

    @staticmethod
    def extract_code(response: str) -> str:
        """Extract fenced Python code, tolerating a missing closing fence."""
        python_fence = re.search(r"```(?:python|py)\s*", response, re.IGNORECASE)
        generic_fence = re.search(r"```\s*", response)
        fence = python_fence or generic_fence
        if fence is None:
            return response.strip()

        start = fence.end()
        end = response.find("```", start)
        if end < 0:
            end = len(response)
        return response[start:end].strip()
