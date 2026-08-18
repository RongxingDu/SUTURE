"""SciCode hidden-target loading and sandboxed evaluation."""

from __future__ import annotations

import ast
import base64
import json
import pickle
import re
import secrets
import site
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from benchmarks.code_generation.evaluator import (
    BubblewrapCodeRunner,
    CodeRunner,
    RunnerResult,
)


_COMPARATOR_IMPORT = re.compile(
    r"(?m)^\s*from\s+scicode\.compare\.cmp\s+import\s+[^\n]+\n?"
)
# The largest official target currently occupies 128,000,000 bytes before
# pickle framing.  Keep a finite ceiling, but leave enough room for the
# complete published HDF5 rather than silently making several official steps
# unevaluable.
_MAX_TARGET_PICKLE_BYTES = 192 * 1024 * 1024
_MOUNTED_TARGET_PATH_TOKEN = "__AWF_SCICODE_TARGET_PICKLE_PATH__"


@dataclass(frozen=True)
class SciCodeEvaluationResult:
    """Private-test outcome without hidden assertion details."""

    passed_tests: int
    total_tests: int
    tests_executed: int
    compiled: bool
    entry_point_present: bool
    outcome_codes: tuple[str, ...] = ()

    @property
    def pass_rate(self) -> float:
        return (
            self.passed_tests / self.total_tests
            if self.total_tests > 0
            else 0.0
        )

    @property
    def all_passed(self) -> bool:
        return (
            self.total_tests > 0
            and self.tests_executed == self.total_tests
            and self.passed_tests == self.total_tests
        )

    @property
    def execution_rate(self) -> float:
        return (
            min(self.tests_executed, self.total_tests) / self.total_tests
            if self.total_tests > 0
            else 0.0
        )


class ScientificBubblewrapRunner(BubblewrapCodeRunner):
    """Bubblewrap runner exposing only installed scientific packages.

    The AWF project, home directory, benchmark JSON and HDF5 targets remain
    unmounted.  A read-only site-packages mount is sufficient for NumPy,
    SciPy, SymPy and generated SciCode imports.
    """

    def __init__(
        self,
        *,
        site_packages: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        # Several official targets are tens to hundreds of MB and the tested
        # implementation may allocate an output of the same size.  This is
        # still bounded, but the generic 256 MB code-runner default cannot
        # execute the complete benchmark.
        kwargs.setdefault("memory_limit_mb", 1024)
        super().__init__(**kwargs)
        source = (
            Path(site_packages)
            if site_packages is not None
            else _find_site_packages()
        )
        if not source.is_dir():
            raise FileNotFoundError(
                f"Scientific site-packages not found: {source}"
            )
        self.scientific_site_packages = source.resolve()
        self.sandbox_site_packages = Path("/opt/awf-scientific-packages")

    def _build_command(self, script: str) -> list[str]:
        del script
        command = super()._build_command("")
        try:
            # The base command has five Python argv entries after the
            # bubblewrap ``--`` separator.  Locate the separator by value
            # instead of relying on that argv tail never changing.
            separator = command.index("--")
        except ValueError as exc:
            raise RuntimeError(
                "Unexpected bubblewrap command layout"
            ) from exc
        if separator >= len(command) - 1:
            raise RuntimeError("Unexpected bubblewrap command layout")
        scientific_mounts = [
            "--dir",
            "/opt",
            "--ro-bind",
            str(self.scientific_site_packages),
            str(self.sandbox_site_packages),
        ]
        fonts = Path("/etc/fonts")
        if fonts.is_dir():
            scientific_mounts.extend(
                [
                    "--dir",
                    "/etc",
                    "--ro-bind",
                    str(fonts),
                    "/etc/fonts",
                ]
            )
        scientific_mounts.extend(
            [
                # BLAS libraries otherwise reserve memory for many worker
                # threads. Under RLIMIT_AS that made even ``import numpy``
                # fail at the previous 256 MB default.
                "--setenv",
                "OPENBLAS_NUM_THREADS",
                "1",
                "--setenv",
                "OMP_NUM_THREADS",
                "1",
                "--setenv",
                "MKL_NUM_THREADS",
                "1",
                "--setenv",
                "HOME",
                "/tmp",
                "--setenv",
                "XDG_CACHE_HOME",
                "/tmp/.cache",
                "--setenv",
                "MPLCONFIGDIR",
                "/tmp/.matplotlib",
                "--setenv",
                "MPLBACKEND",
                "Agg",
            ]
        )
        command[separator:separator] = scientific_mounts
        if command[-2:] != ["-c", ""]:
            raise RuntimeError("Unexpected bubblewrap Python command layout")
        # Feed the harness through a non-seekable pipe.  This avoids Linux's
        # argv-size limit and does not expose hidden assertions through
        # ``/proc/*/cmdline`` or a mounted script file.
        command[-2:] = ["-"]
        return command

    def run(self, script: str, timeout_seconds: float) -> RunnerResult:
        return self._run_script(script, timeout_seconds)

    def run_with_target(
        self,
        script: str,
        target: Any,
        timeout_seconds: float,
    ) -> RunnerResult:
        """Run a harness with one trusted target mounted read-only.

        The full HDF5 remains outside the namespace.  A per-test pickle avoids
        embedding up to 128 MB of numeric data in Python source.
        """
        try:
            serialized = pickle.dumps(
                target,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except (pickle.PickleError, TypeError, ValueError) as exc:
            raise ValueError("SciCode target could not be serialized") from exc
        if len(serialized) > _MAX_TARGET_PICKLE_BYTES:
            raise ValueError("SciCode target exceeds the sandbox transfer limit")

        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix="awf-scicode-target-",
                suffix=".pickle",
            ) as target_file:
                target_file.write(serialized)
                target_file.flush()
                sandbox_target = (
                    f"/opt/.awf-target-{secrets.token_hex(16)}.pickle"
                )
                materialized_script = script.replace(
                    _MOUNTED_TARGET_PATH_TOKEN,
                    sandbox_target,
                )
                return self._run_script(
                    materialized_script,
                    timeout_seconds,
                    target_mount=(
                        Path(target_file.name),
                        Path(sandbox_target),
                    ),
                )
        finally:
            del serialized

    def _run_script(
        self,
        script: str,
        timeout_seconds: float,
        *,
        target_mount: tuple[Path, Path] | None = None,
    ) -> RunnerResult:
        payload = (
            "import sys\n"
            f"sys.path.insert(0, {str(self.sandbox_site_packages)!r})\n"
            + script
        )
        command = self._build_command(script)
        separator = command.index("--")
        if target_mount is not None:
            source, destination = target_mount
            command[separator:separator] = [
                "--ro-bind",
                str(source),
                str(destination),
            ]
        limiter = self._make_resource_limiter(timeout_seconds)

        try:
            with (
                tempfile.TemporaryFile(
                    mode="w+t",
                    encoding="utf-8",
                ) as stdout_file,
                tempfile.TemporaryFile(
                    mode="w+t",
                    encoding="utf-8",
                ) as stderr_file,
            ):
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    cwd="/",
                    env={},
                    start_new_session=True,
                    preexec_fn=limiter,
                )
                try:
                    process.communicate(
                        input=payload,
                        timeout=timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    self._terminate_process_group(process)
                    process.communicate()
                    return RunnerResult(
                        returncode=(
                            process.returncode
                            if process.returncode is not None
                            else -1
                        ),
                        stdout=self._read_bounded(stdout_file),
                        stderr=self._read_bounded(stderr_file),
                        timed_out=True,
                        error_message=(
                            f"Execution timed out after {timeout_seconds}s"
                        ),
                    )
                return RunnerResult(
                    returncode=process.returncode,
                    stdout=self._read_bounded(stdout_file),
                    stderr=self._read_bounded(stderr_file),
                )
        except Exception as exc:
            return RunnerResult(
                returncode=-1,
                error_message=f"Bubblewrap runner failed: {exc}",
            )


class SciCodeEvaluator:
    """Evaluate a generated subproblem implementation against HDF5 targets."""

    def __init__(
        self,
        *,
        private_test_specs: Mapping[str, Mapping[str, Any]],
        hdf5_path: str | Path,
        runner: CodeRunner | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.private_test_specs = {
            str(key): dict(value)
            for key, value in private_test_specs.items()
        }
        self.hdf5_path = Path(hdf5_path)
        self.runner = runner
        self.timeout_seconds = timeout_seconds
        self._asset_cache_key: tuple[int, int, int, int, int] | None = None
        self._asset_cache_result: dict[str, Any] | None = None

    def validate_assets(self) -> dict[str, Any]:
        """Fail fast when hidden targets or optional dependencies are absent."""
        if not self.hdf5_path.is_file():
            raise FileNotFoundError(
                "SciCode numeric targets are missing: "
                f"{self.hdf5_path}. Download official test_data.h5."
            )
        try:
            import h5py  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "SciCode evaluation requires the optional h5py dependency"
            ) from exc
        stat_before = self.hdf5_path.stat()
        cache_key = _stat_cache_key(stat_before)
        if (
            cache_key == self._asset_cache_key
            and self._asset_cache_result is not None
        ):
            return dict(self._asset_cache_result)
        stat_after = self.hdf5_path.stat()
        result = {
            "path": str(self.hdf5_path),
            "size_bytes": stat_after.st_size,
        }
        self._asset_cache_key = cache_key
        self._asset_cache_result = result
        return dict(result)

    @staticmethod
    def extract_code(output: Any) -> str:
        text = "" if output is None else str(output).strip()
        blocks = re.findall(
            r"```(?:python)?\s*\n?([\s\S]*?)```",
            text,
            flags=re.IGNORECASE,
        )
        return (blocks[-1] if blocks else text).strip()

    def evaluate_detailed(
        self,
        output: Any,
        ground_truth: Any,
    ) -> SciCodeEvaluationResult:
        task_id, spec = self._resolve_spec(ground_truth)
        tests = list(spec["tests"])
        code = self.extract_code(output)
        if not code:
            return SciCodeEvaluationResult(
                0,
                len(tests),
                0,
                False,
                False,
                tuple("missing_code" for _ in tests),
            )
        try:
            syntax_tree = ast.parse(code, filename="<generated>", mode="exec")
            compile(
                syntax_tree,
                "<generated>",
                "exec",
                flags=0,
                dont_inherit=True,
            )
        except (SyntaxError, TypeError, ValueError):
            return SciCodeEvaluationResult(
                0,
                len(tests),
                0,
                False,
                False,
                tuple("compile_error" for _ in tests),
            )
        entry_point = str(spec["entry_point"])
        entry_present = any(
            isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            )
            and node.name == entry_point
            for node in syntax_tree.body
        )
        if not entry_present:
            return SciCodeEvaluationResult(
                0,
                len(tests),
                0,
                True,
                False,
                tuple("missing_entry_point" for _ in tests),
            )
        if _has_unsafe_introspection(syntax_tree):
            return SciCodeEvaluationResult(
                0,
                len(tests),
                0,
                True,
                True,
                tuple("unsafe_code" for _ in tests),
            )
        if self.runner is None:
            return SciCodeEvaluationResult(
                0,
                len(tests),
                0,
                True,
                True,
                tuple("execution_disabled" for _ in tests),
            )

        self.validate_assets()
        targets = _load_hdf5_targets(
            self.hdf5_path,
            str(spec["step_id"]),
            len(tests),
        )
        passed = 0
        executed = 0
        outcomes: list[str] = []
        for test, target in zip(tests, targets):
            marker = f"__AWF_SCICODE_PASS_{secrets.token_hex(16)}__"
            try:
                if isinstance(self.runner, ScientificBubblewrapRunner):
                    script = _build_test_script(
                        code=code,
                        dependencies=str(spec.get("dependencies") or ""),
                        test=str(test),
                        target=None,
                        marker=marker,
                        mounted_target=True,
                    )
                    result = self.runner.run_with_target(
                        script,
                        target,
                        self.timeout_seconds,
                    )
                else:
                    script = _build_test_script(
                        code=code,
                        dependencies=str(spec.get("dependencies") or ""),
                        test=str(test),
                        target=target,
                        marker=marker,
                    )
                    result = self.runner.run(script, self.timeout_seconds)
            except (pickle.PickleError, TypeError, ValueError):
                outcomes.append("target_serialization_error")
                continue
            executed += 1
            if (
                result.returncode == 0
                and not result.timed_out
                and marker in result.stdout
            ):
                passed += 1
                outcomes.append("passed")
            else:
                outcomes.append(_classify_failure(result))
        if len(outcomes) < len(tests):
            outcomes.extend(
                "target_missing"
                for _ in range(len(tests) - len(outcomes))
            )
        del task_id
        return SciCodeEvaluationResult(
            passed,
            len(tests),
            executed,
            True,
            True,
            tuple(outcomes),
        )

    def _resolve_spec(
        self,
        ground_truth: Any,
    ) -> tuple[str, dict[str, Any]]:
        if not isinstance(ground_truth, dict):
            raise ValueError("SciCode ground truth must be a mapping")
        task_id = str(ground_truth.get("task_id") or "")
        spec = self.private_test_specs.get(task_id)
        if spec is None:
            raise KeyError(f"No private SciCode test spec for {task_id!r}")
        return task_id, spec


def _find_site_packages() -> Path:
    candidates = [
        Path(path)
        for path in site.getsitepackages()
        if "site-packages" in path
    ]
    for candidate in candidates:
        if (candidate / "numpy").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate a site-packages directory containing NumPy"
    )


def _build_test_script(
    *,
    code: str,
    dependencies: str,
    test: str,
    target: Any,
    marker: str,
    mounted_target: bool = False,
) -> str:
    if mounted_target:
        target_source = (
            "with open("
            f"{_MOUNTED_TARGET_PATH_TOKEN!r}, 'rb'"
            ") as _awf_target_file:\n"
            "    target = pickle.load(_awf_target_file)"
        )
    else:
        serialized = pickle.dumps(target, protocol=pickle.HIGHEST_PROTOCOL)
        if len(serialized) > _MAX_TARGET_PICKLE_BYTES:
            raise ValueError("SciCode target exceeds the sandbox transfer limit")
        encoded = base64.b64encode(serialized).decode("ascii")
        target_source = (
            f"target = pickle.loads(base64.b64decode({encoded!r}))"
        )
    safe_test = _COMPARATOR_IMPORT.sub("", test)
    solution_source = "\n\n".join((dependencies, code))
    return "\n\n".join(
        (
            "import base64\nimport pickle",
            _SCIPY_COMPAT_SOURCE,
            (
                f"_awf_solution_source = {solution_source!r}\n"
                "_awf_solution_globals = {'__builtins__': __builtins__}\n"
                "exec(compile(_awf_solution_source, '<generated>', 'exec'), "
                "_awf_solution_globals)\n"
                "globals().update({\n"
                "    key: value\n"
                "    for key, value in _awf_solution_globals.items()\n"
                "    if not key.startswith('__')\n"
                "})"
            ),
            _COMPARATOR_SOURCE,
            target_source,
            safe_test,
            f"print({marker!r})",
        )
    )


def _classify_failure(result: RunnerResult) -> str:
    if result.timed_out:
        return "timeout"
    combined = "\n".join(
        value
        for value in (result.stderr, result.stdout, result.error_message)
        if value
    )
    if "AssertionError" in combined:
        return "assertion_failure"
    if re.search(r"\b(?:SyntaxError|IndentationError|TabError)\b", combined):
        return "compile_error"
    if result.error_message:
        return "runner_error"
    if result.returncode == 0:
        return "harness_incomplete"
    return "runtime_error"


def _has_unsafe_introspection(tree: ast.AST) -> bool:
    """Reject obvious attempts to inspect the private evaluator harness.

    This is a benchmark-integrity boundary, not the OS security boundary (that
    remains bubblewrap).  Candidate code and hidden assertions must share a
    Python process to call scientific functions, so frame and filesystem
    introspection would otherwise expose ``target`` or its per-test mount.
    """
    blocked_modules = {
        "builtins",
        "glob",
        "importlib",
        "inspect",
        "pathlib",
        "subprocess",
        "sys",
    }
    blocked_calls = {
        "__import__",
        "breakpoint",
        "compile",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
    blocked_attributes = {
        "__builtins__",
        "__code__",
        "__dict__",
        "__getattr__",
        "__getattribute__",
        "__globals__",
        "__mro__",
        "__subclasses__",
        "__traceback__",
        "_getframe",
        "cr_frame",
        "currentframe",
        "f_back",
        "f_globals",
        "f_locals",
        "gi_frame",
        "glob",
        "iterdir",
        "listdir",
        "open",
        "popen",
        "read_bytes",
        "read_text",
        "rglob",
        "scandir",
        "system",
        "tb_frame",
        "walk",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name.partition(".")[0] in blocked_modules
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            if (
                node.module
                and node.module.partition(".")[0] in blocked_modules
            ):
                return True
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in blocked_calls
            ):
                return True
        elif isinstance(node, ast.Attribute):
            if node.attr in blocked_attributes:
                return True
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id == "__builtins__":
                return True
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(
                fragment in node.value
                for fragment in (
                    "/proc/",
                    "/opt/.awf-target-",
                    "awf-scicode-target-",
                )
            ):
                return True
    return False


def _load_hdf5_targets(
    path: Path,
    step_id: str,
    count: int,
) -> list[Any]:
    try:
        import h5py
        import scipy.sparse
    except ImportError as exc:
        raise RuntimeError(
            "SciCode target loading requires h5py and scipy"
        ) from exc

    def sparse(group: Any) -> Any:
        data = group["data"][()]
        shape = tuple(group["shape"][()])
        if "row" in group and "col" in group:
            return scipy.sparse.coo_matrix(
                (data, (group["row"][()], group["col"][()])),
                shape=shape,
            )
        indices = group["indices"][()]
        indptr = group["indptr"][()]
        if "blocksize" in group:
            return scipy.sparse.bsr_matrix(
                (data, indices, indptr),
                shape=shape,
                blocksize=tuple(group["blocksize"][()]),
            )
        return scipy.sparse.csr_matrix(
            (data, indices, indptr),
            shape=shape,
        )

    def group_value(group: Any) -> Any:
        if "list" in group:
            return [object_value(group["list"][key]) for key in group["list"]]
        if "sparse_matrix" in group:
            return sparse(group["sparse_matrix"])
        result: dict[Any, Any] = {}
        for key, value in group.items():
            parsed_key: Any = key
            try:
                parsed_key = float(key)
            except ValueError:
                pass
            if hasattr(value, "keys") and "sparse_matrix" in value:
                result[parsed_key] = sparse(value["sparse_matrix"])
            else:
                result[parsed_key] = object_value(value)
        return result

    def object_value(value: Any) -> Any:
        if hasattr(value, "keys"):
            return group_value(value)
        raw = value[()]
        return raw.decode("utf-8") if isinstance(raw, bytes) else raw

    targets: list[Any] = []
    with h5py.File(path, "r") as handle:
        for test_index in range(1, count + 1):
            group_path = f"{step_id}/test{test_index}"
            if group_path not in handle:
                raise KeyError(
                    f"SciCode HDF5 target is missing: {group_path}"
                )
            group = handle[group_path]
            values = [object_value(group[key]) for key in group]
            targets.append(values[0] if len(values) == 1 else tuple(values))
    return targets


def _file_sha256(path: Path) -> str:
    st = path.stat()
    return f"{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"


def _stat_cache_key(stat_result: Any) -> tuple[int, int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def _sha256_json(value: Any) -> str:
    """Simple JSON identity for test-spec comparison."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


_SCIPY_COMPAT_SOURCE = """
try:
    import scipy.integrate as _awf_scipy_integrate
    if (
        not hasattr(_awf_scipy_integrate, "simps")
        and hasattr(_awf_scipy_integrate, "simpson")
    ):
        _awf_scipy_integrate.simps = _awf_scipy_integrate.simpson
except ImportError:
    pass
"""


_COMPARATOR_SOURCE = r"""
import numpy as np
try:
    import scipy.sparse as _awf_sparse
except ImportError:
    _awf_sparse = None
try:
    import sympy as _awf_sympy
except ImportError:
    _awf_sympy = None

def _awf_process_symbols(value):
    result = {}
    for key, item in value.items():
        new_key = str(key) if _awf_sympy is not None and isinstance(key, _awf_sympy.Symbol) else key
        new_item = str(item) if _awf_sympy is not None and isinstance(item, _awf_sympy.Symbol) else item
        result[new_key] = new_item
    return result

def are_dicts_close(left, right, atol=1e-8, rtol=1e-5):
    left = _awf_process_symbols(left)
    right = _awf_process_symbols(right)
    if left.keys() != right.keys():
        return False
    sparse_types = ()
    if _awf_sparse is not None:
        sparse_types = (
            _awf_sparse.csr_matrix,
            _awf_sparse.csc_matrix,
            _awf_sparse.bsr_matrix,
            _awf_sparse.coo_matrix,
        )
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, sparse_types):
            a, b = a.toarray(), b.toarray()
        try:
            if not np.allclose(a, b, atol=atol, rtol=rtol):
                return False
        except (TypeError, ValueError):
            if a != b:
                return False
    return True

def cmp_tuple_or_list(left, right):
    if len(left) != len(right):
        return False
    sparse_types = ()
    if _awf_sparse is not None:
        sparse_types = (_awf_sparse.csr_matrix, _awf_sparse.csc_matrix)
    for a, b in zip(left, right):
        if isinstance(a, dict):
            if not are_dicts_close(a, b):
                return False
        elif isinstance(a, sparse_types):
            if not np.allclose(a.toarray(), b.toarray()):
                return False
        elif isinstance(a, bool):
            if a != b:
                return False
        else:
            try:
                if not np.allclose(a, b):
                    return False
            except (TypeError, ValueError):
                if a != b:
                    return False
    return True
"""
