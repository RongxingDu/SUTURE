"""Built-in inference adapters for frozen AWF and AFlow artifacts.

These adapters intentionally do not perform workflow search.  Search must
finish first and its telemetry must be supplied by the comparison
specification.  This keeps search cost separate from frozen-workflow
inference cost.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from awf.config.loader import load_config
from awf.executor.runtime import RuntimeExecutor
from awf.protocol.manifest import (
    file_sha256,
    json_sha256,
    public_scientific_config,
)
from awf.scheduler.graph_scheduler import GraphScheduler
from awf.workflow.serializer import load_workflow
from experiments.comparison.adapters import CallableMethodAdapter
from experiments.comparison.models import (
    AdapterInference,
    ComparisonSample,
    PhaseTelemetry,
    validate_method_kind,
)
from experiments.comparison.source_policy import (
    POLICY_VERSION,
    validate_generated_aflow_directory,
)


_AFLOW_GPQA_WORKFLOW_RETRY_MAX_ATTEMPTS = 5
_AFLOW_GPQA_WORKFLOW_RETRY_WAIT_SECONDS = 1.0


def build_awf_artifact_adapter(
    spec: Mapping[str, Any],
) -> CallableMethodAdapter:
    """Build a zero-scheduler-LLM adapter for an AWF YAML checkpoint.

    Required ``identity_metadata.runtime`` fields:

    ``config_path``
        Absolute or current-working-directory-relative experiment config.
    ``benchmark``
        One of the benchmark names supported by AWF's experiment scripts.

    The API key is resolved by ``load_config`` from the environment.  It is
    never accepted in a comparison specification or included in the identity.
    """

    values = _common_values(spec)
    runtime = _runtime_metadata(spec)
    config_path = _required_path(runtime, "config_path")
    benchmark = _required_text(runtime, "benchmark").lower()
    config = load_config(config_path)
    workflow = load_workflow(values["artifact_path"])
    operators = _operators_for_benchmark(benchmark)
    executor = RuntimeExecutor(config.executor, operators=operators)
    scheduler = GraphScheduler(config.scheduler)

    # Import lazily enough that artifact/config validation happens before a
    # provider client is constructed.
    from awf.llm.client import AsyncLLMClient

    llm_client = AsyncLLMClient(config.scheduler.llm)
    public_config = public_scientific_config(config)
    project_root = Path(__file__).resolve().parents[2]

    async def infer(sample: ComparisonSample) -> AdapterInference:
        started = time.perf_counter()
        output, _context, recorder = await executor.execute(
            workflow,
            scheduler,
            sample.query,
            llm_client=llm_client,
        )
        wall_latency = time.perf_counter() - started
        trace = recorder.trace
        llm_latency = sum(
            call.latency_seconds
            for step in trace.steps
            for call in step.llm_calls
        )
        return AdapterInference(
            output=output,
            telemetry=PhaseTelemetry(
                prompt_tokens=trace.total_prompt_tokens,
                completion_tokens=trace.total_completion_tokens,
                llm_calls=trace.total_llm_calls,
                llm_latency_seconds=llm_latency,
                wall_latency_seconds=wall_latency,
            ),
            metadata={
                "workflow_version": trace.workflow_version,
                "runtime_success": trace.success,
            },
        )

    return CallableMethodAdapter(
        name=values["name"],
        kind=values["kind"],
        infer=infer,
        artifact_path=values["artifact_path"],
        search_telemetry=values["search_telemetry"],
        identity_metadata={
            "implementation": "awf_graph_runtime",
            "benchmark": benchmark,
            "config_sha256": json_sha256(public_config),
            "config_file_sha256": file_sha256(config_path),
            "scheduler": "graph",
            "runtime_tree_sha256": _content_tree_sha256(
                project_root,
                (
                    project_root / "awf",
                    project_root / "benchmarks",
                    project_root / "experiments" / "comparison",
                    project_root / "experiments" / "scripts",
                    project_root / "experiments" / "workflows",
                ),
            ),
        },
    )


def build_aflow_artifact_adapter(
    spec: Mapping[str, Any],
) -> CallableMethodAdapter:
    """Build an adapter around one already-generated AFlow ``graph.py``.

    ``identity_metadata.runtime`` additionally requires ``aflow_root``.  The
    workflow is instantiated once, but cumulative AFlow usage is differenced
    around every inference.  Provider latency is measured by wrapping the
    OpenAI-compatible ``completions.create`` call because upstream AFlow does
    not record latency.
    """

    values = _common_values(spec)
    if values["kind"] != "aflow":
        raise ValueError("build_aflow_artifact_adapter requires kind='aflow'")
    runtime = _runtime_metadata(spec)
    config_path = _required_path(runtime, "config_path")
    aflow_root = _required_path(runtime, "aflow_root")
    benchmark = _required_text(runtime, "benchmark")
    if not (aflow_root / "scripts" / "async_llm.py").is_file():
        raise FileNotFoundError(
            f"AFlow root has no scripts/async_llm.py: {aflow_root}"
        )

    config = load_config(config_path)
    llm_config = config.scheduler.llm
    if not llm_config.api_key:
        raise ValueError(
            "The configured API key is empty; inject it through the "
            "environment variable referenced by the experiment config"
        )
    graph_path = values["artifact_path"]
    if graph_path.is_dir():
        graph_path = graph_path / "graph.py"
    module = _load_aflow_graph(graph_path, aflow_root)
    workflow_class = getattr(module, "Workflow", None)
    if not callable(workflow_class):
        raise TypeError(f"AFlow graph has no callable Workflow: {graph_path}")
    workflow = workflow_class(
        name=values["name"],
        llm_config={
            "model": llm_config.model,
            "temperature": llm_config.temperature,
            "key": llm_config.api_key,
            "base_url": llm_config.api_base,
            "top_p": float(llm_config.extra_kwargs.get("top_p", 1.0)),
            "max_tokens": llm_config.max_tokens,
            "timeout_seconds": llm_config.timeout_seconds,
            "max_retries": llm_config.max_retries,
            "extra_kwargs": {
                key: value
                for key, value in llm_config.extra_kwargs.items()
                if key != "top_p"
            },
            "telemetry_role": "workflow_execution",
        },
        dataset=benchmark,
    )
    latency_meter = _install_aflow_latency_meter(workflow)
    public_config = public_scientific_config(config)
    workflow_retry_policy = _aflow_workflow_retry_policy(benchmark)

    async def infer(sample: ComparisonSample) -> AdapterInference:
        before = _aflow_usage_snapshot(workflow)
        latency_before = latency_meter["seconds"]
        calls_before = latency_meter["calls"]
        started = time.perf_counter()
        max_workflow_attempts = int(workflow_retry_policy["max_attempts"])
        retry_wait_seconds = float(workflow_retry_policy["wait_seconds"])
        raw_result: Any = ""
        runtime_success = False
        attempt_count = 0
        failure_type: str | None = None
        for attempt_count in range(1, max_workflow_attempts + 1):
            try:
                raw_result = await _invoke_aflow_workflow(
                    workflow,
                    sample,
                    benchmark=benchmark,
                )
                runtime_success = True
                break
            except Exception as exc:
                failure_type = type(exc).__name__
                if attempt_count < max_workflow_attempts:
                    # Match AFlow's GPQA tenacity policy: wait_fixed(1)
                    # between failed full-workflow attempts.  The delay is part
                    # of end-to-end wall latency, but not provider latency.
                    await asyncio.sleep(retry_wait_seconds)
        wall_latency = time.perf_counter() - started
        after = _aflow_usage_snapshot(workflow)
        output = (
            raw_result[0]
            if isinstance(raw_result, tuple) and raw_result
            else raw_result
        )
        return AdapterInference(
            output=output,
            telemetry=PhaseTelemetry(
                prompt_tokens=(
                    after["prompt_tokens"] - before["prompt_tokens"]
                ),
                completion_tokens=(
                    after["completion_tokens"] - before["completion_tokens"]
                ),
                # Count every logical SDK invocation, including calls that
                # raise before a provider usage object is available.
                llm_calls=int(latency_meter["calls"] - calls_before),
                llm_latency_seconds=(
                    latency_meter["seconds"] - latency_before
                ),
                wall_latency_seconds=wall_latency,
            ),
            metadata={
                "runtime": "aflow_generated_graph",
                "runtime_success": runtime_success,
                "workflow_attempt_count": attempt_count,
                "workflow_retry_max_attempts": max_workflow_attempts,
                "workflow_retry_wait_seconds": retry_wait_seconds,
                "workflow_retry_semantics": str(
                    workflow_retry_policy["semantics"]
                ),
                **(
                    {"failure_type": failure_type}
                    if failure_type is not None and not runtime_success
                    else {}
                ),
            },
        )

    return CallableMethodAdapter(
        name=values["name"],
        kind=values["kind"],
        infer=infer,
        artifact_path=values["artifact_path"],
        search_telemetry=values["search_telemetry"],
        identity_metadata={
            "implementation": "aflow_generated_graph",
            "benchmark": benchmark,
            "aflow_commit": _git_head(aflow_root),
            "generated_source_policy": POLICY_VERSION,
            "workflow_retry_policy": dict(workflow_retry_policy),
            "config_sha256": json_sha256(public_config),
            "config_file_sha256": file_sha256(config_path),
            "runtime_tree_sha256": _content_tree_sha256(
                aflow_root,
                (
                    aflow_root / "scripts",
                    aflow_root / "benchmarks",
                    aflow_root
                    / "workspace"
                    / benchmark.upper()
                    / "workflows"
                    / "template",
                ),
            ),
        },
    )


def _aflow_workflow_retry_policy(benchmark: str) -> dict[str, Any]:
    """Return the frozen full-workflow retry policy for one benchmark."""
    if benchmark.strip().lower() == "gpqa":
        return {
            "max_attempts": _AFLOW_GPQA_WORKFLOW_RETRY_MAX_ATTEMPTS,
            "wait_seconds": _AFLOW_GPQA_WORKFLOW_RETRY_WAIT_SECONDS,
            "retry_on": "Exception",
            "semantics": "AFlow GPQA stop_after_attempt(5)+wait_fixed(1)",
        }
    return {
        "max_attempts": 1,
        "wait_seconds": 0.0,
        "retry_on": "none",
        "semantics": "single full-workflow attempt",
    }


def _common_values(spec: Mapping[str, Any]) -> dict[str, Any]:
    name = _required_text(spec, "name")
    kind = validate_method_kind(_required_text(spec, "kind"))
    artifact_path = _required_path(spec, "artifact_path")
    telemetry_value = spec.get("search_telemetry")
    search_telemetry = (
        PhaseTelemetry()
        if telemetry_value is None and kind == "vanilla"
        else PhaseTelemetry.from_mapping(telemetry_value)
    )
    return {
        "name": name,
        "kind": kind,
        "artifact_path": artifact_path,
        "search_telemetry": search_telemetry,
    }


def _runtime_metadata(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = spec.get("identity_metadata")
    if not isinstance(identity, Mapping):
        raise ValueError("identity_metadata must contain a runtime mapping")
    runtime = identity.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("identity_metadata.runtime must be a mapping")
    return runtime


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_path(mapping: Mapping[str, Any], key: str) -> Path:
    return Path(_required_text(mapping, key)).expanduser().resolve()


def _operators_for_benchmark(benchmark: str) -> dict[str, Any]:
    if benchmark == "math":
        from experiments.workflows.math.operators import extract_final_answer

        return {
            "finalize": extract_final_answer,
            "extract_final_answer": extract_final_answer,
        }
    if benchmark == "code_gen":
        from experiments.workflows.code_gen.operators import extract_final_code

        return {
            "finalize": extract_final_code,
            "extract_final_code": extract_final_code,
        }
    if benchmark == "scicode":
        from experiments.workflows.scicode.operators import (
            extract_scicode_code,
        )

        return {
            "finalize": extract_scicode_code,
            "extract_scicode_code": extract_scicode_code,
        }
    if benchmark in {"gpqa", "mmlu"}:
        return {}
    raise ValueError(f"Unsupported built-in adapter benchmark: {benchmark}")


def _load_aflow_graph(graph_path: Path, aflow_root: Path) -> ModuleType:
    if not graph_path.is_file():
        raise FileNotFoundError(f"AFlow graph not found: {graph_path}")
    validate_generated_aflow_directory(graph_path.parent)
    root_text = str(aflow_root)
    artifact_root_text = str(_aflow_artifact_import_root(graph_path))
    module_name = "awf_comparison_aflow_" + file_sha256(graph_path)[:16]
    module_spec = importlib.util.spec_from_file_location(
        module_name,
        graph_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot load AFlow graph: {graph_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    # AFlow uses a top-level ``benchmarks`` package, which collides with
    # AWF's own package of the same name.  Load the generated module with a
    # temporary AFlow namespace, then restore AWF's already-imported modules.
    saved_benchmarks = {
        name: value
        for name, value in sys.modules.items()
        if name == "benchmarks" or name.startswith("benchmarks.")
    }
    original_sys_path = list(sys.path)
    for name in saved_benchmarks:
        sys.modules.pop(name, None)
    # Generated graphs import their prompt through the canonical
    # ``workspace.<BENCHMARK>.workflows.round_2`` name.  Clear any prior AFlow
    # workspace namespace and put the frozen artifact tree first so an archived
    # snapshot cannot silently import a mutable checkout prompt.py.
    for name in list(sys.modules):
        if name == "workspace" or name.startswith("workspace."):
            sys.modules.pop(name, None)
    sys.path[:] = [
        artifact_root_text,
        *(
            [root_text]
            if root_text != artifact_root_text
            else []
        ),
        *(
            item
            for item in original_sys_path
            if item not in {root_text, artifact_root_text}
        ),
    ]
    try:
        module_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        for name in list(sys.modules):
            if name == "benchmarks" or name.startswith("benchmarks."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_benchmarks)
        sys.path[:] = original_sys_path
    return module


def _aflow_artifact_import_root(graph_path: Path) -> Path:
    """Resolve the tree root above ``workspace/GPQA/workflows/round_2``."""
    round_dir = graph_path.resolve().parent
    if (
        round_dir.name != "round_2"
        or round_dir.parent.name != "workflows"
        or round_dir.parent.parent.name.upper() != "GPQA"
        or round_dir.parent.parent.parent.name != "workspace"
    ):
        raise ValueError(
            "AFlow artifact must preserve workspace/GPQA/workflows/round_2"
        )
    return round_dir.parents[3]


def _install_aflow_latency_meter(workflow: Any) -> dict[str, float]:
    try:
        completions = workflow.llm.aclient.chat.completions
        original_create = completions.create
    except AttributeError as exc:
        raise TypeError(
            "AFlow Workflow does not expose llm.aclient.chat.completions"
        ) from exc
    meter = {"seconds": 0.0, "calls": 0.0}

    async def timed_create(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        meter["calls"] += 1.0
        try:
            return await original_create(*args, **kwargs)
        finally:
            meter["seconds"] += time.perf_counter() - started

    completions.create = timed_create
    return meter


def _aflow_usage_snapshot(workflow: Any) -> dict[str, int]:
    try:
        summary = workflow.llm.get_usage_summary()
    except AttributeError as exc:
        raise TypeError(
            "AFlow Workflow does not expose llm.get_usage_summary()"
        ) from exc
    return {
        "prompt_tokens": int(summary.get("total_input_tokens", 0)),
        "completion_tokens": int(summary.get("total_output_tokens", 0)),
        "llm_calls": int(
            summary.get("call_count", len(summary.get("history", [])))
        ),
    }


async def _invoke_aflow_workflow(
    workflow: Any,
    sample: ComparisonSample,
    *,
    benchmark: str,
) -> Any:
    if benchmark.lower() in {"humaneval", "code_gen"}:
        entry_point = sample.metadata.get("entry_point")
        if not isinstance(entry_point, str) or not entry_point:
            raise ValueError(
                "Blind code inference sample has no public entry_point"
            )
        return await workflow(sample.query, entry_point)
    return await workflow(sample.query)


def _git_head(repo: Path) -> str | None:
    head = repo / ".git" / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = repo / ".git" / value[5:]
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
        packed = repo / ".git" / "packed-refs"
        if packed.is_file():
            suffix = value[5:]
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith(("#", "^")):
                    commit, ref_name = line.split(" ", 1)
                    if ref_name == suffix:
                        return commit
        return None
    return value or None


def _content_tree_sha256(
    base: Path,
    roots: tuple[Path, ...],
) -> str:
    records: list[dict[str, str]] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(
                f"Runtime identity root does not exist: {root}"
            )
        files = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in files:
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix == ".pyc"
            ):
                continue
            records.append(
                {
                    "relative_path": path.relative_to(base).as_posix(),
                    "sha256": file_sha256(path),
                }
            )
    return json_sha256(records)
