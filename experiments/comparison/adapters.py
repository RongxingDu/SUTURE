"""Method adapter protocol and artifact-backed adapter loading."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from awf.protocol.manifest import file_sha256, json_sha256
from experiments.comparison.models import (
    AdapterInference,
    ComparisonSample,
    PhaseTelemetry,
    validate_method_kind,
)


class ArtifactNotReadyError(FileNotFoundError):
    """Raised before evaluation when a declared method artifact is absent."""


class TelemetryUnavailableError(ValueError):
    """Raised when a searched method has no recorded search telemetry."""


@runtime_checkable
class MethodAdapter(Protocol):
    """Minimal contract implemented by Vanilla/AFlow/AWF/S-CWU adapters."""

    name: str
    kind: str
    artifact_path: Path | None
    search_telemetry: PhaseTelemetry

    def validate_ready(self) -> None:
        """Fail before any benchmark execution when the method is unavailable."""

    def identity(self) -> Mapping[str, Any]:
        """Return a credential-free, stable scientific method identity."""

    async def infer(self, sample: ComparisonSample) -> AdapterInference:
        """Run one already-frozen method on one sample."""


InferenceCallable = Callable[
    [ComparisonSample],
    AdapterInference | Awaitable[AdapterInference],
]


class CallableMethodAdapter:
    """Small adapter useful for research integrations and deterministic tests."""

    def __init__(
        self,
        *,
        name: str,
        kind: str,
        infer: InferenceCallable,
        search_telemetry: PhaseTelemetry | None = None,
        artifact_path: str | Path | None = None,
        identity_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("adapter name must be non-empty")
        if not callable(infer):
            raise TypeError("infer must be callable")
        self.name = normalized_name
        self.kind = validate_method_kind(kind)
        self._infer = infer
        self.artifact_path = (
            Path(artifact_path).expanduser().resolve()
            if artifact_path is not None
            else None
        )
        if search_telemetry is None:
            if self.kind == "vanilla":
                search_telemetry = PhaseTelemetry()
            else:
                raise TelemetryUnavailableError(
                    f"{self.name}: {self.kind} requires separately recorded "
                    "search telemetry; zero must be explicit when genuine"
                )
        self.search_telemetry = search_telemetry
        self._identity_metadata = dict(identity_metadata or {})

    def validate_ready(self) -> None:
        if self.kind == "aflow":
            _validate_aflow_artifact(self.name, self.artifact_path)
        elif self.artifact_path is not None:
            _validate_generic_artifact(self.name, self.artifact_path)

    def identity(self) -> Mapping[str, Any]:
        identity: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "artifact_sha256": (
                _artifact_sha256(self.artifact_path)
                if self.artifact_path is not None
                else None
            ),
            "metadata": self._identity_metadata,
        }
        return identity

    async def infer(self, sample: ComparisonSample) -> AdapterInference:
        result = self._infer(sample)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, AdapterInference):
            raise TypeError(
                f"{self.name}.infer must return AdapterInference, "
                f"got {type(result).__name__}"
            )
        return result


class ConfiguredMethodAdapter(CallableMethodAdapter):
    """Adapter assembled from one CLI method specification."""


def validate_adapters_ready(adapters: list[MethodAdapter]) -> None:
    """Validate the complete roster before the first inference call."""
    if len(adapters) < 2:
        raise ValueError("Paired comparison requires at least two methods")
    names = [adapter.name for adapter in adapters]
    if len(set(names)) != len(names):
        raise ValueError("Method adapter names must be unique")
    for adapter in adapters:
        adapter.validate_ready()
        # Compute identities now too.  An unreadable artifact must fail before
        # a different method consumes API budget.
        json_sha256(adapter.identity())


def validate_configured_specs_ready(
    specs: list[Mapping[str, Any]],
) -> None:
    """Preflight every configured artifact before importing any factory."""
    if len(specs) < 2:
        raise ValueError("Paired comparison requires at least two methods")
    names: list[str] = []
    for spec in specs:
        if not isinstance(spec, Mapping):
            raise TypeError("method specification must be a mapping")
        try:
            name = str(spec["name"]).strip()
            kind = validate_method_kind(str(spec["kind"]))
        except KeyError as exc:
            raise ValueError(
                f"Method specification missing required field {exc.args[0]!r}"
            ) from exc
        if not name:
            raise ValueError("adapter name must be non-empty")
        names.append(name)
        artifact_value = spec.get("artifact_path")
        artifact_path = (
            Path(str(artifact_value)).expanduser().resolve()
            if artifact_value is not None
            else None
        )
        if kind == "aflow":
            _validate_aflow_artifact(name, artifact_path)
        elif artifact_path is not None:
            _validate_generic_artifact(name, artifact_path)
        if spec.get("search_telemetry") is None and kind != "vanilla":
            raise TelemetryUnavailableError(
                f"{name}: missing search_telemetry for searched method {kind}"
            )
        if spec.get("search_telemetry") is not None:
            PhaseTelemetry.from_mapping(spec["search_telemetry"])
    if len(set(names)) != len(names):
        raise ValueError("Method adapter names must be unique")


def load_configured_adapter(spec: Mapping[str, Any]) -> ConfiguredMethodAdapter:
    """Load a CLI adapter without importing its factory before artifact checks.

    Factory syntax is ``"package.module:callable"``.  The callable receives a
    shallow copy of the method specification and may return either another
    ``MethodAdapter`` or an inference callable.
    """
    if not isinstance(spec, Mapping):
        raise TypeError("method specification must be a mapping")
    allowed = {
        "name",
        "kind",
        "factory",
        "artifact_path",
        "search_telemetry",
        "identity_metadata",
    }
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(
            "Unknown method specification fields: "
            + ", ".join(sorted(unknown))
        )
    try:
        name = str(spec["name"]).strip()
        kind = validate_method_kind(str(spec["kind"]))
        factory_path = str(spec["factory"]).strip()
    except KeyError as exc:
        raise ValueError(
            f"Method specification missing required field {exc.args[0]!r}"
        ) from exc
    artifact_value = spec.get("artifact_path")
    artifact_path = (
        Path(str(artifact_value)).expanduser().resolve()
        if artifact_value is not None
        else None
    )

    # AFlow must fail before importing a factory that could initialize an API
    # client or otherwise suggest that an absent graph was evaluated.
    if kind == "aflow":
        _validate_aflow_artifact(name, artifact_path)
    elif artifact_path is not None:
        _validate_generic_artifact(name, artifact_path)

    telemetry_value = spec.get("search_telemetry")
    if telemetry_value is None and kind != "vanilla":
        raise TelemetryUnavailableError(
            f"{name}: missing search_telemetry for searched method {kind}"
        )
    search_telemetry = (
        PhaseTelemetry()
        if telemetry_value is None
        else PhaseTelemetry.from_mapping(telemetry_value)
    )

    factory = _resolve_factory(factory_path)
    delegate = factory(dict(spec))
    if isinstance(delegate, MethodAdapter):
        infer = delegate.infer
        delegate_identity = dict(delegate.identity())
        if delegate.name != name:
            raise ValueError(
                f"Adapter factory {factory_path!r} returned method name "
                f"{delegate.name!r}, expected {name!r}"
            )
        if validate_method_kind(delegate.kind) != kind:
            raise ValueError(
                f"Adapter factory {factory_path!r} returned method kind "
                f"{delegate.kind!r}, expected {kind!r}"
            )
    elif callable(delegate):
        infer = delegate
        delegate_identity = None
    else:
        raise TypeError(
            f"Adapter factory {factory_path!r} must return a MethodAdapter "
            "or inference callable"
        )
    return ConfiguredMethodAdapter(
        name=name,
        kind=kind,
        infer=infer,
        search_telemetry=search_telemetry,
        artifact_path=artifact_path,
        identity_metadata={
            "factory": factory_path,
            **dict(spec.get("identity_metadata") or {}),
            **(
                {"delegate_identity": delegate_identity}
                if delegate_identity is not None
                else {}
            ),
        },
    )


def method_identity_sha256(adapter: MethodAdapter) -> str:
    return json_sha256(adapter.identity())


def _resolve_factory(value: str) -> Callable[[dict[str, Any]], Any]:
    if not value or ":" not in value:
        raise ValueError(
            "factory must use 'package.module:callable' syntax"
        )
    module_name, attribute_name = value.rsplit(":", 1)
    if not module_name or not attribute_name:
        raise ValueError(
            "factory must use 'package.module:callable' syntax"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError(f"Adapter factory {value!r} is not callable")
    return factory


def _validate_aflow_artifact(
    method_name: str,
    artifact_path: Path | None,
) -> None:
    if artifact_path is None:
        raise ArtifactNotReadyError(
            f"{method_name}: AFlow requires a generated workflow artifact; "
            "no artifact_path was supplied"
        )
    if not artifact_path.exists():
        raise ArtifactNotReadyError(
            f"{method_name}: AFlow artifact does not exist: {artifact_path}"
        )
    graph_path = (
        artifact_path / "graph.py"
        if artifact_path.is_dir()
        else artifact_path
    )
    if not graph_path.is_file() or graph_path.stat().st_size <= 0:
        raise ArtifactNotReadyError(
            f"{method_name}: AFlow artifact is not a non-empty graph file: "
            f"{graph_path}"
        )


def _validate_generic_artifact(
    method_name: str,
    artifact_path: Path,
) -> None:
    if not artifact_path.exists():
        raise ArtifactNotReadyError(
            f"{method_name}: method artifact does not exist: {artifact_path}"
        )
    if artifact_path.is_file() and artifact_path.stat().st_size <= 0:
        raise ArtifactNotReadyError(
            f"{method_name}: method artifact is empty: {artifact_path}"
        )
    if artifact_path.is_dir() and not any(
        child.is_file()
        for child in artifact_path.rglob("*")
        if "__pycache__" not in child.parts and child.suffix != ".pyc"
    ):
        raise ArtifactNotReadyError(
            f"{method_name}: artifact directory contains no files: "
            f"{artifact_path}"
        )


def _artifact_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    if path.is_file():
        return file_sha256(path)
    files = sorted(
        child
        for child in path.rglob("*")
        if child.is_file()
        and "__pycache__" not in child.parts
        and child.suffix != ".pyc"
    )
    return json_sha256(
        [
            {
                "relative_path": child.relative_to(path).as_posix(),
                "sha256": file_sha256(child),
            }
            for child in files
        ]
    )
