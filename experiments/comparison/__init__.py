"""Research-oriented paired method comparison utilities.

The comparison layer deliberately treats workflow search and workflow
inference as separate phases.  It evaluates already-frozen method artifacts;
it never performs workflow optimization while benchmark examples are being
compared.
"""

from experiments.comparison.adapters import (
    ArtifactNotReadyError,
    CallableMethodAdapter,
    ConfiguredMethodAdapter,
    MethodAdapter,
    TelemetryUnavailableError,
    load_configured_adapter,
    validate_adapters_ready,
    validate_configured_specs_ready,
)
from experiments.comparison.models import (
    AdapterInference,
    ComparisonSample,
    FrozenMethodRoster,
    PhaseTelemetry,
)
from experiments.comparison.runner import ComparisonRunner

__all__ = [
    "AdapterInference",
    "ArtifactNotReadyError",
    "CallableMethodAdapter",
    "ComparisonRunner",
    "ComparisonSample",
    "ConfiguredMethodAdapter",
    "FrozenMethodRoster",
    "MethodAdapter",
    "PhaseTelemetry",
    "TelemetryUnavailableError",
    "load_configured_adapter",
    "validate_adapters_ready",
    "validate_configured_specs_ready",
]
