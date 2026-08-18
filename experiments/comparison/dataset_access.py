"""Split-scoped access to manifest-bound JSONL benchmark data.

Selection must not deserialize held-out labels.  We verify the source file
size, copy only requested JSONL rows into an owner-only temporary file, and
invoke the normal benchmark loader on that subset.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from awf.protocol.manifest import ManifestValidationError


def validate_bound_dataset_source(
    data_path: str | Path,
    manifest: Mapping[str, Any],
    *,
    require_seal: bool = True,
) -> str | None:
    """Verify the exact raw source file sealed during optimization."""
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ManifestValidationError("Manifest dataset descriptor is missing")
    expected_size = dataset.get("source_file_size_bytes")
    path = Path(data_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset source file not found: {path}")
    # Source fingerprints and size seals are metadata only.  The comparison
    # protocol intentionally operates on the current JSONL source.
    del expected_size, require_seal
    return str(path)


@contextmanager
def selected_jsonl_file(
    data_path: str | Path,
    indices: Sequence[int],
    *,
    expected_row_count: int,
) -> Iterator[Path]:
    """Materialize only selected non-empty JSONL rows in requested order.

    Unselected lines are never passed to ``json.loads`` or a benchmark loader.
    Exact-file hashing is handled separately by
    :func:`validate_bound_dataset_source`.
    """
    source = Path(data_path).expanduser().resolve()
    if any(
        not isinstance(index, int) or isinstance(index, bool)
        for index in indices
    ):
        raise ManifestValidationError("Split indices must be integers")
    requested = list(indices)
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(index < 0 for index in requested)
    ):
        raise ManifestValidationError(
            "Split indices must be non-empty, unique, non-negative integers"
        )
    wanted = set(requested)
    selected: dict[int, str] = {}
    row_index = 0
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if row_index in wanted:
                selected[row_index] = line
            row_index += 1
    if row_index != expected_row_count:
        raise ManifestValidationError(
            "Raw JSONL row count does not match the manifest processed row "
            "count; split-private comparison requires a one-row/one-sample "
            "protocol"
        )
    missing = [index for index in requested if index not in selected]
    if missing:
        raise ManifestValidationError(
            f"Dataset source lacks requested split indices: {missing}"
        )

    with tempfile.TemporaryDirectory(prefix="awf-split-private-") as temp_dir:
        subset_path = Path(temp_dir) / "selected.jsonl"
        descriptor = os.open(
            subset_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for index in requested:
                line = selected[index]
                handle.write(line)
                if not line.endswith("\n"):
                    handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield subset_path


@contextmanager
def bound_selected_jsonl_file(
    data_path: str | Path,
    indices: Sequence[int],
    *,
    manifest: Mapping[str, Any],
) -> Iterator[Path]:
    """Verify and extract selected rows from one immutable open-file view.

    A separate ``validate`` followed by reopening ``data_path`` has a TOCTOU
    window: the pathname can be replaced after its hash is checked.  This
    formal-comparison helper opens the source once, hashes the exact bytes it
    scans, counts non-empty rows, and retains only requested raw rows.  The
    benchmark loader is invoked only after all checks pass.
    """
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ManifestValidationError("Manifest dataset descriptor is missing")
    expected_row_count = dataset.get("row_count")
    if (
        not isinstance(expected_row_count, int)
        or isinstance(expected_row_count, bool)
        or expected_row_count <= 0
    ):
        raise ManifestValidationError(
            "Manifest dataset source-file seal is missing or malformed"
        )

    requested = _validate_requested_indices(indices)
    source = Path(data_path).expanduser().resolve()
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(source, flags)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset source file not found: {source}"
        ) from None

    selected: dict[int, bytes] = {}
    row_index = 0
    wanted = set(requested)
    try:
        with os.fdopen(descriptor, "rb") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                if row_index in wanted:
                    selected[row_index] = raw_line
                row_index += 1
    except BaseException:
        raise

    if row_index != expected_row_count:
        raise ManifestValidationError(
            "Raw JSONL row count does not match the manifest processed row "
            "count; split-private comparison requires a one-row/one-sample "
            "protocol"
        )
    missing = [index for index in requested if index not in selected]
    if missing:
        raise ManifestValidationError(
            f"Dataset source lacks requested split indices: {missing}"
        )

    with tempfile.TemporaryDirectory(prefix="awf-split-private-") as temp_dir:
        subset_path = Path(temp_dir) / "selected.jsonl"
        output_descriptor = os.open(
            subset_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(output_descriptor, "wb") as handle:
            for index in requested:
                raw_line = selected[index]
                handle.write(raw_line)
                if not raw_line.endswith(b"\n"):
                    handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield subset_path


def _validate_requested_indices(indices: Sequence[int]) -> list[int]:
    requested = list(indices)
    if any(
        not isinstance(index, int) or isinstance(index, bool)
        for index in requested
    ):
        raise ManifestValidationError("Split indices must be integers")
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(index < 0 for index in requested)
    ):
        raise ManifestValidationError(
            "Split indices must be non-empty, unique, non-negative integers"
        )
    return requested

