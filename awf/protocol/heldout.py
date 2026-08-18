"""Shared, irreversible claims for every held-out evaluation entry point.

The legacy filename is retained because completed comparison runs already use
it.  Its semantics are broader now: whether the claimant is ``awf-test`` or a
frozen method comparison, the one dataset-bound held-out split is consumed.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from awf.protocol.manifest import atomic_write_json_0600


HELDOUT_LEDGER_FILENAME = "comparison_test_ledger.json"
_IMMUTABLE_CLAIM_KEYS = frozenset(
    {
        "schema_version",
        "claim_id",
        "status",
        "claimed_at",
        "access_mode",
        "benchmark",
        "requested_output_path",
    }
)


def heldout_test_ledger_path(protocol_results_path: str | Path) -> Path:
    """Return the single permanent ledger shared by all test commands."""
    return (
        Path(protocol_results_path).expanduser().resolve().parent
        / HELDOUT_LEDGER_FILENAME
    )


@contextmanager
def heldout_protocol_lock(
    ledger_path: str | Path,
    *,
    exclusive: bool,
) -> Iterator[None]:
    """Order validation publication before any held-out execution.

    Selection holds a shared lock through its durable output commit. Test
    entry points hold the exclusive lock from preflight through ledger/result
    completion. The permanent claim still enforces one-shot semantics after a
    process exits or crashes.
    """
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.protocol.lock")
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT,
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(descriptor, operation)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def claim_heldout_test(
    ledger_path: str | Path,
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish a complete claim before held-out label access."""
    path = Path(ledger_path)
    value = dict(claim)
    if value.get("status") != "claimed":
        raise ValueError("Held-out claim must start with status='claimed'")
    if not isinstance(value.get("claim_id"), str) or not value["claim_id"]:
        raise ValueError("Held-out claim must contain a non-empty claim_id")

    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_ledger_lock(path):
        _publish_claim_no_replace(path, value)
    return value


def complete_heldout_test(
    ledger_path: str | Path,
    *,
    claim_id: str,
    completion: Mapping[str, Any],
    expected_claim_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Complete one active claim without permitting claimant substitution."""
    path = Path(ledger_path)
    completion_value = dict(completion)
    with _exclusive_ledger_lock(path):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Held-out test ledger is missing or malformed"
            ) from exc
        if not isinstance(value, dict) or value.get("status") != "claimed":
            raise RuntimeError("Held-out test ledger is not an active claim")
        if value.get("claim_id") != claim_id:
            raise RuntimeError(
                "Held-out test claim_id changed before completion"
            )
        for key, expected in dict(expected_claim_fields or {}).items():
            if value.get(key) != expected:
                raise RuntimeError(
                    "Held-out test claim identity changed before completion: "
                    f"{key}"
                )

        conflicts = set(completion_value).intersection(value)
        reserved = set(completion_value).intersection(_IMMUTABLE_CLAIM_KEYS)
        forbidden = sorted(conflicts | reserved)
        if forbidden:
            raise ValueError(
                "Held-out completion cannot replace claim fields: "
                + ", ".join(forbidden)
            )

        completed = {
            **value,
            **completion_value,
            "status": "completed",
        }
        atomic_write_json_0600(path, completed)
        return completed


@contextmanager
def _exclusive_ledger_lock(ledger_path: Path) -> Iterator[None]:
    """Serialize ledger state transitions on one never-replaced lock inode."""
    lock_path = ledger_path.with_name(f".{ledger_path.name}.lock")
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT,
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _publish_claim_no_replace(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    """Write privately, then publish by an atomic no-replace hard link."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".claim.tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    published = False
    try:
        os.fchmod(descriptor, 0o600)
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(
                "The manifest-bound held-out split was already claimed; "
                f"refusing repeated test access ({path})"
            ) from exc
        published = True
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if published:
            _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes where the filesystem supports it."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # The claim is still atomically visible on filesystems without
        # directory fsync support.
        pass
