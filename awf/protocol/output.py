"""No-clobber reservations for long-running experiment artifacts."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from types import TracebackType
from typing import Any

from awf.protocol.manifest import atomic_write_json_0600


class OutputReservation:
    """Reserve a new path with ``O_EXCL`` before any paid work starts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.reservation_id = secrets.token_hex(16)
        self._identity: tuple[int, int] | None = None
        self._committed = False

    def __enter__(self) -> "OutputReservation":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            marker = {
                "status": "reserved",
                "reservation_id": self.reservation_id,
            }
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                json.dump(marker, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                value = os.fstat(handle.fileno())
                self._identity = (int(value.st_dev), int(value.st_ino))
            _fsync_directory(self.path.parent)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            self.path.unlink(missing_ok=True)
            raise
        return self

    def commit_json(self, value: Any) -> None:
        if self._committed or self._identity is None:
            raise RuntimeError("Output reservation is not active")
        current = os.stat(self.path, follow_symlinks=False)
        if (int(current.st_dev), int(current.st_ino)) != self._identity:
            raise RuntimeError(
                "Reserved output path was replaced during the experiment"
            )
        try:
            marker = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Reserved output path was modified during the experiment"
            ) from exc
        if (
            not isinstance(marker, dict)
            or marker.get("status") != "reserved"
            or marker.get("reservation_id") != self.reservation_id
        ):
            raise RuntimeError(
                "Reserved output path was modified during the experiment"
            )
        atomic_write_json_0600(self.path, value)
        self._committed = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc, traceback
        if not self._committed and self._identity is not None:
            try:
                current = os.stat(self.path, follow_symlinks=False)
                identity = (int(current.st_dev), int(current.st_ino))
                if identity == self._identity:
                    self.path.unlink()
                    _fsync_directory(self.path.parent)
            except FileNotFoundError:
                pass
        return False


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass
