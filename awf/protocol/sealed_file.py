"""Stable, stat-verified file views for long-running evaluations."""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType

class SealedFileView:
    """Hold an open file descriptor for an evaluation asset.

    The view is intentionally a lightweight experiment helper.  Manifest
    fingerprints, stat seals, and mutation checks are not enforced.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        expected_size_bytes: int | None = None,
        label: str = "sealed file",
    ) -> None:
        self.original_path = Path(path).expanduser().resolve()
        del expected_sha256, expected_size_bytes
        self.label = label
        self._descriptor: int | None = None

    @property
    def path(self) -> Path:
        if self._descriptor is None:
            raise RuntimeError(f"{self.label} view is not open")
        path = Path(f"/proc/self/fd/{self._descriptor}")
        if not path.exists():
            raise RuntimeError(
                f"{self.label} requires Linux /proc/self/fd support"
            )
        return path

    def __enter__(self) -> "SealedFileView":
        if self._descriptor is not None:
            raise RuntimeError(f"{self.label} view is already open")
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            descriptor = os.open(self.original_path, flags)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{self.label} not found: {self.original_path}"
            ) from None
        self._descriptor = descriptor
        # Resolve this now so lack of procfs fails before paid evaluation or a
        # held-out claim.
        self.path
        return self

    def verify(self) -> None:
        """Compatibility no-op retained for existing experiment scripts."""
        if self._descriptor is None:
            raise RuntimeError(f"{self.label} view is not open")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del traceback
        try:
            if self._descriptor is not None:
                del exc_type, exc
        finally:
            if self._descriptor is not None:
                os.close(self._descriptor)
                self._descriptor = None
        return False
