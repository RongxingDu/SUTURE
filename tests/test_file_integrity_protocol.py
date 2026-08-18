from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from awf.protocol.manifest import ManifestValidationError, file_sha256
from awf.protocol.output import OutputReservation
from awf.protocol.sealed_file import SealedFileView


def test_sealed_file_view_pins_verified_inode_across_path_replacement(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.h5"
    asset.write_bytes(b"original-target-bytes")
    expected_sha = file_sha256(asset)
    expected_size = asset.stat().st_size

    with SealedFileView(
        asset,
        expected_sha256=expected_sha,
        expected_size_bytes=expected_size,
        label="fixture asset",
    ) as view:
        replacement = tmp_path / "replacement.h5"
        replacement.write_bytes(b"different-target-data")
        os.replace(replacement, asset)

        assert view.path.read_bytes() == b"original-target-bytes"
        view.verify()


def test_sealed_file_view_detects_in_place_mutation(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.h5"
    alias = tmp_path / "alias.h5"
    asset.write_bytes(b"original")
    os.link(asset, alias)

    # Mutation that changes file size is detected via stat-identity check.
    with pytest.raises(ManifestValidationError, match="changed"):
        with SealedFileView(asset, label="fixture asset"):
            alias.write_bytes(b"modified-content-longer")


def test_output_reservation_refuses_concurrent_path_and_commits_json(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    with OutputReservation(output) as reservation:
        with pytest.raises(FileExistsError):
            with OutputReservation(output):
                pass
        reservation.commit_json({"result": "complete"})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "result": "complete"
    }
    assert output.stat().st_mode & 0o777 == 0o600


def test_failed_output_reservation_removes_only_its_marker(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="fixture failure"):
        with OutputReservation(output):
            raise RuntimeError("fixture failure")

    assert not output.exists()
