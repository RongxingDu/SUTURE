from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from awf.protocol import heldout
from awf.protocol.heldout import (
    claim_heldout_test,
    complete_heldout_test,
)


def _claim(claim_id: str = "claim-1") -> dict[str, object]:
    return {
        "schema_version": 1,
        "claim_id": claim_id,
        "status": "claimed",
        "claimed_at": "2026-07-27T00:00:00+00:00",
        "access_mode": "method_comparison",
        "benchmark": "gpqa",
        "manifest_sha256": "a" * 64,
        "experiment_tag": "frozen",
    }


def test_claim_is_published_only_after_complete_private_json_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = tmp_path / "comparison_test_ledger.json"
    observed: dict[str, object] = {}
    original_link = os.link

    def inspecting_link(source, destination, *args, **kwargs):
        assert Path(destination) == ledger
        assert not ledger.exists()
        observed.update(json.loads(Path(source).read_text(encoding="utf-8")))
        return original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(heldout.os, "link", inspecting_link)

    value = claim_heldout_test(ledger, _claim())

    assert observed == value
    assert json.loads(ledger.read_text(encoding="utf-8")) == value
    assert ledger.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob("*.claim.tmp"))


def test_concurrent_claim_has_exactly_one_complete_winner(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "comparison_test_ledger.json"
    barrier = Barrier(2)

    def worker(claim_id: str) -> tuple[str, str]:
        barrier.wait()
        try:
            claim_heldout_test(ledger, _claim(claim_id))
        except RuntimeError:
            return claim_id, "rejected"
        return claim_id, "claimed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(worker, ("claim-a", "claim-b")))

    assert sorted(status for _, status in outcomes) == ["claimed", "rejected"]
    winner = next(
        claim_id for claim_id, status in outcomes if status == "claimed"
    )
    assert json.loads(ledger.read_text(encoding="utf-8"))["claim_id"] == winner


def test_concurrent_completion_is_one_atomic_state_transition(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "comparison_test_ledger.json"
    claim_heldout_test(ledger, _claim())
    barrier = Barrier(2)

    def worker(label: str) -> tuple[str, str]:
        barrier.wait()
        try:
            complete_heldout_test(
                ledger,
                claim_id="claim-1",
                completion={"result_path": label},
            )
        except RuntimeError:
            return label, "rejected"
        return label, "completed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(worker, ("result-a", "result-b")))

    assert sorted(status for _, status in outcomes) == [
        "completed",
        "rejected",
    ]
    winner = next(
        label for label, status in outcomes if status == "completed"
    )
    value = json.loads(ledger.read_text(encoding="utf-8"))
    assert value["status"] == "completed"
    assert value["result_path"] == winner


@pytest.mark.parametrize(
    "field",
    [
        "claim_id",
        "status",
        "access_mode",
        "benchmark",
        "manifest_sha256",
        "experiment_tag",
    ],
)
def test_completion_cannot_replace_claim_fields(
    tmp_path: Path,
    field: str,
) -> None:
    ledger = tmp_path / "comparison_test_ledger.json"
    original = claim_heldout_test(ledger, _claim())

    with pytest.raises(ValueError, match="cannot replace claim fields"):
        complete_heldout_test(
            ledger,
            claim_id="claim-1",
            completion={field: "substituted"},
        )

    assert json.loads(ledger.read_text(encoding="utf-8")) == original


def test_completion_keeps_identity_and_adds_new_result_fields(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "comparison_test_ledger.json"
    original = claim_heldout_test(ledger, _claim())

    completed = complete_heldout_test(
        ledger,
        claim_id="claim-1",
        completion={
            "completed_at": "2026-07-27T00:01:00+00:00",
            "result_path": "/results/test.json",
            "result_file_sha256": "b" * 64,
        },
    )

    for key, value in original.items():
        if key != "status":
            assert completed[key] == value
    assert completed["status"] == "completed"
    assert completed["result_path"] == "/results/test.json"
