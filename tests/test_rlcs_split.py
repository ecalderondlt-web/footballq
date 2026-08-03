from __future__ import annotations

import json
from pathlib import Path

import pytest

from footballq.data.rlcs_replay import (
    freeze_chronological_split_manifest,
    load_frozen_rlcs_split,
    split_for_inventory_row,
)
from footballq.repro.splits import load_split_manifest


def inventory_row(replay_id: str, split: int, region: str, regional: int) -> dict[str, object]:
    return {
        "replay_id": replay_id,
        "split_number": split,
        "region": region,
        "regional_number": regional,
        "event_time_utc": f"2025-0{split}-0{regional}T00:00:00Z",
    }


def test_frozen_chronological_split_assigns_exact_protocol(tmp_path: Path):
    template = json.loads(Path("splits/rlcs_2025_chronological_v1.json").read_text())
    source = tmp_path / "template.json"
    source.write_text(json.dumps(template), encoding="utf-8")
    records = [
        inventory_row("train-eu", 1, "EU", 1),
        inventory_row("train-na", 1, "NA", 3),
        inventory_row("val-eu", 2, "EU", 1),
        inventory_row("val-na", 2, "NA", 1),
        inventory_row("test-eu-r2", 2, "EU", 2),
        inventory_row("test-eu-r3", 2, "EU", 3),
        inventory_row("test-na-r2", 2, "NA", 2),
    ]
    destination = freeze_chronological_split_manifest(
        records,
        template_path=source,
        output_path=tmp_path / "frozen.json",
        inventory_sha256="abc",
    )
    payload = load_frozen_rlcs_split(destination)
    assert set(payload["train_match_ids"]) == {"train-eu", "train-na"}
    assert set(payload["val_match_ids"]) == {"val-eu", "val-na"}
    assert set(payload["test_match_ids"]) == {
        "test-eu-r2",
        "test-eu-r3",
        "test-na-r2",
    }
    generic = load_split_manifest(destination)
    assert len(generic.all_match_ids) == 7


def test_checked_in_manifest_is_frozen_after_live_inventory():
    payload = load_frozen_rlcs_split("splits/rlcs_2025_chronological_v1.json")
    assert payload["expected_count"] == 1595
    assert len(payload["train_match_ids"]) == 867
    assert len(payload["val_match_ids"]) == 246
    assert len(payload["test_match_ids"]) == 482
    assert len(str(payload["inventory_sha256"])) == 64


def test_out_of_protocol_inventory_row_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        split_for_inventory_row(inventory_row("major", 2, "EU", 4))


def test_split_assignment_is_region_symmetric():
    for region in ("EU", "NA"):
        assert split_for_inventory_row(inventory_row("a", 1, region, 3)) == "train"
        assert split_for_inventory_row(inventory_row("b", 2, region, 1)) == "val"
        assert split_for_inventory_row(inventory_row("c", 2, region, 2)) == "test"
