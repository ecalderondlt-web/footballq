from __future__ import annotations

import json
from pathlib import Path

import pytest

from footballq.training.eval_matchup import (
    TestUnlockError as UnlockError,
)
from footballq.training.eval_matchup import (
    consume_test_unlock,
    validate_test_unlock,
)
from footballq.training.train_matchup import (
    RLCSDecisionDataset,
)
from footballq.training.train_matchup import (
    TestSplitLockedError as LockedSplitError,
)


def test_training_dataset_rejects_test_before_reading_manifest(tmp_path: Path):
    with pytest.raises(LockedSplitError):
        RLCSDecisionDataset(tmp_path / "does-not-exist.json", "test")


def test_missing_or_incomplete_unlock_is_rejected(tmp_path: Path):
    dataset = tmp_path / "dataset.json"
    split = tmp_path / "split.json"
    unlock = tmp_path / "unlock.json"
    dataset.write_text("{}", encoding="utf-8")
    split.write_text("{}", encoding="utf-8")
    unlock.write_text(json.dumps({"status": "draft"}), encoding="utf-8")
    with pytest.raises(UnlockError, match="protocol"):
        validate_test_unlock(
            unlock,
            dataset_manifest_path=dataset,
            split_manifest_path=split,
        )


def test_unlock_receipt_can_only_be_consumed_once(tmp_path: Path):
    unlock = tmp_path / "unlock.json"
    unlock.write_text('{"nonce":"one"}', encoding="utf-8")
    output = tmp_path / "result"
    receipt = consume_test_unlock(unlock, output_dir=output)
    assert receipt.exists()
    with pytest.raises(UnlockError, match="already consumed"):
        consume_test_unlock(unlock, output_dir=output)
    assert consume_test_unlock(unlock, output_dir=output, resume=True) == receipt


def test_unlock_resume_is_bound_to_original_output_directory(tmp_path: Path):
    unlock = tmp_path / "unlock.json"
    unlock.write_text('{"nonce":"one"}', encoding="utf-8")
    consume_test_unlock(unlock, output_dir=tmp_path / "first")
    with pytest.raises(UnlockError, match="already consumed"):
        consume_test_unlock(unlock, output_dir=tmp_path / "second", resume=True)
