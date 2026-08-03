from __future__ import annotations

from pathlib import Path

import pytest

from footballq.training.eval_rlcs_value import V2TestUnlockError, evaluate_value_bundle
from footballq.training.train_rlcs_value import (
    RLCSValueDataset,
    V2TestSplitLockedError,
)


def test_dataset_refuses_test_before_reading_manifest(tmp_path: Path):
    with pytest.raises(V2TestSplitLockedError):
        RLCSValueDataset(tmp_path / "does-not-exist.json", "test")


def test_ordinary_evaluator_cannot_open_test(tmp_path: Path):
    with pytest.raises(V2TestUnlockError):
        evaluate_value_bundle(
            tmp_path / "missing-config.yaml",
            bundle_path=tmp_path / "missing-bundle.json",
            stage="test",
            output_dir=tmp_path / "out",
        )
