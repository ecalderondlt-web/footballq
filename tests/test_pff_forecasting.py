import json
from pathlib import Path

import torch

from footballq.data.pff_forecasting import (
    PFFForecastDataset,
    prepare_pff_forecast_targets,
)
from footballq.data.td_jepa_dataset import TDJEPAData
from footballq.repro.manifest import file_sha256
from footballq.repro.splits import load_split_manifest


def _split_manifest(path: Path) -> Path:
    payload = {
        "name": "pff_forecast_test_v1",
        "version": 1,
        "dataset": "pff_fc",
        "protocol": "inductive_match_holdout",
        "train_match_ids": ["m1"],
        "val_match_ids": ["m2"],
        "test_match_ids": ["m3"],
        "all_match_ids": ["m1", "m2", "m3"],
        "expected_count": 3,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _frame_state(frames: torch.Tensor) -> torch.Tensor:
    state = torch.zeros(*frames.shape, 23, 5)
    state[..., 0] = frames.unsqueeze(-1).float() / 10.0
    state[..., 1] = torch.arange(23).view(1, 1, 23).float() / 100.0
    state[..., 2] = torch.tensor([1.0] + [0.0] * 22).view(1, 1, 23)
    state[..., 3] = torch.tensor([0.0] + [1.0] * 11 + [0.0] * 11).view(1, 1, 23)
    state[..., 4] = torch.tensor([0.0] * 12 + [1.0] * 11).view(1, 1, 23)
    return state


def _td_data(match_id: str) -> TDJEPAData:
    context_frames = torch.tensor([[0, 1], [1, 2], [2, 3]])
    target_frames = torch.tensor([[4, 5], [4, 5], [4, 5]])
    state = _frame_state(context_frames)
    target = _frame_state(target_frames)
    mask = torch.ones(3, 2, 23, dtype=torch.bool)
    entity_type = torch.tensor([[0] + [1] * 22] * 3)
    team_id = torch.tensor([[0] + [1] * 11 + [2] * 11] * 3)
    return TDJEPAData(
        state_t=state,
        state_t_plus_delta=target,
        delta_state=target[:, :1],
        mask_t=mask,
        mask_t_plus_delta=mask,
        delta_mask=mask[:, :1],
        entity_type=entity_type,
        team_id=team_id,
        match_id=[match_id] * 3,
        period=[1] * 3,
        frame_t=[0, 1, 2],
        sample_id=[f"{match_id}:1:{value}" for value in range(3)],
        delta_frames=1,
        feature_names=["x_norm", "y_norm", "is_ball", "is_home", "is_away"],
        fps=10.0,
        context_seconds=0.2,
        delta_seconds=0.1,
        stride_seconds=0.1,
        objective_mode="future_nonoverlap_context_only",
        prediction_gap_frames=2,
        feature_view="position_only",
        context_frame_indices=context_frames,
        target_frame_indices=target_frames,
        delta_frame_indices=target_frames[:, :1],
    )


def build_source_manifest(
    tmp_path: Path, *, include_test: bool = False
) -> tuple[Path, Path]:
    split_path = _split_manifest(tmp_path / "split.json")
    split = load_split_manifest(split_path)
    source_root = tmp_path / "source"
    entries = []
    split_matches = [("train", "m1"), ("val", "m2")]
    if include_test:
        split_matches.append(("test", "m3"))
    for split_name, match_id in split_matches:
        relative = Path("observed_only") / split_name / match_id / "td_p1_s0000.pt"
        tensor_path = source_root / relative
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        data = _td_data(match_id)
        torch.save(data.to_dict(), tensor_path)
        entries.append(
            {
                "path": str(relative),
                "match_id": match_id,
                "split": split_name,
                "period": 1,
                "example_count": 3,
                "tensor_sha256": file_sha256(tensor_path),
                "manifest_payload_sha256": "unit-test",
            }
        )
    manifest = {
        "status": "complete",
        "dataset": "pff_fc",
        "split_manifest_sha256": split.sha256,
        "manifest_payload_sha256": "source-unit-test",
        "feature_view": "position_only",
        "tensor_hashes_complete": True,
        "shards": entries,
    }
    manifest_path = source_root / "observed_only" / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, split_path


def test_prepare_forecast_targets_aligns_exact_future_frames(tmp_path):
    source_manifest, split_path = build_source_manifest(tmp_path)
    output = tmp_path / "forecast"
    manifest = prepare_pff_forecast_targets(
        source_manifest,
        output,
        split_path,
        horizons_seconds=(0.1, 0.2),
    )

    assert manifest["included_splits"] == ["train", "val"]
    assert manifest["test_included"] is False
    assert manifest["example_count"] == 6
    dataset = PFFForecastDataset(output / "dataset_manifest.json", "train")
    first = dataset[0]
    assert first["target_frame_indices"].tolist() == [2, 3]
    assert torch.allclose(first["future_xy"][:, 0, 0], torch.tensor([0.2, 0.3]))
    assert bool(first["future_mask"].all())


def test_forecast_data_rejects_test_preparation_and_loading(tmp_path):
    source_manifest, split_path = build_source_manifest(tmp_path, include_test=True)
    output = tmp_path / "forecast"
    try:
        prepare_pff_forecast_targets(
            source_manifest,
            output,
            split_path,
            included_splits=("test",),
        )
    except ValueError as exc:
        assert "only permits train and validation" in str(exc)
    else:
        raise AssertionError("test target preparation should be rejected")

    prepare_pff_forecast_targets(
        source_manifest,
        output,
        split_path,
        horizons_seconds=(0.1, 0.2),
    )
    try:
        PFFForecastDataset(output / "dataset_manifest.json", "test")
    except ValueError as exc:
        assert "explicit confirmatory-test path" in str(exc)
    else:
        raise AssertionError("test dataset loading should be rejected")


def test_confirmatory_forecast_access_requires_complete_test_only_scope(tmp_path):
    source_manifest, split_path = build_source_manifest(tmp_path, include_test=True)
    output = tmp_path / "forecast_test"

    manifest = prepare_pff_forecast_targets(
        source_manifest,
        output,
        split_path,
        horizons_seconds=(0.1, 0.2),
        included_splits=("test",),
        confirmatory_test=True,
    )

    assert manifest["included_splits"] == ["test"]
    assert manifest["test_included"] is True
    assert manifest["access_protocol"] == "confirmatory_test_only_v1"
    dataset = PFFForecastDataset(
        output / "dataset_manifest.json",
        "test",
        allow_confirmatory_test=True,
    )
    assert len(dataset) == 3
    assert dataset[0]["match_id"] == "m3"

    try:
        prepare_pff_forecast_targets(
            source_manifest,
            tmp_path / "selected_test",
            split_path,
            included_splits=("test",),
            match_ids=["m3"],
            confirmatory_test=True,
        )
    except ValueError as exc:
        assert "forbids test-match selection" in str(exc)
    else:
        raise AssertionError("confirmatory test preparation should reject cherry-picking")
