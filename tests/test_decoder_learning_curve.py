import csv

import pandas as pd
import pytest
import torch

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.decoding.dataset import (
    DecoderDataset,
    build_decoder_dataset,
    decoder_split_diagnostics,
    save_decoder_dataset,
    subset_decoder_dataset,
)
from footballq.decoding.learning_curve import run_decoder_learning_curve
from footballq.decoding.stress import STRESS_SLICE_NAMES, compute_stress_slices
from footballq.synthetic.generate import generate_synthetic_tracking


def _decoder_dataset(tmp_path, matches=3, horizon_seconds=1.0, horizon_steps=4, suffix=""):
    frames = []
    for idx in range(matches):
        frames.append(
            generate_synthetic_tracking(
                match_id=f"decoder_curve_{idx}",
                duration_s=5.0,
                fps=5.0,
                seed=idx,
            )
        )
    windows = build_tracking_windows(
        pd.concat(frames, ignore_index=True),
        fps_out=5.0,
        context_seconds=1.0,
        horizon_seconds=horizon_seconds,
        stride_seconds=0.2,
    )
    windows_path = save_windows_pt(windows, tmp_path / f"windows{suffix}.pt")
    torch.save(
        {
            "z": torch.randn(len(windows.match_id), 8),
            "match_id": windows.match_id,
            "frame_t": windows.start_frame,
            "delta_frames": [1 for _ in windows.match_id],
            "source_split": ["synthetic" for _ in windows.match_id],
            "config": {},
        },
        tmp_path / f"embeddings{suffix}.pt",
    )
    return build_decoder_dataset(
        tmp_path / f"embeddings{suffix}.pt",
        windows_path,
        horizon_steps=horizon_steps,
        context_z_steps=3,
        rollout_steps=2,
    )


def test_decoder_split_diagnostics_are_disjoint_with_three_matches(tmp_path):
    data = _decoder_dataset(tmp_path, matches=3)
    diagnostics = decoder_split_diagnostics(data)
    assert diagnostics["num_matches"] == 3
    assert diagnostics["disjoint_match_split"] is True
    assert diagnostics["smoke_split"] is False


def test_subset_smoke_split_warns_with_one_match(tmp_path):
    data = _decoder_dataset(tmp_path, matches=3)
    first_match = data.splits["train_match_ids"][0]
    indices = [
        idx for idx, match_id in enumerate(data.examples["match_id"]) if match_id == first_match
    ]
    subset = subset_decoder_dataset(data, indices)
    diagnostics = decoder_split_diagnostics(subset)
    assert diagnostics["num_matches"] == 1
    assert diagnostics["smoke_split"] is True
    assert subset.metadata["subset_warnings"]


def test_decoder_learning_curve_writes_results(tmp_path):
    dataset_path = save_decoder_dataset(
        _decoder_dataset(tmp_path, matches=3),
        tmp_path / "decoder.pt",
    )
    result = run_decoder_learning_curve(
        dataset_path,
        tmp_path / "learning_curve",
        match_counts=[1, 3],
        split="test",
        device="cpu",
        epochs=1,
        max_train_batches=1,
        max_eval_batches=1,
        batch_size=4,
        run_root=tmp_path / "runs",
    )
    assert result["results_csv"].exists()
    assert result["stress_results_csv"].exists()
    assert result["summary_json"].exists()
    with result["results_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    models = {row["model"] for row in rows}
    assert "coordinate_constant_velocity" in models
    assert "raw_past_summary_mlp" in models
    assert "context_only_decoder" in models
    assert "z_plus_context_decoder" in models
    assert "residual_context_only_decoder" in models
    assert "residual_z_plus_context_decoder" in models
    assert any(row["smoke_split"] == "True" for row in rows)
    required_columns = {
        "match_ids_train",
        "match_ids_val",
        "match_ids_test",
        "horizon_seconds",
        "model_name",
        "split",
        "slice_name",
        "num_slice_examples",
    }
    assert required_columns.issubset(rows[0])
    for row in rows:
        assert row["finite_metrics"] == "True"
        assert "current_team_centroid_error_m" in row
    assert result["summary"]["subset_diagnostics"]
    assert "raw_match_count" in result["summary"]
    assert "decoder_example_count_by_horizon" in result["summary"]


def test_stress_slice_computation_has_expected_keys_and_keeps_inputs_clean(tmp_path):
    data = _decoder_dataset(tmp_path, matches=3)
    masks, thresholds = compute_stress_slices(data)
    assert set(STRESS_SLICE_NAMES).issubset(masks)
    assert thresholds["slice_counts_all_examples"]["all_windows"] == data.num_examples
    assert all(int(mask.sum().item()) >= 0 for mask in masks.values())
    sample = DecoderDataset(data, mode="residual_future_from_z_past_context")[0]
    assert not any(str(key).startswith(("stress", "slice")) for key in sample)


def test_learning_curve_multi_dataset_filtered_models_and_slices(tmp_path):
    h2 = save_decoder_dataset(
        _decoder_dataset(tmp_path, matches=3, horizon_seconds=1.0, horizon_steps=4, suffix="_h2"),
        tmp_path / "decoder_h2.pt",
    )
    h4 = save_decoder_dataset(
        _decoder_dataset(tmp_path, matches=3, horizon_seconds=1.4, horizon_steps=7, suffix="_h4"),
        tmp_path / "decoder_h4.pt",
    )
    result = run_decoder_learning_curve(
        h2,
        tmp_path / "learning_curve_multi",
        datasets=[h2, h4],
        match_counts=[1, 3],
        models=[
            "coordinate_constant_velocity",
            "last_coordinate_position",
            "residual_context_only",
            "residual_z_plus_context",
        ],
        split="test",
        device="cpu",
        epochs=1,
        max_train_batches=1,
        max_eval_batches=1,
        batch_size=4,
        run_root=tmp_path / "runs_multi",
    )
    with result["results_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    models = {row["model"] for row in rows}
    assert "residual_context_only_decoder" in models
    assert "residual_z_plus_context_decoder" in models
    assert len({row["horizon_steps"] for row in rows}) == 2
    assert any(row["slice_name"] != "all_windows" for row in rows)
    assert any(row["smoke_split"] == "True" for row in rows)
    real_rows = [row for row in rows if row["smoke_split"] == "False"]
    assert real_rows
    assert all(row["disjoint_match_split"] == "True" for row in real_rows)
    for row in rows:
        if row["all_entity_ADE_m"]:
            assert torch.isfinite(torch.tensor(float(row["all_entity_ADE_m"])))
    assert "best_model_per_horizon" in result["summary"]
    assert "best_model_per_stress_slice" in result["summary"]
    assert "residual_z_plus_context_vs_context_only" in result["summary"]
    assert result["stress_results_csv"].exists()
    with result["stress_results_csv"].open("r", encoding="utf-8", newline="") as handle:
        stress_rows = list(csv.DictReader(handle))
    assert stress_rows
    assert {
        "z_plus_context_minus_context_only_ADE_m",
        "z_plus_context_minus_coordinate_cv_ADE_m",
        "fraction_of_test_set",
    }.issubset(stress_rows[0])
    for row in stress_rows:
        assert float(row["num_examples"]) >= 0
        assert torch.isfinite(torch.tensor(float(row["fraction_of_test_set"])))
    summary = result["summary"]
    assert summary["raw_match_count"] == 3
    assert summary["split_disjoint"] is True
    assert "six_second_completed" in summary
    assert "main_limitation" in summary


def test_learning_curve_can_require_real_split(tmp_path):
    dataset_path = save_decoder_dataset(_decoder_dataset(tmp_path, matches=1), tmp_path / "one.pt")
    with pytest.raises(ValueError, match="cannot support"):
        run_decoder_learning_curve(
            dataset_path,
            tmp_path / "learning_curve_require_real",
            match_counts=[1],
            models=["coordinate_constant_velocity"],
            split="test",
            device="cpu",
            require_real_split=True,
            run_root=tmp_path / "runs_require_real",
        )


def test_decoder_learning_curve_all_match_selection(tmp_path):
    dataset_path = save_decoder_dataset(_decoder_dataset(tmp_path, matches=5), tmp_path / "five.pt")
    result = run_decoder_learning_curve(
        dataset_path,
        tmp_path / "learning_curve_all",
        match_counts=[1, 3, "all"],
        models=["coordinate_constant_velocity"],
        split="test",
        device="cpu",
        require_real_split=True,
        run_root=tmp_path / "runs_all",
    )
    counts = {int(item["num_matches"]) for item in result["summary"]["subset_diagnostics"]}
    assert {1, 3, 5}.issubset(counts)
    real = [
        item
        for item in result["summary"]["subset_diagnostics"]
        if int(item["num_matches"]) == 5
    ][0]
    assert real["disjoint_match_split"] is True
    assert len(real["train_match_ids"]) >= 2
