import pandas as pd
import pytest
import torch

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.discovery.transitions import build_transition_dataset
from footballq.synthetic.generate import generate_synthetic_tracking


def _transition_inputs(tmp_path, matches=3, dim=6):
    frames = []
    for idx in range(matches):
        frame = generate_synthetic_tracking(
            match_id=f"transition_m{idx}",
            duration_s=5.0,
            fps=5.0,
            seed=idx,
        )
        frame["possession_team_id"] = "home" if idx % 2 == 0 else "away"
        frames.append(frame)
    windows = build_tracking_windows(
        pd.concat(frames, ignore_index=True),
        fps_out=5.0,
        context_seconds=1.0,
        horizon_seconds=1.0,
        stride_seconds=0.2,
    )
    windows_path = save_windows_pt(windows, tmp_path / "windows.pt")
    z = torch.randn(len(windows.match_id), dim, generator=torch.Generator().manual_seed(7))
    embeddings_path = tmp_path / "embeddings.pt"
    torch.save(
        {
            "z": z,
            "match_id": windows.match_id,
            "frame_t": windows.start_frame,
            "source_split": [
                "train" if idx % 3 == 0 else "val" if idx % 3 == 1 else "test"
                for idx in range(len(windows.match_id))
            ],
        },
        embeddings_path,
    )
    return windows, windows_path, embeddings_path


def test_transition_dataset_builds_and_preserves_metadata(tmp_path):
    windows, windows_path, embeddings_path = _transition_inputs(tmp_path)
    data = build_transition_dataset(
        embeddings_path,
        windows_path,
        delta_steps=[1, 2],
        fps=5.0,
    )
    assert data.num_examples > 0
    assert data.examples["z_t"].shape[1] == 6
    assert data.examples["delta_z"].shape == data.examples["z_t"].shape
    assert set(data.examples["match_id"]) == set(windows.match_id)
    assert "possession_team_id" in data.examples["metadata"]
    assert "future_ball_displacement_m" in data.examples["metadata"]
    assert "period" in data.metadata["missing_metadata_fields"]
    assert data.features["normalized_delta_z"].shape == data.examples["delta_z"].shape


def test_transition_dataset_fails_when_windows_miss_match(tmp_path):
    windows, windows_path, embeddings_path = _transition_inputs(tmp_path, matches=3)
    payload = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    payload["match_id"] = [
        "missing_window_match" if value == windows.match_id[0] else value
        for value in payload["match_id"]
    ]
    missing_path = tmp_path / "embeddings_missing.pt"
    torch.save(payload, missing_path)
    with pytest.raises(ValueError, match="Tracking windows do not cover"):
        build_transition_dataset(missing_path, windows_path, delta_steps=[1], fps=5.0)


def test_transition_dataset_sequence_fallback_reports_unavailable_exact_delta(tmp_path):
    _, windows_path, embeddings_path = _transition_inputs(tmp_path, matches=3)
    data = build_transition_dataset(
        embeddings_path,
        windows_path,
        delta_steps=[3],
        fps=5.0,
    )
    diagnostics = data.metadata["pairing_diagnostics"][0]
    assert diagnostics["pairing_mode"] in {"exact_frame_delta", "sequence_offset_fallback"}
    if diagnostics["pairing_mode"] == "sequence_offset_fallback":
        assert "warning" in diagnostics
