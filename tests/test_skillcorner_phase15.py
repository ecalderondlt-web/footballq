import json

import pytest
import torch

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.io.skillcorner import SkillCornerAdapter


def _write_skillcorner_fixture(raw_dir, frames: int = 50, missing_away_11_frame: int | None = None):
    match_dir = raw_dir / "fixture_match"
    match_dir.mkdir(parents=True)
    path = match_dir / "fixture_tracking_extrapolated.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for frame in range(frames):
            time_s = frame / 10.0
            players = []
            for idx in range(1, 12):
                players.append(
                    {
                        "player_id": f"h{idx}",
                        "number": idx,
                        "team": "home",
                        "x": -42.0 + idx + 0.1 * frame,
                        "y": -25.0 + idx,
                        "is_detected": True,
                    }
                )
            for idx in range(1, 12):
                visible = not (idx == 11 and frame == missing_away_11_frame)
                players.append(
                    {
                        "player_id": f"a{idx}",
                        "number": idx,
                        "team": "away",
                        "x": 42.0 - idx - 0.1 * frame,
                        "y": 25.0 - idx,
                        "is_detected": visible,
                    }
                )
            record = {
                "period": 1,
                "frame": frame,
                "timestamp": time_s,
                "fps": 10,
                "possession_team": "home",
                "ball_data": {"x": 0.2 * frame, "y": 0.0, "is_detected": True},
                "player_data": players,
            }
            handle.write(json.dumps(record) + "\n")
    return path


def test_skillcorner_missing_files_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No SkillCorner tracking JSON/JSONL files found"):
        SkillCornerAdapter(tmp_path / "missing", match_id="missing").load_tracking()


def test_skillcorner_schema_from_sample_or_fixture(tmp_path):
    _write_skillcorner_fixture(tmp_path)
    tracking = SkillCornerAdapter(tmp_path, match_id="fixture").load_tracking()
    expected = {
        "match_id",
        "period",
        "frame_id",
        "time_s",
        "fps",
        "entity_id",
        "entity_type",
        "team_id",
        "player_id",
        "jersey_number",
        "role",
        "x_m",
        "y_m",
        "vx_mps",
        "vy_mps",
        "visible",
        "has_possession",
        "possession_team_id",
        "phase",
        "event_type",
    }
    assert expected.issubset(tracking.columns)
    assert tracking.groupby(["match_id", "period", "frame_id"])["entity_id"].nunique().min() == 23
    ball = tracking[tracking["entity_id"] == "ball"].iloc[0]
    assert ball["x_m"] == pytest.approx(52.5)
    assert ball["y_m"] == pytest.approx(34.0)


def test_skillcorner_windows_shape(tmp_path):
    _write_skillcorner_fixture(tmp_path)
    tracking = SkillCornerAdapter(tmp_path, match_id="fixture").load_tracking()
    windows = build_tracking_windows(
        tracking,
        fps_out=10.0,
        context_seconds=2.0,
        horizon_seconds=2.0,
        stride_seconds=0.2,
    )
    assert len(windows.match_id) > 0
    assert windows.past.shape[1:] == (20, 23, len(windows.feature_names))
    assert windows.future_xy.shape[1:] == (20, 23, 2)
    out = save_windows_pt(windows, tmp_path / "skillcorner_windows.pt")
    payload = torch.load(out, map_location="cpu")
    assert "windows" in payload
    first = payload["windows"][0]
    assert first["past"].shape[1] == 23
    assert first["future_xy"].shape[1] == 23
    assert first["future_xy"].shape[-1] == 2


def test_real_data_or_fixture_preserves_23_entity_shape(tmp_path):
    _write_skillcorner_fixture(tmp_path, missing_away_11_frame=10)
    tracking = SkillCornerAdapter(tmp_path, match_id="fixture").load_tracking()
    windows = build_tracking_windows(
        tracking,
        fps_out=10.0,
        context_seconds=2.0,
        horizon_seconds=2.0,
        stride_seconds=0.2,
    )
    assert windows.past.shape[2] == 23
    assert windows.future_xy.shape[2] == 23
    assert not windows.past_mask.all()
