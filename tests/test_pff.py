import bz2
import json

import pytest

from footballq.data.windows import build_tracking_windows
from footballq.io.pff import (
    PFFAdapter,
    discover_pff_tracking_files,
    pff_xy_to_meters,
)


def _record(frame: int, *, duplicate_geometry: bool = False) -> dict:
    home = [
        {
            "jerseyNum": str(index + 1),
            "confidence": "HIGH",
            "visibility": "VISIBLE" if index < 6 else "ESTIMATED",
            "x": -20.0 + index + frame * 0.1,
            "y": -10.0 + index,
        }
        for index in range(11)
    ]
    away = [
        {
            "jerseyNum": str(index + 1),
            "confidence": "HIGH",
            "visibility": "VISIBLE",
            "x": 20.0 - index - frame * 0.1,
            "y": 10.0 - index,
        }
        for index in range(11)
    ]
    if duplicate_geometry:
        home = [item for player in home for item in (player, {**player, "x": player["x"] + 5})]
        away = [item for player in away for item in (player, {**player, "x": player["x"] + 5})]
    return {
        "version": "4.1.0",
        "gameRefId": 10502,
        "videoTimeMs": frame * 1000 / 29.97,
        "frameNum": frame,
        "period": 1,
        "periodElapsedTime": frame / 29.97,
        "homePlayersSmoothed": home,
        "awayPlayersSmoothed": away,
        "ballsSmoothed": {"visibility": "ESTIMATED", "x": 0.0, "y": 0.0, "z": 0.5},
        "game_event_id": None,
        "possession_event_id": None,
        "game_event": None,
        "possession_event": None,
    }


def test_pff_coordinate_conversion_matches_pitch_corners():
    assert pff_xy_to_meters(-52.5, 34.0) == pytest.approx((0.0, 0.0))
    assert pff_xy_to_meters(52.5, -34.0) == pytest.approx((105.0, 68.0))
    assert pff_xy_to_meters(0.0, 0.0) == pytest.approx((52.5, 34.0))


def test_pff_adapter_deduplicates_frames_and_provider_player_pairs(tmp_path):
    records = [_record(0, duplicate_geometry=True), _record(0), _record(1)]
    path = tmp_path / "10502.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    tracking = PFFAdapter(tmp_path, match_id="10502").load_tracking()

    assert len(tracking) == 46
    assert tracking.groupby("frame_id").size().to_dict() == {0: 23, 1: 23}
    home_one = tracking[(tracking["frame_id"] == 0) & (tracking["agent_id"] == "home_1")]
    assert home_one.iloc[0]["raw_x"] == pytest.approx(-20.0)
    assert bool(home_one.iloc[0]["is_observed"])
    estimated = tracking[(tracking["frame_id"] == 0) & (tracking["agent_id"] == "home_11")]
    assert bool(estimated.iloc[0]["visible"])
    assert not bool(estimated.iloc[0]["is_observed"])


def test_pff_adapter_builds_fixed_windows(tmp_path):
    path = tmp_path / "10502.jsonl"
    path.write_text(
        "\n".join(json.dumps(_record(frame)) for frame in range(10)), encoding="utf-8"
    )
    tracking = PFFAdapter(path, match_id="10502").load_tracking()

    windows = build_tracking_windows(
        tracking,
        fps_out=10.0,
        context_seconds=0.2,
        horizon_seconds=0.2,
        stride_seconds=0.2,
    )

    assert windows.past.shape == (1, 2, 23, 10)
    assert windows.future_xy.shape == (1, 2, 23, 2)
    assert windows.match_id == ["10502"]


def test_pff_discovery_prefers_extracted_file_without_double_counting(tmp_path):
    record = json.dumps(_record(0)) + "\n"
    compressed = tmp_path / "10502.jsonl.bz2"
    with bz2.open(compressed, "wt", encoding="utf-8") as handle:
        handle.write(record)
    extracted_dir = tmp_path / "10502.jsonl"
    extracted_dir.mkdir()
    extracted = extracted_dir / "10502.jsonl"
    extracted.write_text(record, encoding="utf-8")

    discovered = discover_pff_tracking_files(tmp_path)

    assert discovered == {"10502": extracted}
