import importlib.util
from pathlib import Path

import torch

from footballq.data.windows import FEATURE_NAMES, TrackingWindowTensorData, save_windows_pt
from footballq.io.skillcorner_report import SkillCornerRawMatch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_tracking_horizons",
    ROOT / "scripts" / "prepare_tracking_horizons.py",
)
assert SPEC is not None
assert SPEC.loader is not None
prepare_tracking_horizons = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_tracking_horizons)


def _windows(periods: list[int]) -> TrackingWindowTensorData:
    n = len(periods)
    return TrackingWindowTensorData(
        past=torch.zeros((n, 2, 3, len(FEATURE_NAMES))),
        future_xy=torch.zeros((n, 2, 3, 2)),
        past_mask=torch.ones((n, 2, 3), dtype=torch.bool),
        future_mask=torch.ones((n, 2, 3), dtype=torch.bool),
        entity_type=torch.zeros((n, 3), dtype=torch.long),
        team_id=torch.zeros((n, 3), dtype=torch.long),
        match_id=["1001"] * n,
        period=periods,
        start_frame=list(range(n)),
        feature_names=list(FEATURE_NAMES),
        fps=5.0,
        context_seconds=0.4,
        horizon_seconds=0.4,
        stride_seconds=0.2,
    )


def _raw_match() -> SkillCornerRawMatch:
    return SkillCornerRawMatch(
        match_id="1001",
        match_dir="",
        tracking_files=[],
        metadata_files=[],
        event_files=[],
        raw_frame_count_by_period={"1": 100, "2": 100},
    )


def test_resume_cache_guard_rejects_cached_windows_missing_raw_periods(
    tmp_path,
    capsys,
):
    cache_path = tmp_path / "1001_skillcorner_windows_h2s.pt"
    save_windows_pt(_windows([1]), cache_path)

    assert not prepare_tracking_horizons._cache_covers_raw_periods(cache_path, _raw_match())

    output = capsys.readouterr().out
    assert "stale_cache_missing_periods" in output
    assert "missing=2" in output


def test_resume_cache_guard_accepts_cached_windows_covering_raw_periods(tmp_path):
    cache_path = tmp_path / "1001_skillcorner_windows_h2s.pt"
    save_windows_pt(_windows([1, 2]), cache_path)

    assert prepare_tracking_horizons._cache_covers_raw_periods(cache_path, _raw_match())
