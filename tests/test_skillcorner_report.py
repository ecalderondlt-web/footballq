import torch

from footballq.data.windows import FEATURE_NAMES, TrackingWindowTensorData, save_windows_pt
from footballq.io.skillcorner_report import (
    build_skillcorner_availability_report,
    discover_skillcorner_raw_matches,
    horizon_label,
)


def test_skillcorner_raw_match_report(tmp_path):
    raw = tmp_path / "raw" / "skillcorner"
    for match_id in ["1001", "1002"]:
        match_dir = raw / match_id
        match_dir.mkdir(parents=True)
        (match_dir / f"{match_id}_tracking_extrapolated.jsonl").write_text(
            "{}\n",
            encoding="utf-8",
        )
        (match_dir / f"{match_id}_match.json").write_text("{}", encoding="utf-8")
    matches = discover_skillcorner_raw_matches(raw)
    assert [match.match_id for match in matches] == ["1001", "1002"]
    assert all(match.has_tracking for match in matches)
    assert all(match.has_metadata for match in matches)


def test_horizon_labeling_2s_4s_6s():
    assert horizon_label(2.0) == "h2s"
    assert horizon_label(4.0) == "h4s"
    assert horizon_label(6.0) == "h6s"
    assert horizon_label(2.5) == "h2p5s"


def test_availability_report_includes_embedding_alignment(tmp_path):
    raw = tmp_path / "raw" / "skillcorner" / "1001"
    raw.mkdir(parents=True)
    (raw / "1001_tracking_extrapolated.jsonl").write_text("{}\n", encoding="utf-8")
    processed = tmp_path / "processed"
    processed.mkdir()
    embeddings = processed / "embeddings.pt"
    torch.save({"z": torch.randn(1, 4), "match_id": ["1001"], "frame_t": [10]}, embeddings)
    report = build_skillcorner_availability_report(
        raw.parent,
        processed,
        embeddings=embeddings,
        horizons=[2.0, 4.0, 6.0],
    )
    assert report["raw_match_count"] == 1
    assert report["embedding_match_ids"] == ["1001"]
    assert [item["horizon_label"] for item in report["horizons"]] == ["h2s", "h4s", "h6s"]


def test_availability_report_includes_window_period_coverage(tmp_path):
    raw = tmp_path / "raw" / "skillcorner" / "1001"
    raw.mkdir(parents=True)
    (raw / "1001_tracking_extrapolated.jsonl").write_text("{}\n", encoding="utf-8")
    processed = tmp_path / "processed"
    processed.mkdir()
    n = 3
    windows = TrackingWindowTensorData(
        past=torch.zeros((n, 2, 3, len(FEATURE_NAMES))),
        future_xy=torch.zeros((n, 2, 3, 2)),
        past_mask=torch.ones((n, 2, 3), dtype=torch.bool),
        future_mask=torch.ones((n, 2, 3), dtype=torch.bool),
        entity_type=torch.zeros((n, 3), dtype=torch.long),
        team_id=torch.zeros((n, 3), dtype=torch.long),
        match_id=["1001", "1001", "1001"],
        period=[1, 2, 2],
        start_frame=[10, 20, 22],
        feature_names=list(FEATURE_NAMES),
        fps=5.0,
        context_seconds=0.4,
        horizon_seconds=0.4,
        stride_seconds=0.2,
    )
    save_windows_pt(windows, processed / "skillcorner_windows_h2s.pt")
    embeddings = processed / "embeddings.pt"
    torch.save(
        {"z": torch.randn(1, 4), "match_id": ["1001"], "frame_t": [10]},
        embeddings,
    )
    report = build_skillcorner_availability_report(
        raw.parent,
        processed,
        embeddings=embeddings,
        horizons=[2.0],
    )
    horizon = report["horizons"][0]
    assert horizon["window_periods"] == [1, 2]
    assert horizon["window_count_by_period"] == {"1": 1, "2": 2}
    assert horizon["window_count_by_match_period"] == {"1001": {"1": 1, "2": 2}}
    assert horizon["window_start_frame_range_by_match_period"]["1001"]["2"] == {
        "count": 2,
        "min_start_frame": 20,
        "max_start_frame": 22,
    }
    assert horizon["embedding_alignment"]["matching_window_keys"] == 1
