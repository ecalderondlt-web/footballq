from pathlib import Path

import torch

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
