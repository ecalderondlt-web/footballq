import json

from test_pff import _record
from test_pff_shards import _split

from footballq.io.pff_quality import summarize_pff_canonical_quality
from footballq.io.pff_shards import prepare_pff_dataset_shards


def test_pff_quality_summary_scans_frame_shapes(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    records = [_record(0), _record(1)]
    records[1]["ballsSmoothed"] = None
    (raw / "10502.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    for match_id in ("10503", "10504"):
        (raw / f"{match_id}.jsonl").write_text(json.dumps(_record(0)) + "\n")
    split_path = tmp_path / "split.json"
    _split(split_path)
    canonical = tmp_path / "canonical"
    prepare_pff_dataset_shards(
        raw,
        canonical,
        split_path,
        match_ids=["10502"],
        hash_source=True,
    )

    report = summarize_pff_canonical_quality(canonical)

    assert report["frame_shape_scan_complete"]
    assert report["frame_shape_counts"] == {
        "home=11,away=11,ball=1": 1,
        "home=11,away=11,ball=0": 1,
    }
    assert report["source_hash_missing_match_ids"] == []
