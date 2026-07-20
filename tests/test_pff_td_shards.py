import json

import torch
from test_pff import _record
from test_pff_shards import _split

from footballq.data.pff_td_shards import (
    finalize_pff_td_jepa_manifest,
    prepare_pff_td_jepa_shards,
)
from footballq.io.pff_shards import prepare_pff_dataset_shards


def test_pff_td_shards_cross_boundaries_without_duplicate_samples(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    records = [_record(frame) for frame in range(60)]
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
        frames_per_shard=30,
        hash_source=False,
    )

    manifest = prepare_pff_td_jepa_shards(
        canonical,
        tmp_path / "td",
        split_path,
        match_ids=["10502"],
        context_seconds=0.2,
        delta_seconds=0.1,
        stride_seconds=0.1,
        prediction_gap_seconds=0.1,
        workers=2,
    )

    assert manifest["example_count"] == manifest["unique_sample_id_count"]
    assert len(manifest["shards"]) == 2
    payloads = [
        torch.load(tmp_path / "td" / item["path"], weights_only=False)
        for item in manifest["shards"]
    ]
    sample_ids = [sample_id for payload in payloads for sample_id in payload["sample_id"]]
    assert len(sample_ids) == len(set(sample_ids))
    first_frames = payloads[0]["frame_t"]
    assert first_frames and max(first_frames) <= 29
    finalized = finalize_pff_td_jepa_manifest(
        tmp_path / "td" / "all_available" / "dataset_manifest.json"
    )
    assert finalized["tensor_hashes_complete"]
    assert all(item["tensor_sha256"] for item in finalized["shards"])


def test_pff_td_shards_support_observed_only_control(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    records = [_record(frame) for frame in range(30)]
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
        hash_source=False,
    )

    manifest = prepare_pff_td_jepa_shards(
        canonical,
        tmp_path / "td",
        split_path,
        match_ids=["10502"],
        context_seconds=0.2,
        prediction_gap_seconds=0.1,
        visibility_mode="observed_only",
    )
    payload = torch.load(tmp_path / "td" / manifest["shards"][0]["path"], weights_only=False)

    assert manifest["config"]["visibility_mode"] == "observed_only"
    assert not payload["mask_t"][:, :, 11].any()
