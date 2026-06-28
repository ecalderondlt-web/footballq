import pytest
import torch

from footballq.repro.identity import ensure_unique_sample_ids, make_sample_id


def test_period_aware_sample_id_distinguishes_periods():
    assert make_sample_id("m1", 1, 10) == "m1:1:10"
    assert make_sample_id("m1", 1, 10) != make_sample_id("m1", 2, 10)


def test_duplicate_sample_ids_fail():
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        ensure_unique_sample_ids(["m1:1:10", "m1:1:10"], context="unit test")


def test_cross_period_transition_pairs_are_not_created(tmp_path):
    from footballq.discovery.transitions import build_transition_dataset

    embeddings = tmp_path / "embeddings.pt"
    torch.save(
        {
            "z": torch.randn(4, 3),
            "match_id": ["m", "m", "m", "m"],
            "period": [1, 1, 2, 2],
            "frame_t": [0, 1, 0, 1],
            "sample_id": ["m:1:0", "m:1:1", "m:2:0", "m:2:1"],
            "source_split": ["train", "train", "test", "test"],
        },
        embeddings,
    )
    data = build_transition_dataset(embeddings, None, delta_steps=[1], fps=1.0)
    assert set(data.examples["period"]) == {1, 2}
    for frame_t, frame_next in zip(
        data.examples["frame_t"].tolist(),
        data.examples["frame_next"].tolist(),
        strict=True,
    ):
        assert frame_next - frame_t == 1
