import copy

import pytest

from footballq.repro.splits import (
    load_split_manifest,
    split_manifest_sha256,
    validate_split_manifest,
)


def test_split_manifest_validates_and_hashes_stably():
    split = load_split_manifest("splits/skillcorner_10match_inductive_v1.json")
    assert split.name == "skillcorner_10match_inductive_v1"
    assert len(split.all_match_ids) == 10
    assert split.sha256 == split_manifest_sha256(split.payload)
    assert split.sha256 == split_manifest_sha256(copy.deepcopy(split.payload))


def test_split_manifest_rejects_overlap_and_bad_union():
    split = load_split_manifest("splits/skillcorner_10match_inductive_v1.json")
    payload = copy.deepcopy(split.payload)
    payload["test_match_ids"][0] = payload["train_match_ids"][0]
    with pytest.raises(ValueError, match="duplicate|overlap"):
        validate_split_manifest(payload)

    payload = copy.deepcopy(split.payload)
    payload["all_match_ids"] = payload["all_match_ids"][:-1]
    with pytest.raises(ValueError, match="all_match_ids"):
        validate_split_manifest(payload)


def test_split_manifest_requires_minimum_scientific_match_count():
    payload = {
        "name": "tiny",
        "version": 1,
        "dataset": "synthetic",
        "protocol": "inductive",
        "train_match_ids": ["m1"],
        "val_match_ids": [],
        "test_match_ids": [],
        "all_match_ids": ["m1"],
        "expected_count": 1,
    }
    with pytest.raises(ValueError, match="at least 3"):
        validate_split_manifest(payload)


def test_split_manifest_rejects_unknown_dataset_name():
    split = load_split_manifest("splits/skillcorner_10match_inductive_v1.json")
    payload = copy.deepcopy(split.payload)
    payload["dataset"] = "mystery_provider"
    with pytest.raises(ValueError, match="dataset"):
        validate_split_manifest(payload)


def test_pff_world_cup_split_covers_all_64_matches_without_overlap():
    split = load_split_manifest("splits/pff_wc2022_64match_inductive_v1.json")

    assert split.payload["dataset"] == "pff_fc"
    assert len(split.train_match_ids) == 48
    assert len(split.val_match_ids) == 8
    assert len(split.test_match_ids) == 8
    assert len(split.all_match_ids) == 64
