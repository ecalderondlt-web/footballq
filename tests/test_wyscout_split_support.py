from footballq.repro.splits import load_split_manifest, validate_split_manifest


def test_wyscout_is_allowed_for_immutable_split_manifests(tmp_path) -> None:
    payload = {
        "name": "wyscout_fixture",
        "version": 1,
        "dataset": "wyscout",
        "protocol": "fixture",
        "train_match_ids": ["1"],
        "val_match_ids": ["2"],
        "test_match_ids": ["3"],
        "all_match_ids": ["1", "2", "3"],
        "expected_count": 3,
    }

    validate_split_manifest(payload)

    path = tmp_path / "split.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    assert load_split_manifest(path).payload["dataset"] == "wyscout"
