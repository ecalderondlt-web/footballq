import json

import h5py
import numpy as np
import pytest

from footballq.io.footpass import (
    FOOTPASS_ACTION_CLASSES,
    FOOTPASS_PLAYER_IDS,
    FootpassHalfKey,
    FootpassTacticalStore,
    audit_footpass_tactical_data,
    extract_footpass_lineup_signatures,
    rank_footpass_lineup_matches,
)
from footballq.repro.splits import load_split_manifest


def _row(
    frame: int,
    player_id: int,
    *,
    action_class: int = 0,
    roi: bool = True,
    x: float = 0.25,
) -> list[float]:
    team = 0 if player_id < 200 else 1
    roi_values = [100.0, 200.0, 30.0, 60.0] if roi else [np.nan] * 4
    return [
        float(frame),
        float(player_id),
        float(team),
        float(player_id % 100 or 1),
        1.0 if player_id in {100, 200} else 12.0,
        x,
        0.5,
        0.01,
        -0.02,
        *roi_values,
        float(action_class),
    ]


def _write_source(path, *, match_ids=(1,), include_labels=True):
    with h5py.File(path, "w") as handle:
        for match_id in match_ids:
            for period in (1, 2):
                rows = []
                for frame in range(10, 13):
                    rows.append(
                        _row(
                            frame,
                            100,
                            action_class=2 if frame == 11 else 0,
                            x=-0.01 if frame == 12 else 0.25,
                        )
                    )
                    rows.append(_row(frame, 200, roi=frame != 10, x=0.75))
                data = np.asarray(rows, dtype=np.float32)
                if not include_labels:
                    data = data[:, :13]
                handle.create_dataset(f"game_{match_id}_H{period}", data=data)


def _write_split(path, match_ids):
    payload = {
        "name": "footpass_test_split",
        "version": 1,
        "dataset": "footpass",
        "protocol": "test",
        "train_match_ids": [str(match_ids[0])],
        "val_match_ids": [str(match_ids[1])],
        "test_match_ids": [str(match_ids[2])],
        "all_match_ids": [str(value) for value in match_ids],
        "expected_count": len(match_ids),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_footpass_half_key_round_trip():
    half = FootpassHalfKey.parse("game_42_H2")

    assert half.match_id == "42"
    assert half.period == 2
    assert FootpassHalfKey.from_components("42", 2) == half
    with pytest.raises(ValueError, match="Invalid FOOTPASS"):
        FootpassHalfKey.parse("game_42")


def test_footpass_store_keeps_geometry_labels_and_identity_separate(tmp_path):
    source = tmp_path / "train_tactical_data.h5"
    _write_source(source)

    with FootpassTacticalStore(source) as store:
        window = store.read_window("1", 1, 10, 12)

    assert window.sample_ids == ("1:1:10", "1:1:11", "1:1:12")
    assert window.feature_names == ("x_norm", "y_norm", "vx_norm", "vy_norm")
    assert window.geometry.shape == (3, len(FOOTPASS_PLAYER_IDS), 4)
    assert window.active_mask.sum(axis=1).tolist() == [2, 2, 2]
    assert window.finite_geometry_mask.sum(axis=1).tolist() == [2, 2, 2]
    assert not window.pitch_bounds_mask[2, 0]
    assert window.action_class is not None
    assert window.action_class[1, 0] == 2
    assert window.roi_valid_mask[:, 16].tolist() == [False, True, True]
    assert window.team_index[0, 0] == 0
    assert window.team_index[0, 16] == 1
    assert window.geometry.shape[-1] == 4


def test_footpass_store_supports_unlabelled_test_schema(tmp_path):
    source = tmp_path / "test_tactical_data.h5"
    _write_source(source, include_labels=False)

    with FootpassTacticalStore(source) as store:
        window = store.read_window("1", 1, 10, 12)

    assert window.geometry.shape == (3, len(FOOTPASS_PLAYER_IDS), 4)
    assert window.action_class is None
    assert window.active_mask.sum(axis=1).tolist() == [2, 2, 2]


def test_footpass_audit_matches_split_and_counts_events(tmp_path):
    source = tmp_path / "train_tactical_data.h5"
    split_path = tmp_path / "split.json"
    _write_source(source, match_ids=(1, 2, 3))
    _write_split(split_path, (1, 2, 3))

    report = audit_footpass_tactical_data(
        source,
        split_manifest_path=split_path,
        full_scan=True,
        hash_source=True,
    )

    assert report["match_count"] == 3
    assert report["half_count"] == 6
    assert report["total_unique_frames"] == 18
    assert report["event_rows"] == 6
    assert report["event_class_counts"] == {"pass": 6}
    assert report["frame_player_count_distribution"] == {"2": 18}
    assert report["geometry_nan_rows"] == 0
    assert report["coordinate_outlier_counts"]["x_lt_0"] == 6
    assert report["source"]["sha256"]
    assert report["split_manifest_sha256"]
    assert report["ball_coordinates_present"] is False


def test_frozen_footpass_development_split_is_internal_38_5_5():
    split = load_split_manifest("splits/footpass_train48_development_v1.json")

    assert split.payload["dataset"] == "footpass"
    assert len(split.train_match_ids) == 38
    assert len(split.val_match_ids) == 5
    assert len(split.test_match_ids) == 5
    assert set(split.payload["strata"]["coherent_21_player_suffix"]["all_match_ids"]) == {
        "5",
        "9",
        "42",
        "44",
    }


def test_footpass_action_class_mapping_is_stable():
    assert FOOTPASS_ACTION_CLASSES[1] == "drive"
    assert FOOTPASS_ACTION_CLASSES[2] == "pass"
    assert FOOTPASS_ACTION_CLASSES[7] == "tackle"


def test_footpass_lineup_signatures_and_weighted_matches(tmp_path):
    source = tmp_path / "train_tactical_data.h5"
    _write_source(source, match_ids=(1, 2))

    signatures = extract_footpass_lineup_signatures(source)
    ranked = rank_footpass_lineup_matches(signatures, minimum_overlap=1)

    assert [signature.appearance_id for signature in signatures] == [
        "1:0",
        "1:1",
        "2:0",
        "2:1",
    ]
    assert signatures[0].shirt_numbers == (1,)
    assert signatures[0].shirt_role_pairs == ((1, 1),)
    assert ranked[0]["weighted_jaccard"] == pytest.approx(1.0)
    assert ranked[0]["overlap_shirt_numbers"] == [1]
