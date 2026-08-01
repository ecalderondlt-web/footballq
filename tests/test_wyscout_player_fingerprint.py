from __future__ import annotations

import json

import pandas as pd
import pytest

from footballq.analysis.wyscout_player_fingerprint import (
    FEATURE_NAMES,
    build_player_vectors,
    evaluate_cross_team_retrieval,
    evaluate_support_curve,
)
from scripts.run_wyscout_player_fingerprint_v1 import (
    _start_confirmatory_unseal,
    _verify_file_record,
)


def _rows(
    *,
    player_id: int,
    team_id: int,
    role: int,
    destination_zone: int,
    start_zone: int,
) -> list[dict[str, object]]:
    rows = []
    for match_id in (1, 2):
        for index in range(3):
            rows.append(
                {
                    "player_id": player_id,
                    "team_id": team_id,
                    "role": role,
                    "match_id": match_id,
                    "dateutc": f"2018-01-0{match_id}",
                    "accurate": 1,
                    "start_x": 50.0,
                    "start_y": 40.0,
                    "destination_x": 70.0 + destination_zone,
                    "destination_y": 40.0,
                    "destination_zone": destination_zone,
                    "start_zone": start_zone,
                    "subevent_id": 85,
                    "key_pass": 0,
                    "shot_within_horizon": index == 0,
                }
            )
    return rows


def test_player_vector_has_frozen_75_feature_contract() -> None:
    frame = pd.DataFrame(
        _rows(
            player_id=1,
            team_id=10,
            role=2,
            destination_zone=5,
            start_zone=10,
        )
    )

    vectors = build_player_vectors(
        frame,
        minimum_matches=1,
        minimum_passes=1,
    )

    assert len(FEATURE_NAMES) == 75
    assert vectors.vectors.shape == (1, 75)
    assert vectors.vectors[0, 25 + 5] == 1.0
    assert vectors.vectors[0, 25 + 30 + 10] == 1.0


def test_retrieval_recovers_distinct_players_after_team_change() -> None:
    support = pd.DataFrame(
        [
            *_rows(
                player_id=1,
                team_id=10,
                role=2,
                destination_zone=2,
                start_zone=8,
            ),
            *_rows(
                player_id=2,
                team_id=10,
                role=2,
                destination_zone=20,
                start_zone=15,
            ),
        ]
    )
    query = pd.DataFrame(
        [
            *_rows(
                player_id=1,
                team_id=100,
                role=2,
                destination_zone=2,
                start_zone=8,
            ),
            *_rows(
                player_id=2,
                team_id=100,
                role=2,
                destination_zone=20,
                start_zone=15,
            ),
        ]
    )
    support_vectors = build_player_vectors(
        support,
        minimum_matches=1,
        minimum_passes=1,
    )
    query_vectors = build_player_vectors(
        query,
        minimum_matches=1,
        minimum_passes=1,
    )

    result = evaluate_cross_team_retrieval(
        support_vectors,
        query_vectors,
        feature_view="full_75",
        bootstrap_replicates=100,
        bootstrap_seed=7,
        confidence_level=0.95,
    )

    assert result["changed_team_fraction"] == 1.0
    assert result["ranking"]["same_role"]["top1"] == 1.0
    assert result["same_role_pairwise_auc"]["point"] == 1.0


def test_support_curve_keeps_non_query_players_as_distractors() -> None:
    support = pd.DataFrame(
        [
            *_rows(
                player_id=1,
                team_id=10,
                role=2,
                destination_zone=2,
                start_zone=8,
            ),
            *_rows(
                player_id=2,
                team_id=10,
                role=2,
                destination_zone=20,
                start_zone=15,
            ),
            *_rows(
                player_id=3,
                team_id=20,
                role=2,
                destination_zone=12,
                start_zone=12,
            ),
        ]
    )
    query = pd.DataFrame(
        [
            *_rows(
                player_id=1,
                team_id=100,
                role=2,
                destination_zone=2,
                start_zone=8,
            ),
            *_rows(
                player_id=2,
                team_id=100,
                role=2,
                destination_zone=20,
                start_zone=15,
            ),
        ]
    )
    cohort = {
        "support_min_matches": 1,
        "support_min_passes": 1,
        "query_min_matches": 1,
        "query_min_passes": 1,
        "support_match_caps": [2],
        "main_support_match_cap": 2,
    }
    evaluation = {
        "bootstrap_replicates": 100,
        "bootstrap_seed": 7,
        "confidence_level": 0.95,
    }

    result = evaluate_support_curve(
        support,
        query,
        cohort,
        evaluation,
        feature_view="full_75",
    )

    assert result["eligible_query_players"] == 2
    assert result["eligible_support_candidates"] == 3
    assert result["support_match_caps"]["2"]["query_players"] == 2
    assert result["support_match_caps"]["2"]["support_candidates"] == 3
    assert result["main_vs_first_support"]["common_players"] == 2


def test_frozen_file_record_rejects_a_changed_file(tmp_path) -> None:
    path = tmp_path / "protocol.txt"
    path.write_text("frozen\n", encoding="utf-8")
    record = {
        "path": str(path),
        "sha256": (
            "a6c915e4d76e0c64a625b08e9a42d1432be5a1c5d4b7"
            "41f7b52a4a7e10fb2b89"
        ),
    }

    with pytest.raises(ValueError, match="hash changed"):
        _verify_file_record(record, "protocol")


def test_confirmatory_unseal_sentinel_can_only_be_created_once(
    tmp_path,
) -> None:
    freeze_path = tmp_path / "freeze.json"
    config_path = tmp_path / "config.yaml"
    freeze_path.write_text(json.dumps({"frozen": True}), encoding="utf-8")
    config_path.write_text("frozen: true\n", encoding="utf-8")

    sentinel = _start_confirmatory_unseal(
        output_root=tmp_path / "runs",
        freeze_path=freeze_path,
        config_path=config_path,
    )

    assert sentinel.is_file()
    with pytest.raises(FileExistsError):
        _start_confirmatory_unseal(
            output_root=tmp_path / "runs",
            freeze_path=freeze_path,
            config_path=config_path,
        )
