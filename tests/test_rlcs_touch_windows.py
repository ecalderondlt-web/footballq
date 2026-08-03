from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rlcs_test_utils import synthetic_replay

from footballq.data.rlcs_replay import (
    ReplayParseError,
    quality_control_replay,
    repair_score_columns,
)
from footballq.data.rlcs_touch_windows import (
    N_ENTITIES,
    N_FEATURES,
    STATE_MASK_SIZE,
    STATE_SIZE,
    TIME_STEPS,
    TouchWindowError,
    build_replay_decisions,
    encode_decision_identities,
    extract_touches,
    fit_identity_vocabulary,
    next_touch_zone,
    reflect_next_touch_zone,
    reflect_state_x,
    write_decision_parquet,
    write_decision_parquet_batches,
)


def test_builds_exact_schema_and_geometric_relative_target():
    parsed, observations, roster, inventory = synthetic_replay()
    rows = build_replay_decisions(
        parsed,
        inventory=inventory,
        split="train",
        observations=observations,
        roster_ids=roster,
    )
    assert len(rows) == 1
    row = rows[0]
    assert len(row["state_flat"]) == STATE_SIZE == TIME_STEPS * N_ENTITIES * N_FEATURES
    assert len(row["state_mask"]) == STATE_MASK_SIZE
    assert row["next_touch_entity"] in {3, 4, 5}
    assert 0 <= row["next_touch_zone"] < 18
    assert row["retained_possession"] is False
    assert np.isclose(row["next_touch_dt_s"], 1.0)


def test_same_player_contacts_under_point_two_seconds_are_merged():
    parsed, observations, roster, _ = synthetic_replay()
    duplicate = parsed.events.iloc[0].copy()
    duplicate["event_number"] = 2
    duplicate["observed_frame_number"] = 21
    duplicate["frame_number"] = 21
    duplicate["game_time_s_precise"] = 2.1
    parsed.events.loc[1:, "event_number"] += 1
    parsed.events = __import__("pandas").concat(
        [parsed.events.iloc[:1], duplicate.to_frame().T, parsed.events.iloc[1:]],
        ignore_index=True,
    )
    touches = extract_touches(parsed.events, observations, roster)
    assert len(touches) == 2
    assert touches[0].frame_idx == 21


def test_score_repair_deduplicates_synthesized_official_goal_rows():
    events = pd.DataFrame(
        [
            {
                "event_number": 1,
                "event_type": "kickoff",
                "observed_frame_number": 0,
                "event_team": "blue",
                "blue_score": 0,
                "orange_score": 0,
                "goal_number": 1,
                "official_goal": False,
            },
            {
                "event_number": 2,
                "event_type": "goal",
                "observed_frame_number": 100,
                "event_team": "blue",
                "blue_score": 0,
                "orange_score": 0,
                "goal_number": 1,
                "official_goal": False,
            },
            {
                "event_number": 3,
                "event_type": "goal",
                "observed_frame_number": 130,
                "event_team": "blue",
                "blue_score": 2,
                "orange_score": 0,
                "goal_number": np.nan,
                "official_goal": True,
            },
        ]
    )
    repaired = repair_score_columns(
        events,
        expected_blue_score=1,
        expected_orange_score=0,
    )
    assert repaired.iloc[-1]["blue_score"] == 1
    assert repaired.iloc[-1]["orange_score"] == 0
    with pytest.raises(ReplayParseError, match="final score"):
        repair_score_columns(
            events,
            expected_blue_score=2,
            expected_orange_score=0,
        )


def test_actor_orientation_flips_orange_attack_to_positive_y():
    parsed, observations, roster, inventory = synthetic_replay(
        next_player_prefix="blue_player_1"
    )
    parsed.events.iloc[0, parsed.events.columns.get_loc("event_player_1_id")] = "103"
    parsed.events.iloc[0, parsed.events.columns.get_loc("event_player_1_name")] = "Player 103"
    parsed.events.iloc[0, parsed.events.columns.get_loc("event_player_1_team")] = "orange"
    parsed.events.iloc[0, parsed.events.columns.get_loc("event_team")] = "orange"
    rows = build_replay_decisions(
        parsed,
        inventory=inventory,
        split="train",
        observations=observations,
        roster_ids=roster,
    )
    state = np.asarray(rows[0]["state_flat"], dtype=np.float32).reshape(20, 7, 27)
    assert state[-1, 1, 1] < 0  # orange actor at +raw-y is reflected to actor-relative -y
    assert state[-1, 1, 20] == 1
    assert state[-1, 4, 20] == -1


def test_left_right_reflection_is_involution_and_preserves_team_sign():
    parsed, observations, roster, inventory = synthetic_replay()
    row = build_replay_decisions(
        parsed,
        inventory=inventory,
        split="train",
        observations=observations,
        roster_ids=roster,
    )[0]
    state = np.asarray(row["state_flat"], dtype=np.float32).reshape(20, 7, 27)
    reflected_twice = reflect_state_x(reflect_state_x(state))
    np.testing.assert_allclose(reflected_twice, state, atol=1e-5)
    np.testing.assert_array_equal(reflect_state_x(state)[..., 20], state[..., 20])


def test_left_right_reflection_matches_full_rotation_matrix_conjugation():
    state = np.random.default_rng(19).normal(size=(2, 3, 27)).astype(np.float32)
    expected = state.copy()
    expected[..., [0, 3, 12, 21, 24]] *= -1.0
    reflection = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
    for index in np.ndindex(expected.shape[:-1]):
        first = expected[index][6:9]
        second = expected[index][9:12]
        third = np.cross(first, second)
        rotation = np.stack([first, second, third], axis=1)
        transformed = reflection @ rotation @ reflection
        expected[index][6:12] = np.concatenate([transformed[:, 0], transformed[:, 1]])

    np.testing.assert_array_equal(reflect_state_x(state), expected)


def test_zone_grid_has_three_lateral_and_six_longitudinal_bins():
    assert next_touch_zone((-4096, -5120, 0), actor_team="blue") == 0
    assert next_touch_zone((4095, 5119, 0), actor_team="blue") == 17
    assert next_touch_zone((4095, 5119, 0), actor_team="orange") == 2


def test_left_right_reflection_mirrors_only_the_zone_lateral_bin():
    assert reflect_next_touch_zone(0) == 2
    assert reflect_next_touch_zone(1) == 1
    assert reflect_next_touch_zone(2) == 0
    assert reflect_next_touch_zone(15) == 17
    assert reflect_next_touch_zone(17) == 15
    with pytest.raises(ValueError, match=r"\[0, 17\]"):
        reflect_next_touch_zone(18)


def test_quality_control_requires_exact_standard_three_v_three():
    parsed, _, _, _ = synthetic_replay()
    accepted = quality_control_replay(
        parsed.frames,
        parsed.events,
        minimum_duration_seconds=0,
        maximum_duration_seconds=10,
    )
    assert accepted.accepted
    parsed.frames["team_size"] = 2
    rejected = quality_control_replay(
        parsed.frames,
        parsed.events,
        minimum_duration_seconds=0,
        maximum_duration_seconds=10,
    )
    assert not rejected.accepted
    assert "not_standard_3v3" in rejected.reasons


def test_arrow_output_uses_fixed_state_and_identity_lists(tmp_path):
    import pyarrow.parquet as pq

    parsed, observations, roster, inventory = synthetic_replay()
    row = build_replay_decisions(
        parsed,
        inventory=inventory,
        split="train",
        observations=observations,
        roster_ids=roster,
    )[0]
    vocabulary = fit_identity_vocabulary([row["player_ids"]])
    path = write_decision_parquet(
        encode_decision_identities([row], vocabulary), tmp_path / "train.parquet"
    )
    table = pq.read_table(path)
    assert table.num_rows == 1
    assert table.schema.metadata[b"state_shape"] == b"20,7,27"
    assert table.schema.field("state_flat").type.list_size == 3780
    assert table.schema.field("player_identity_idx").type.list_size == 6


def test_batched_arrow_writer_is_atomic_and_detects_cross_batch_duplicates(tmp_path):
    import pyarrow.parquet as pq

    parsed, observations, roster, inventory = synthetic_replay()
    row = build_replay_decisions(
        parsed,
        inventory=inventory,
        split="train",
        observations=observations,
        roster_ids=roster,
    )[0]
    vocabulary = fit_identity_vocabulary([row["player_ids"]])
    encoded = encode_decision_identities([row], vocabulary)[0]
    second = dict(encoded)
    second["sample_id"] = f"{encoded['sample_id']}:second"
    path = write_decision_parquet_batches(
        [[encoded], [second]], tmp_path / "streamed.parquet", max_rows_per_group=1
    )
    assert pq.read_metadata(path).num_rows == 2
    with pytest.raises(TouchWindowError, match="Duplicate sample_id"):
        write_decision_parquet_batches(
            [[encoded], [encoded]], tmp_path / "duplicate.parquet", max_rows_per_group=1
        )
    assert not (tmp_path / "duplicate.parquet.tmp").exists()
