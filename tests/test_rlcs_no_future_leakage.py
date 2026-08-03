from __future__ import annotations

import numpy as np
import pytest
from rlcs_test_utils import synthetic_replay

from footballq.data.rlcs_touch_windows import (
    TouchWindowError,
    build_replay_decisions,
    select_past_context,
)


def test_context_selection_never_uses_a_frame_after_touch():
    parsed, _, _, _ = synthetic_replay()
    selection = select_past_context(
        parsed.frames,
        touch_frame_idx=20,
        touch_time_s=2.0,
    )
    assert max(selection.row_indices) == 20
    assert max(selection.observed_times) <= 2.0
    assert all(observed <= requested for observed, requested in zip(
        selection.observed_times, selection.requested_times, strict=True
    ))


def test_future_poison_does_not_change_current_context_features():
    clean, observations, roster, inventory = synthetic_replay()
    poisoned, poisoned_observations, poisoned_roster, poisoned_inventory = synthetic_replay(
        future_poison=True
    )
    clean_row = build_replay_decisions(
        clean,
        inventory=inventory,
        split="train",
        observations=observations,
        roster_ids=roster,
    )[0]
    poisoned_row = build_replay_decisions(
        poisoned,
        inventory=poisoned_inventory,
        split="train",
        observations=poisoned_observations,
        roster_ids=poisoned_roster,
    )[0]
    np.testing.assert_allclose(clean_row["state_flat"], poisoned_row["state_flat"])


def test_window_crossing_goal_or_kickoff_boundary_is_excluded():
    for boundary in ("goal", "kickoff"):
        parsed, observations, roster, inventory = synthetic_replay(boundary_event=boundary)
        rows = build_replay_decisions(
            parsed,
            inventory=inventory,
            split="train",
            observations=observations,
            roster_ids=roster,
            exclude_goal_reset_seconds=0.0,
        )
        assert rows == []


def test_nonofficial_classified_goal_remains_a_hard_boundary():
    parsed, observations, roster, inventory = synthetic_replay(boundary_event="goal")
    parsed.events["official_goal"] = False
    rows = build_replay_decisions(
        parsed,
        inventory=inventory,
        split="train",
        observations=observations,
        roster_ids=roster,
        exclude_goal_reset_seconds=0.0,
    )
    assert rows == []


def test_parser_segment_change_excludes_sample():
    parsed, observations, roster, inventory = synthetic_replay()
    parsed.frames.loc[parsed.frames["observed_frame_number"] >= 25, "stint_number"] = 2
    rows = build_replay_decisions(
        parsed,
        inventory=inventory,
        split="train",
        observations=observations,
        roster_ids=roster,
    )
    assert rows == []


def test_sparse_context_is_rejected_instead_of_using_future_nearest_frame():
    parsed, _, _, _ = synthetic_replay()
    sparse = parsed.frames.iloc[::4].reset_index(drop=True)
    with pytest.raises(TouchWindowError, match="gap|reused"):
        select_past_context(sparse, touch_frame_idx=20, touch_time_s=2.0)
