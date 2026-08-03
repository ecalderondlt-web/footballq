from __future__ import annotations

from footballq.data.rlcs_touch_windows import Touch
from footballq.data.rlcs_value_windows import (
    CONCEDE,
    NO_GOAL,
    SCORE,
    BoundaryEvent,
    label_ten_touch_outcome,
)


def _touches(count: int = 12) -> list[Touch]:
    return [
        Touch(
            frame_idx=index * 10,
            game_time_s=float(index),
            player_prefix=f"{'blue' if index % 2 == 0 else 'orange'}_player_1",
            player_id=f"p-{index % 2}",
            team="blue" if index % 2 == 0 else "orange",
            ball_position=(0.0, 0.0, 100.0),
            blue_score=0,
            orange_score=0,
        )
        for index in range(count)
    ]


def test_goal_is_labeled_from_current_actor_perspective():
    touches = _touches()
    score = label_ten_touch_outcome(
        touches,
        0,
        [BoundaryEvent(35, 3.5, "goal", "blue")],
    )
    concede = label_ten_touch_outcome(
        touches,
        0,
        [BoundaryEvent(35, 3.5, "goal", "orange")],
    )
    assert score.label == SCORE
    assert concede.label == CONCEDE
    assert score.terminated_by == "goal"


def test_ten_touches_without_boundary_is_no_goal():
    outcome = label_ten_touch_outcome(_touches(), 0, [])
    assert outcome.label == NO_GOAL
    assert outcome.touches_observed == 10
    assert outcome.terminated_by == "ten_touches"


def test_kickoff_stops_horizon_and_censored_tail_is_excluded():
    kickoff = label_ten_touch_outcome(
        _touches(), 0, [BoundaryEvent(25, 2.5, "kickoff", None)]
    )
    censored = label_ten_touch_outcome(_touches(count=5), 3, [])
    assert kickoff.label == NO_GOAL
    assert kickoff.terminated_by == "kickoff"
    assert censored.label is None
    assert censored.eligible is False
