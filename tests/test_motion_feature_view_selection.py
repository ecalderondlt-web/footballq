from scripts.evaluate_motion_feature_view_selection import evaluate_selection


def _invariants(*, passed=True, retention=0.9):
    return {"passed": passed, "retention_fraction": retention}


def test_selection_prefers_lagged_view_when_motion_gate_passes():
    result = evaluate_selection(
        _invariants(),
        _invariants(),
        {"status": "preflight_passed", "blocking_conditions": []},
        position_projection_matches=True,
    )

    assert result["selected_feature_view"] == (
        "jump_segmented_causal_position_difference_0p5s"
    )


def test_selection_falls_back_to_position_only_after_lagged_block():
    result = evaluate_selection(
        _invariants(),
        _invariants(),
        {"status": "blocked", "blocking_conditions": ["acceleration_gap"]},
        position_projection_matches=True,
    )

    assert result["status"] == "position_only_selected_for_future_model_protocol"
    assert result["selected_feature_view"] == "position_only"
    assert not result["model_training_authorized"]


def test_selection_blocks_when_position_integrity_fails():
    result = evaluate_selection(
        _invariants(),
        _invariants(passed=False),
        {"status": "blocked", "blocking_conditions": ["acceleration_gap"]},
        position_projection_matches=True,
    )

    assert result["status"] == "blocked"
    assert result["selected_feature_view"] is None
