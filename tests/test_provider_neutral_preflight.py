from scripts.evaluate_provider_neutral_preflight import evaluate_preflight_gate


def _report(
    acceleration,
    turn,
    speed,
    acceleration_mean=7.4,
    acceleration_p99=30.6,
):
    return {
        "scope": "train_only",
        "sampling": {
            "real": {"shards": [{"path": "observed_only\\train\\match\\td.pt"}]},
            "synthetic": {"shards": [{"path": "profile\\train\\job\\td.pt"}]},
        },
        "global_gap_ranking": [
            {
                "metric": "player_acceleration_mps2",
                "gap_score": acceleration,
                "synthetic": {
                    "mean": acceleration_mean,
                    "p99": acceleration_p99,
                },
            },
            {"metric": "player_turn_deg", "gap_score": turn},
            {"metric": "player_speed_mps", "gap_score": speed},
        ],
    }


def test_provider_neutral_preflight_passes_frozen_conditions():
    result = evaluate_preflight_gate(
        _report(1.45, 1.10, 0.83),
        _report(0.8, 1.0, 0.85, acceleration_mean=6.0, acceleration_p99=25.0),
        {"passed": True},
    )

    assert result["status"] == "preflight_passed"
    assert not result["blocking_conditions"]


def test_provider_neutral_preflight_reports_motion_and_boundary_blockers():
    candidate = _report(
        1.1,
        1.3,
        1.0,
        acceleration_mean=22.0,
        acceleration_p99=31.0,
    )
    candidate["sampling"]["real"]["shards"][0]["path"] = "observed_only\\val\\td.pt"

    result = evaluate_preflight_gate(
        _report(1.45, 1.10, 0.83),
        candidate,
        {"passed": False},
    )

    assert result["status"] == "blocked"
    assert "train_tensor_invariants" in result["blocking_conditions"]
    assert "player_acceleration_gap_below_one" in result["blocking_conditions"]
    assert "player_acceleration_mean_non_degradation" in result["blocking_conditions"]
    assert "player_acceleration_p99_non_degradation" in result["blocking_conditions"]
    assert "train_only_audit_paths" in result["blocking_conditions"]


def test_subset_preflight_enforces_frozen_retention_floor():
    result = evaluate_preflight_gate(
        _report(1.45, 1.10, 0.83),
        _report(0.8, 1.0, 0.85, acceleration_mean=6.0, acceleration_p99=25.0),
        {"passed": True, "retention_fraction": 0.74},
        minimum_retention_fraction=0.75,
    )

    assert result["status"] == "blocked"
    assert "train_example_retention" in result["blocking_conditions"]
