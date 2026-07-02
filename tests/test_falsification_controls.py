import torch

from footballq.repro.falsification import apply_td_falsification_control


def _batch():
    return {
        "state_t": torch.randn(3, 4, 23, 5),
        "mask_t": torch.ones(3, 4, 23, dtype=torch.bool),
        "state_t_plus_delta": torch.randn(3, 4, 23, 5),
        "mask_t_plus_delta": torch.ones(3, 4, 23, dtype=torch.bool),
        "match_id": ["m0", "m1", "m0"],
    }


def test_player_slot_permutation_preserves_temporal_continuity():
    controlled = apply_td_falsification_control(
        _batch(),
        "consistent_player_slot_permutation",
        seed=7,
    )
    perm = controlled["control_permutation"]
    assert perm[0].item() == 0
    assert sorted(perm.tolist()) == list(range(23))
    assert controlled["state_t"].shape == (3, 4, 23, 5)


def test_shuffled_future_records_condition():
    controlled = apply_td_falsification_control(_batch(), "shuffled_future_within_batch")
    assert controlled["control_condition"] == "shuffled_future_within_batch"
    assert "control_permutation" in controlled


def test_wrong_match_future_uses_other_match_ids():
    batch = _batch()
    controlled = apply_td_falsification_control(batch, "future_from_another_match", seed=0)
    perm = controlled["control_permutation"].tolist()
    for row_idx, source_idx in enumerate(perm):
        assert batch["match_id"][row_idx] != batch["match_id"][source_idx]


def test_team_label_swap_flips_channels_without_slot_permutation():
    batch = _batch()
    batch["state_t"] = torch.zeros(3, 4, 23, 5)
    batch["state_t"][..., 3] = 1.0
    controlled = apply_td_falsification_control(
        batch,
        "team_label_swap",
        feature_names=["x_norm", "y_norm", "vx_norm", "is_home", "is_away"],
    )
    assert torch.all(controlled["state_t"][..., 3] == 0.0)
    assert torch.all(controlled["state_t"][..., 4] == 1.0)
    assert "control_permutation" not in controlled


def test_target_player_slot_permutation_changes_only_target():
    batch = _batch()
    before_context = batch["state_t"].clone()
    controlled = apply_td_falsification_control(
        batch,
        "target_consistent_player_slot_permutation",
        seed=7,
    )
    assert torch.equal(controlled["state_t"], before_context)
    assert "control_permutation" in controlled
    assert controlled["state_t_plus_delta"].shape == batch["state_t_plus_delta"].shape
