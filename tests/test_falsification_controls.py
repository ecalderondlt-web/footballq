import torch

from footballq.repro.falsification import apply_td_falsification_control


def _batch():
    return {
        "state_t": torch.randn(3, 4, 23, 5),
        "mask_t": torch.ones(3, 4, 23, dtype=torch.bool),
        "state_t_plus_delta": torch.randn(3, 4, 23, 5),
        "mask_t_plus_delta": torch.ones(3, 4, 23, dtype=torch.bool),
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
