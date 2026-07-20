import json

import pytest

from scripts.summarize_grf_position_scale_gate import summarize_position_scale_gate


def _metrics(tmp_path, family, seed, total, td, spread=0.2):
    path = tmp_path / family / str(seed) / "metrics_val.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "total_loss": total,
                "td_loss": td,
                "z_online_std_mean": spread,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_position_scale_gate_passes_frozen_primary_criteria(tmp_path):
    seeds = (7, 11, 23)
    scratch = {seed: _metrics(tmp_path, "scratch", seed, 1.0, 0.5) for seed in seeds}
    families = {
        "1x": {seed: _metrics(tmp_path, "1x", seed, 0.97, 0.48) for seed in seeds},
        "1x_replay": {
            seed: _metrics(tmp_path, "1x_replay", seed, 0.96, 0.47) for seed in seeds
        },
        "4x": {seed: _metrics(tmp_path, "4x", seed, 0.93, 0.46) for seed in seeds},
        "8x": {seed: _metrics(tmp_path, "8x", seed, 0.90, 0.45) for seed in seeds},
    }

    summary = summarize_position_scale_gate(scratch, families)

    assert summary["status"] == "controls_passed"
    assert not summary["blocking_conditions"]
    assert summary["criteria"]["eight_x_total_wins_vs_scratch"]["value"] == 3
    assert summary["descriptive_total_loss_order"][0] == "8x"


def test_position_scale_gate_blocks_same_compute_replay_failure(tmp_path):
    seeds = (7, 11, 23)
    scratch = {seed: _metrics(tmp_path, "scratch", seed, 1.0, 0.5) for seed in seeds}
    families = {
        "1x": {seed: _metrics(tmp_path, "1x", seed, 0.97, 0.48) for seed in seeds},
        "1x_replay": {
            seed: _metrics(tmp_path, "1x_replay", seed, 0.90, 0.44) for seed in seeds
        },
        "4x": {seed: _metrics(tmp_path, "4x", seed, 0.93, 0.46) for seed in seeds},
        "8x": {seed: _metrics(tmp_path, "8x", seed, 0.92, 0.45) for seed in seeds},
    }

    summary = summarize_position_scale_gate(scratch, families)

    assert summary["status"] == "blocked"
    assert "eight_x_mean_total_improvement_vs_replay" in summary["blocking_conditions"]
    assert "eight_x_mean_td_no_worse_than_replay" in summary["blocking_conditions"]


def test_position_scale_gate_requires_exact_family_and_seed_sets(tmp_path):
    scratch = {7: _metrics(tmp_path, "scratch", 7, 1.0, 0.5)}
    with pytest.raises(ValueError, match="families mismatch"):
        summarize_position_scale_gate(scratch, {})
