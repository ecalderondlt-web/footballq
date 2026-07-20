from scripts.run_grf_position_scale_v1 import SEEDS, SYNTHETIC_CONFIGS


def test_frozen_position_scale_runner_has_expected_run_count():
    synthetic_runs = len(SYNTHETIC_CONFIGS) * len(SEEDS)
    pff_runs = (1 + len(SYNTHETIC_CONFIGS)) * len(SEEDS)

    assert tuple(SEEDS) == (7, 11, 23)
    assert list(SYNTHETIC_CONFIGS) == ["1x", "1x_replay", "4x", "8x"]
    assert synthetic_runs == 12
    assert pff_runs == 15
