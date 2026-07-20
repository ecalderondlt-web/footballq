from scripts.run_pff_4x_tracking_complete_v1 import CURVE_STEPS, SEEDS


def test_complete_tracking_runner_freezes_expected_scope():
    assert SEEDS == (7, 11, 23)
    assert CURVE_STEPS == (100, 250, 500, 1000, 2000, 5000, 10000)
    assert 2 * len(SEEDS) == 6
