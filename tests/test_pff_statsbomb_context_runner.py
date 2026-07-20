from scripts.run_pff_statsbomb_context_residual_v1 import (
    CURVE_STEPS,
    FAMILIES,
    FINAL_EXAMPLES,
    FINAL_STEP,
    SEEDS,
)


def test_context_residual_runner_freezes_expected_scope():
    assert FAMILIES == ("tracking", "raw", "random", "pretrained")
    assert SEEDS == (7, 11, 23)
    assert CURVE_STEPS == (100, 500, 1000, 2000)
    assert FINAL_STEP == 2000
    assert FINAL_EXAMPLES == 64000
    assert len(FAMILIES) * len(SEEDS) == 12
