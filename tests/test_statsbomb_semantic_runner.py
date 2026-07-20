from scripts.run_statsbomb_semantic_pretrain_v1 import (
    CURVE_STEPS,
    FINAL_ANCHORED_EVENT_TARGETS,
    FINAL_EVENT_TARGETS,
    FINAL_STEP,
    FINAL_WINDOWS,
    SEEDS,
)


def test_statsbomb_semantic_runner_freezes_expected_scope():
    assert SEEDS == (7, 11, 23)
    assert CURVE_STEPS == (100, 500, 1000, 2500, 5700)
    assert FINAL_STEP == 5700
    assert FINAL_WINDOWS == 93479
    assert FINAL_EVENT_TARGETS == 2991328
    assert FINAL_ANCHORED_EVENT_TARGETS == 278803
    assert 2 * len(SEEDS) == 6
