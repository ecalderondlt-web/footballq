from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.audit_rlcs_identity import estimate_validation_series_power


def test_validation_series_power_is_deterministic_and_uses_no_test_data(tmp_path):
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "val.parquet"
    pq.write_table(
        pa.table(
            {
                "next_touch_entity": [value % 6 for value in range(120)],
                "next_touch_zone": [value % 18 for value in range(120)],
            }
        ),
        train_path,
    )
    series_ids = [f"series-{series}" for series in range(8) for _ in range(18)]
    pq.write_table(
        pa.table(
            {
                "series_id": series_ids,
                "score_diff_actor": [0] * len(series_ids),
                "seconds_remaining": [60.0] * len(series_ids),
                "overtime": [False] * len(series_ids),
                "player_known_mask": [[True] * 6 for _ in series_ids],
                "next_touch_entity": [index % 6 for index in range(len(series_ids))],
                "next_touch_zone": [index % 18 for index in range(len(series_ids))],
            }
        ),
        validation_path,
    )
    first = estimate_validation_series_power(
        train_path,
        validation_path,
        simulation_trials=50,
        sign_flip_permutations=2_000,
        seed=17,
    )
    second = estimate_validation_series_power(
        train_path,
        validation_path,
        simulation_trials=50,
        sign_flip_permutations=2_000,
        seed=17,
    )
    assert first == second
    assert first["validation_series"] == 8
    assert first["validation_primary_samples"] == len(series_ids)
    assert first["estimated_power"] >= 0.80
    assert first["status"] == "passed"
