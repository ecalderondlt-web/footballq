import pandas as pd

from footballq.schema import TRACKING_COLUMNS, canonical_tracking_frame, validate_tracking_schema


def test_tracking_schema_validation_accepts_canonical_rows():
    df = canonical_tracking_frame(
        pd.DataFrame(
            [
                {
                    "match_id": "m1",
                    "dataset": "synthetic",
                    "period": 1,
                    "frame_id": 0,
                    "time_s": 0.0,
                    "agent_id": "ball",
                    "agent_type": "ball",
                    "team_id": "ball",
                    "x_m": 52.5,
                    "y_m": 34.0,
                }
            ]
        )
    )

    assert list(df.columns[: len(TRACKING_COLUMNS)]) == TRACKING_COLUMNS
    result = validate_tracking_schema(df)
    assert result.ok
    assert result.missing_columns == []
    assert result.invalid_agent_types == []

