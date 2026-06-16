import pandas as pd
import pytest

from footballq.io.metrica import flatten_metrica_columns, metrica_tracking_wide_to_long


def test_metrica_wide_to_long_conversion_with_fake_multirow_header():
    columns = pd.MultiIndex.from_tuples(
        [
            ("Period", ""),
            ("Frame", ""),
            ("Time [s]", ""),
            ("Player11", "x"),
            ("Unnamed: 4_level_0", "y"),
            ("Ball", "x"),
            ("Unnamed: 6_level_0", "y"),
        ]
    )
    wide = pd.DataFrame(
        [
            [1, 0, 0.0, 0.10, 0.20, 0.50, 0.50],
            [1, 1, 0.1, 0.11, 0.20, 0.51, 0.50],
        ],
        columns=columns,
    )
    wide.columns = flatten_metrica_columns(wide.columns)

    long = metrica_tracking_wide_to_long(wide, team="home", match_id="m1", source_file="fake.csv")

    assert set(long["agent_id"]) == {"home_player11", "ball"}
    player = long[(long["agent_id"] == "home_player11") & (long["frame_id"] == 0)].iloc[0]
    assert player["x_m"] == pytest.approx(10.5)
    assert player["y_m"] == pytest.approx(13.6)
    ball = long[(long["agent_id"] == "ball") & (long["frame_id"] == 0)].iloc[0]
    assert ball["team_id"] == "ball"
