import pandas as pd

from footballq.processing.coordinates import to_meters_from_normalized, validate_pitch_bounds


def test_normalized_coordinate_conversion():
    x_m, y_m = to_meters_from_normalized(0.5, 0.25)
    assert x_m == 52.5
    assert y_m == 17.0


def test_quality_bounds_report_flags_out_of_bounds_coordinates():
    df = pd.DataFrame({"x_m": [0.0, 106.0], "y_m": [34.0, 12.0]})
    report = validate_pitch_bounds(df)
    assert not report["ok"]
    assert report["out_of_bounds_rows"] == 1

