import json

from scripts.summarize_probe_incremental import (
    contrast_rows,
    parse_suite_spec,
    rows_from_suite,
    summarize_contrasts,
)


def test_parse_suite_spec():
    seed, path = parse_suite_spec("7:runs/probe_suite/results.json")
    assert seed == "7"
    assert str(path) == "runs\\probe_suite\\results.json" or str(path) == (
        "runs/probe_suite/results.json"
    )


def test_rows_from_suite_reads_match_level(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "eval_test.json").write_text(
        json.dumps(
            {
                "match_level": {
                    "primary_metric": "macro_f1",
                    "summary": {
                        "count": 2,
                        "mean": 0.4,
                        "std": 0.1,
                        "min": 0.3,
                        "max": 0.5,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    suite = tmp_path / "results.json"
    suite.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "target": "future_ball_global_x_bucket",
                        "task_type": "classification",
                        "feature_source": "raw_plus_td_jepa",
                        "probe_type": "linear",
                        "test_macro_f1": 0.5,
                        "num_train": 1,
                        "num_val": 1,
                        "num_test": 1,
                        "run_dir": str(run_dir),
                        "error": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = rows_from_suite("7", suite)
    assert rows[0]["metric_name"] == "macro_f1"
    assert rows[0]["match_level_count"] == 2
    assert rows[0]["match_level_mean"] == 0.4


def test_contrast_rows_signs_regression_improvement():
    rows = [
        {
            "seed": "7",
            "target": "future_ball_displacement_m",
            "task_type": "regression",
            "feature_source": "raw_state_summary",
            "probe_type": "linear",
            "metric_name": "rmse",
            "metric_direction": "lower_is_better",
            "metric_value": 7.5,
        },
        {
            "seed": "7",
            "target": "future_ball_displacement_m",
            "task_type": "regression",
            "feature_source": "raw_plus_td_jepa",
            "probe_type": "linear",
            "metric_name": "rmse",
            "metric_direction": "lower_is_better",
            "metric_value": 7.0,
        },
    ]
    contrasts = contrast_rows(rows)
    assert contrasts[0]["signed_improvement"] == 0.5
    summary = summarize_contrasts(contrasts)
    key = "future_ball_displacement_m|linear|raw_plus_td_jepa_vs_raw"
    assert summary[key]["signed_improvement"]["all_positive"] is True
