import json

from scripts.summarize_td_falsification import (
    MATCH_INVARIANT_V1_EXPECTATIONS,
    MATCH_INVARIANT_V1_METRICS,
    parse_summary_spec,
    rows_from_summary,
    summarize_rows,
)


def test_parse_summary_spec():
    seed, path = parse_summary_spec("7:runs/example.json")
    assert seed == "7"
    assert str(path) == "runs\\example.json" or str(path) == "runs/example.json"


def test_summarize_rows_marks_blocking_controls():
    rows = [
        {
            "seed": "7",
            "condition": "correct_temporal_pairing",
            "td_loss_ratio_vs_correct": 1.0,
            "td_loss_margin_vs_correct": 0.0,
        },
        {
            "seed": "7",
            "condition": "shuffled_future_within_batch",
            "td_loss_ratio_vs_correct": 3.0,
            "td_loss_margin_vs_correct": 0.2,
        },
        {
            "seed": "7",
            "condition": "team_swap",
            "td_loss_ratio_vs_correct": 1.01,
            "td_loss_margin_vs_correct": 0.001,
        },
        {
            "seed": "7",
            "condition": "masked_ball",
            "td_loss_ratio_vs_correct": 1.01,
            "td_loss_margin_vs_correct": 0.001,
        },
    ]
    summary = summarize_rows(rows, pass_ratio=1.25, caution_ratio=1.05)
    assert summary["conditions"]["correct_temporal_pairing"]["status"] == "reference"
    assert summary["conditions"]["shuffled_future_within_batch"]["status"] == "pass"
    assert summary["conditions"]["team_swap"]["status"] == "fail"
    assert summary["conditions"]["masked_ball"]["status"] == "fail"
    assert summary["blocking_conditions"] == ["team_swap"]
    assert summary["scientific_claim_status"] == "blocked"


def test_rows_from_summary_can_use_total_loss_gate_metric(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": {
                    "correct_temporal_pairing": {"td_loss": 1.0, "total_loss": 2.0},
                    "team_swap": {"td_loss": 1.0, "total_loss": 4.0},
                }
            }
        ),
        encoding="utf-8",
    )
    rows = rows_from_summary("7", summary_path, metric="total_loss")
    team_swap = [row for row in rows if row["condition"] == "team_swap"][0]
    assert team_swap["gate_metric"] == "total_loss"
    assert team_swap["metric_ratio_vs_correct"] == 2.0


def test_match_invariant_policy_uses_condition_metrics_and_symmetry(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": {
                    "correct_temporal_pairing": {
                        "td_loss": 1.0,
                        "total_loss": 2.0,
                        "ball_dynamic_reconstruction_loss": 3.0,
                    },
                    "masked_ball": {
                        "td_loss": 1.1,
                        "total_loss": 2.1,
                        "ball_dynamic_reconstruction_loss": 6.0,
                    },
                    "no_motion_predictor": {
                        "td_loss": 1.1,
                        "total_loss": 3.0,
                        "ball_dynamic_reconstruction_loss": 3.0,
                    },
                    "team_swap": {
                        "td_loss": 1.05,
                        "total_loss": 5.0,
                        "ball_dynamic_reconstruction_loss": 3.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    rows = rows_from_summary(
        "7",
        summary_path,
        condition_metrics=MATCH_INVARIANT_V1_METRICS,
    )
    summary = summarize_rows(
        rows,
        expectations=MATCH_INVARIANT_V1_EXPECTATIONS,
        nonblocking_conditions=set(),
    )
    assert summary["conditions"]["masked_ball"]["status"] == "pass"
    assert summary["conditions"]["no_motion_predictor"]["status"] == "pass"
    assert summary["conditions"]["team_swap"]["status"] == "pass"
    assert summary["blocking_conditions"] == []
