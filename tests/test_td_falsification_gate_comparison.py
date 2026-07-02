import json

from scripts.compare_td_falsification_gates import (
    compare_rows,
    parse_gate_spec,
    rows_from_gate,
)


def test_parse_gate_spec():
    label, path = parse_gate_spec("gap0p5:runs/gate.json")
    assert label == "gap0p5"
    assert str(path) == "runs\\gate.json" or str(path) == "runs/gate.json"


def test_compare_rows_reports_ratio_deltas(tmp_path):
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(
        json.dumps(
            {
                "conditions": {
                    "no_motion_predictor": {
                        "status": "caution",
                        "td_loss_ratio_vs_correct": {"mean": 1.2, "min": 1.0, "max": 1.4},
                        "td_loss_margin_vs_correct": {"mean": 0.1},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            {
                "conditions": {
                    "no_motion_predictor": {
                        "status": "pass",
                        "td_loss_ratio_vs_correct": {"mean": 1.5, "min": 1.3, "max": 1.7},
                        "td_loss_margin_vs_correct": {"mean": 0.2},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    rows = [
        *rows_from_gate("gap0p5", reference),
        *rows_from_gate("gap1p0", candidate),
    ]
    comparison = compare_rows(rows)
    item = comparison["comparisons"]["no_motion_predictor"]["candidates"][0]
    assert item["gate"] == "gap1p0"
    assert item["status"] == "pass"
    assert round(item["ratio_min_delta"], 6) == 0.3
