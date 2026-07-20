import json

import pytest

from scripts.summarize_pff_curriculum_gate import summarize_curriculum_gate


def _metrics(path, *, total, td, spread=0.2):
    path.write_text(
        json.dumps({"total_loss": total, "td_loss": td, "z_online_std_mean": spread}),
        encoding="utf-8",
    )
    return path


def test_curriculum_gate_reports_frozen_blockers(tmp_path):
    baseline = {
        7: _metrics(tmp_path / "easy7.jsonl", total=1.0, td=0.1),
        11: _metrics(tmp_path / "easy11.jsonl", total=1.0, td=0.1),
    }
    candidate = {
        7: _metrics(tmp_path / "v2_7.jsonl", total=0.99, td=0.11),
        11: _metrics(tmp_path / "v2_11.jsonl", total=0.98, td=0.11),
    }

    summary = summarize_curriculum_gate(baseline, candidate)

    assert summary["status"] == "blocked"
    assert summary["criteria"]["balanced_v2_total_wins"]["passed"]
    assert not summary["criteria"]["mean_total_relative_improvement"]["passed"]
    assert not summary["criteria"]["mean_td_relative_change"]["passed"]


def test_curriculum_gate_passes_material_candidate_improvement(tmp_path):
    baseline = {7: _metrics(tmp_path / "easy.jsonl", total=1.0, td=0.1)}
    candidate = {7: _metrics(tmp_path / "v2.jsonl", total=0.95, td=0.09)}

    summary = summarize_curriculum_gate(baseline, candidate, min_total_wins=1)

    assert summary["status"] == "controls_passed"


def test_curriculum_gate_requires_matching_seeds(tmp_path):
    baseline = {7: _metrics(tmp_path / "easy.jsonl", total=1.0, td=0.1)}
    candidate = {11: _metrics(tmp_path / "v2.jsonl", total=0.9, td=0.09)}

    with pytest.raises(ValueError, match="seeds must match"):
        summarize_curriculum_gate(baseline, candidate)


def test_curriculum_gate_supports_prespecified_family_labels(tmp_path):
    baseline = {7: _metrics(tmp_path / "natural.jsonl", total=1.0, td=0.1)}
    candidate = {7: _metrics(tmp_path / "sqrt.jsonl", total=0.95, td=0.09)}

    summary = summarize_curriculum_gate(
        baseline,
        candidate,
        baseline_label="natural_v2",
        candidate_label="sqrt_v2",
        min_total_wins=1,
    )

    assert summary["comparison"] == {"baseline": "natural_v2", "candidate": "sqrt_v2"}
    assert summary["criteria"]["sqrt_v2_total_wins"]["passed"]
