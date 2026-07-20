import json

import pytest

from scripts.summarize_pff_transfer_gate import summarize_transfer_gate


def _metrics(path, *, total, td, spread=0.2):
    path.write_text(
        json.dumps(
            {
                "total_loss": total,
                "td_loss": td,
                "z_online_std_mean": spread,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_transfer_gate_reports_prespecified_blockers(tmp_path):
    scratch = {
        7: _metrics(tmp_path / "scratch7.jsonl", total=1.0, td=0.1),
        11: _metrics(tmp_path / "scratch11.jsonl", total=1.0, td=0.1),
    }
    transfer = {
        7: _metrics(tmp_path / "transfer7.jsonl", total=0.99, td=0.11),
        11: _metrics(tmp_path / "transfer11.jsonl", total=0.98, td=0.11),
    }

    summary = summarize_transfer_gate(scratch, transfer)

    assert summary["status"] == "blocked"
    assert summary["criteria"]["transfer_total_wins"]["passed"]
    assert not summary["criteria"]["mean_total_relative_improvement"]["passed"]
    assert not summary["criteria"]["mean_td_relative_change"]["passed"]


def test_transfer_gate_requires_matching_seeds(tmp_path):
    scratch = {7: _metrics(tmp_path / "scratch.jsonl", total=1.0, td=0.1)}
    transfer = {11: _metrics(tmp_path / "transfer.jsonl", total=0.9, td=0.09)}

    with pytest.raises(ValueError, match="seeds must match"):
        summarize_transfer_gate(scratch, transfer)
