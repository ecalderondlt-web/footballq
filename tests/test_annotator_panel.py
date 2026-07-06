"""Tests for the model-annotator panel tooling."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))

from analyze_annotator_panel import cohen_kappa, fleiss_kappa  # noqa: E402
from run_annotator_panel import parse_labels  # noqa: E402


def test_parse_labels_extracts_only_controlled_vocabulary() -> None:
    raw = (
        "blind_id,annotation\n"
        "blind_001,tactical_pattern\n"
        "blind_002 , routine_motion\n"
        "TextPart(type='text', text='blind_003,tracking_artifact\\nblind_004,ambiguous')\n"
        "blind_005,not_a_label\n"
    )
    labels = parse_labels(raw)
    assert labels == {
        "blind_001": "tactical_pattern",
        "blind_002": "routine_motion",
        "blind_003": "tracking_artifact",
        "blind_004": "ambiguous",
    }


def test_cohen_kappa_perfect_and_chance() -> None:
    a = ["x", "y", "x", "y"]
    assert cohen_kappa(a, list(a)) == 1.0
    b = ["x", "x", "y", "y"]
    assert abs(cohen_kappa(a, b)) < 1.0


def test_fleiss_kappa_perfect_agreement() -> None:
    items = [["x", "x", "x"], ["y", "y", "y"]]
    assert fleiss_kappa(items) == 1.0


def test_per_annotator_metrics_consistency_decisiveness_and_majority() -> None:
    from analyze_annotator_panel import per_annotator_metrics

    # Three annotators, four items. `a` and `b` agree on every item; `c`
    # dissents and hedges. `a`/`b` should be maximally consistent and decisive;
    # `c` should be the least consistent and least decisive (one ambiguous).
    label_matrix = {
        "i0": {"a": "tactical_pattern", "b": "tactical_pattern", "c": "routine_motion"},
        "i1": {"a": "routine_motion", "b": "routine_motion", "c": "tracking_artifact"},
        "i2": {"a": "tracking_artifact", "b": "tracking_artifact", "c": "ambiguous"},
        "i3": {"a": "tactical_pattern", "b": "tactical_pattern", "c": "tactical_pattern"},
    }
    complete_ids = ["i0", "i1", "i2", "i3"]
    names = ["a", "b", "c"]

    metrics = per_annotator_metrics(label_matrix, complete_ids, names)

    # All three keys are present for every annotator.
    for name in names:
        assert set(metrics[name]) >= {
            "mean_pairwise_cohen_kappa",
            "decisiveness",
            "agreement_with_majority",
        }

    # `c` hedges on one of four items, so decisiveness = 3/4.
    assert metrics["c"]["decisiveness"] == 0.75
    # `a` and `b` never hedge, so decisiveness = 1.0.
    assert metrics["a"]["decisiveness"] == 1.0
    assert metrics["b"]["decisiveness"] == 1.0

    # `a` and `b` form the two-vote majority on every item; `c` agrees only on i3.
    assert metrics["a"]["agreement_with_majority"] == 1.0
    assert metrics["b"]["agreement_with_majority"] == 1.0
    assert metrics["c"]["agreement_with_majority"] == 0.25

    # Consistency: a and b are perfectly correlated, so their mean pairwise
    # kappa strictly exceeds c's.
    assert metrics["a"]["mean_pairwise_cohen_kappa"] > metrics["c"]["mean_pairwise_cohen_kappa"]
    assert metrics["b"]["mean_pairwise_cohen_kappa"] > metrics["c"]["mean_pairwise_cohen_kappa"]


def test_panel_analyzer_end_to_end(tmp_path: Path) -> None:
    template = tmp_path / "annotations.csv"
    fieldnames = ["blind_id", "clip_path", "annotation"]
    rows = [
        {"blind_id": f"blind_{i:03d}", "clip_path": f"clips/{i}.gif", "annotation": ""}
        for i in range(4)
    ]
    with template.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labels = {
        "a": ["tactical_pattern", "routine_motion", "routine_motion", "ambiguous"],
        "b": ["tactical_pattern", "routine_motion", "tracking_artifact", "ambiguous"],
    }
    csv_paths = {}
    for name, values in labels.items():
        path = tmp_path / f"{name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row, label in zip(rows, values):
                writer.writerow({**row, "annotation": label})
        csv_paths[name] = path

    out_dir = tmp_path / "panel"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "analyze_annotator_panel.py"),
            "--annotations",
            f"a:{csv_paths['a']}",
            "--annotations",
            f"b:{csv_paths['b']}",
            "--template-csv",
            str(template),
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "panel summary" in result.stdout
    summary = json.loads((out_dir / "panel_summary.json").read_text())
    assert summary["num_complete_items"] == 4
    assert summary["annotators"] == ["a", "b"]
    assert 0.0 < summary["pairwise"]["a|b"]["raw_agreement"] <= 1.0
    assert summary["per_annotator_metrics"]["a"]["decisiveness"] == 0.75
    assert summary["per_annotator_metrics"]["b"]["decisiveness"] == 0.75
    assert summary["per_annotator_metrics"]["a"]["agreement_with_majority"] == 1.0
    with (out_dir / "panel_majority_annotations.csv").open() as handle:
        majority = list(csv.DictReader(handle))
    assert majority[0]["annotation"] == "tactical_pattern"
    assert majority[1]["annotation"] == "routine_motion"
    assert majority[2]["annotation"] == ""
    assert majority[3]["annotation"] == "ambiguous"


def test_contact_sheet_renders_from_gif(tmp_path: Path) -> None:
    gif_path = tmp_path / "clip.gif"
    frames = [Image.new("RGB", (60, 40), color=(10 * i, 20, 30)) for i in range(8)]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100)
    annotator_dir = tmp_path / "annotator"
    annotator_dir.mkdir()
    csv_path = annotator_dir / "annotations.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_id", "clip_path", "annotation"])
        writer.writeheader()
        writer.writerow({"blind_id": "blind_001", "clip_path": str(gif_path), "annotation": ""})
    out_dir = tmp_path / "sheets"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "make_contact_sheets.py"),
            "--annotator-csv",
            str(csv_path),
            "--out-dir",
            str(out_dir),
            "--max-frames",
            "6",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    sheet = out_dir / "blind_001_sheet.png"
    assert sheet.exists()
    manifest = json.loads((out_dir / "sheets_manifest.json").read_text())
    assert manifest["rendered"] == 1
    with Image.open(sheet) as image:
        assert image.width > 0
