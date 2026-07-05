import csv
import importlib.util
import json
from pathlib import Path


def _load_analyzer():
    path = Path("scripts/analyze_blinded_annotations.py")
    spec = importlib.util.spec_from_file_location("analyze_blinded_annotations", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _package(
    tmp_path: Path,
    annotations: list[str],
    positive_controls: list[bool],
) -> tuple[Path, Path, Path]:
    media = tmp_path / "media"
    media.mkdir()
    annotator_rows = []
    key_rows = []
    for idx, (annotation, positive_control) in enumerate(
        zip(annotations, positive_controls, strict=True)
    ):
        blind_id = f"blind_{idx:05d}"
        clip_path = media / f"{blind_id}.gif"
        clip_path.write_bytes(b"GIF89a")
        annotator_rows.append(
            {
                "blind_id": blind_id,
                "match_id": "m1",
                "period": 1,
                "frame_t": 10 + idx,
                "clip_path": str(clip_path),
                "annotation": annotation,
            }
        )
        key_rows.append(
            {
                "blind_id": blind_id,
                "cluster_id": idx % 2,
                "latent_residual_score": float(10 - idx),
                "positive_control": positive_control,
                "rank_source": "latent_residual_cv"
                if positive_control
                else "low_latent_residual_cv",
                "control_group": f"group_{idx // 2:05d}",
                "control_match_reason": "positive" if positive_control else "same_cluster",
            }
        )
    annotator = _write_csv(
        tmp_path / "annotator" / "annotations.csv",
        ["blind_id", "match_id", "period", "frame_t", "clip_path", "annotation"],
        annotator_rows,
    )
    key = _write_csv(
        tmp_path / "private" / "annotation_key.csv",
        [
            "blind_id",
            "cluster_id",
            "latent_residual_score",
            "positive_control",
            "rank_source",
            "control_group",
            "control_match_reason",
        ],
        key_rows,
    )
    manifest = tmp_path / "render_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "claim_status": "diagnostic_only",
                "rows": len(annotator_rows),
                "rows_with_clip_path": len(annotator_rows),
                "render_stats": {"missing_windows": 0},
            }
        ),
        encoding="utf-8",
    )
    return annotator, key, manifest


def test_analyze_blinded_annotations_reports_incomplete_for_blank_annotations(tmp_path):
    module = _load_analyzer()
    annotator, key, manifest = _package(tmp_path, ["", ""], [True, False])

    summary = module.analyze_blinded_annotations(
        annotator_csv=annotator,
        key_csv=key,
        manifest_json=manifest,
        positive_labels=["tactical_pattern"],
    )

    assert summary["annotation_status"] == "incomplete"
    assert summary["claim_status"] == "diagnostic_only"
    assert summary["completed_count"] == 0
    assert summary["validation"]["validation_status"] == "passed"


def test_analyze_blinded_annotations_reports_control_enrichment(tmp_path):
    module = _load_analyzer()
    annotator, key, manifest = _package(
        tmp_path,
        ["tactical_pattern", "tactical_pattern", "routine_motion", "routine_motion"],
        [True, True, False, False],
    )

    summary = module.analyze_blinded_annotations(
        annotator_csv=annotator,
        key_csv=key,
        manifest_json=manifest,
        positive_labels=["tactical_pattern"],
    )

    assert summary["annotation_status"] == "analyzed"
    assert summary["completed_count"] == 4
    assert summary["groups"]["positive"]["positive_label_rate"] == 1.0
    assert summary["groups"]["control"]["positive_label_rate"] == 0.0
    assert summary["enrichment"]["risk_difference"] == 1.0
    assert summary["enrichment"]["fisher_greater_pvalue"] == 1 / 6


def test_analyze_blinded_annotations_rejects_uncontrolled_labels(tmp_path):
    module = _load_analyzer()
    annotator, key, manifest = _package(tmp_path, ["maybe_tactical"], [True])

    summary = module.analyze_blinded_annotations(
        annotator_csv=annotator,
        key_csv=key,
        manifest_json=manifest,
    )

    assert summary["annotation_status"] == "invalid_labels"
    assert summary["invalid_labels"] == ["maybe_tactical"]
    assert "controlled vocabulary" in summary["issues"][0]
