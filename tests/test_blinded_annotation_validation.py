import csv
import importlib.util
import json
from pathlib import Path


def _load_validator():
    path = Path("scripts/validate_blinded_annotation_package.py")
    spec = importlib.util.spec_from_file_location("validate_blinded_annotation_package", path)
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


def _valid_package(tmp_path: Path) -> tuple[Path, Path, Path]:
    media = tmp_path / "media"
    media.mkdir()
    clip = media / "blind_00000.gif"
    clip.write_bytes(b"GIF89a")
    annotator = _write_csv(
        tmp_path / "annotator" / "annotations.csv",
        ["blind_id", "match_id", "period", "frame_t", "clip_path", "annotation"],
        [
            {
                "blind_id": "blind_00000",
                "match_id": "m1",
                "period": 1,
                "frame_t": 10,
                "clip_path": str(clip),
                "annotation": "",
            }
        ],
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
        [
            {
                "blind_id": "blind_00000",
                "cluster_id": 2,
                "latent_residual_score": 1.5,
                "positive_control": True,
                "rank_source": "latent_residual_cv",
                "control_group": "group_00000",
                "control_match_reason": "positive",
            }
        ],
    )
    manifest = tmp_path / "render_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "claim_status": "diagnostic_only",
                "rows": 1,
                "rows_with_clip_path": 1,
                "render_stats": {"missing_windows": 0},
            }
        ),
        encoding="utf-8",
    )
    return annotator, key, manifest


def test_validate_blinded_annotation_package_passes_complete_blinded_package(tmp_path):
    module = _load_validator()
    annotator, key, manifest = _valid_package(tmp_path)

    report = module.validate_blinded_annotation_package(
        annotator_csv=annotator,
        key_csv=key,
        manifest_json=manifest,
    )

    assert report["validation_status"] == "passed"
    assert report["row_count"] == 1
    assert report["rows_with_clip_path"] == 1
    assert report["issues"] == []


def test_validate_blinded_annotation_package_rejects_private_field_leak(tmp_path):
    module = _load_validator()
    annotator, key, manifest = _valid_package(tmp_path)
    _write_csv(
        annotator,
        [
            "blind_id",
            "match_id",
            "period",
            "frame_t",
            "clip_path",
            "annotation",
            "cluster_id",
        ],
        [
            {
                "blind_id": "blind_00000",
                "match_id": "m1",
                "period": 1,
                "frame_t": 10,
                "clip_path": str(tmp_path / "media" / "blind_00000.gif"),
                "annotation": "",
                "cluster_id": 2,
            }
        ],
    )

    report = module.validate_blinded_annotation_package(
        annotator_csv=annotator,
        key_csv=key,
        manifest_json=manifest,
    )

    assert report["validation_status"] == "failed"
    assert any("private fields" in issue for issue in report["issues"])


def test_validate_blinded_annotation_package_rejects_missing_clips(tmp_path):
    module = _load_validator()
    annotator, key, manifest = _valid_package(tmp_path)
    _write_csv(
        annotator,
        ["blind_id", "match_id", "period", "frame_t", "clip_path", "annotation"],
        [
            {
                "blind_id": "blind_00000",
                "match_id": "m1",
                "period": 1,
                "frame_t": 10,
                "clip_path": "",
                "annotation": "",
            }
        ],
    )

    report = module.validate_blinded_annotation_package(
        annotator_csv=annotator,
        key_csv=key,
        manifest_json=manifest,
    )

    assert report["validation_status"] == "failed"
    assert any("blank clip_path" in issue for issue in report["issues"])
