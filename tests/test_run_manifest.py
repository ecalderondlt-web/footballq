import pytest

from footballq.repro.manifest import build_run_manifest, validate_run_manifest


def test_run_manifest_validates_required_fields(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("experiment: test\n", encoding="utf-8")
    manifest = build_run_manifest(
        command=["footballq", "test"],
        config_path=cfg,
        split_manifest_path="splits/skillcorner_10match_inductive_v1.json",
        evaluation_protocol="inductive",
        feature_view="geometry_only",
        objective_mode="future_nonoverlap_context_only",
        dataset_paths={"td_jepa": "data/processed/example.pt"},
        output_paths={"out": "runs/example"},
        warnings=[],
    )
    validate_run_manifest(manifest)
    assert manifest["split_manifest_sha256"]
    assert manifest["feature_view"] == "geometry_only"


def test_run_manifest_rejects_missing_split_hash():
    with pytest.raises(ValueError, match="split_manifest_sha256"):
        validate_run_manifest(
            {
                "created_at_utc": "2026-06-25T00:00:00+00:00",
                "command": "x",
                "git": {"remote_url": "", "branch": "", "commit": "", "dirty": False},
                "config_path": "c",
                "config_sha256": "h",
                "split_manifest_path": "s",
                "split_manifest_sha256": "",
                "evaluation_protocol": "inductive",
                "feature_view": "geometry_only",
                "objective_mode": "legacy_shifted_overlap",
                "dataset_paths": {},
                "output_paths": {},
                "warnings": [],
                "python": {},
            }
        )
