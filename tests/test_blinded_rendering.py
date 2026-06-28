import csv
import importlib.util
from pathlib import Path


def _load_renderer():
    path = Path("scripts/render_diagnostic_clips.py")
    spec = importlib.util.spec_from_file_location("render_diagnostic_clips", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_blinded_renderer_separates_annotator_file_from_key(tmp_path):
    module = _load_renderer()
    annotator, key = module.write_blinded_annotation_files(
        [
            {
                "match_id": "m1",
                "period": 1,
                "frame_t": 10,
                "clip_path": "clip.mp4",
                "cluster_id": 7,
                "latent_residual_score": 2.5,
                "positive_control": True,
            }
        ],
        tmp_path / "annotator.csv",
        tmp_path / "key.csv",
    )
    annotator_rows = list(csv.DictReader(annotator.open("r", encoding="utf-8")))
    key_rows = list(csv.DictReader(key.open("r", encoding="utf-8")))
    assert "cluster_id" not in annotator_rows[0]
    assert "latent_residual_score" not in annotator_rows[0]
    assert key_rows[0]["cluster_id"] == "7"
