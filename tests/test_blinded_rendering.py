import csv
import importlib.util
import json
from pathlib import Path

import torch

from footballq.data.normalize import normalize_xy_from_meters
from footballq.data.windows import (
    FEATURE_NAMES,
    TEAM_AWAY,
    TEAM_HOME,
    TrackingWindowTensorData,
    save_windows_pt,
)


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


def test_blinded_renderer_can_select_hidden_low_residual_controls(tmp_path):
    module = _load_renderer()
    rows = [
        {
            "rank_source": "latent_residual_cv",
            "match_id": "m1",
            "period": 1,
            "frame_t": 10,
            "cluster_id": 2,
            "latent_residual_score": 9.0,
        },
        {
            "rank_source": "latent_residual_cv",
            "match_id": "m2",
            "period": 1,
            "frame_t": 20,
            "cluster_id": 3,
            "latent_residual_score": 8.0,
        },
        {
            "rank_source": "low_latent_residual_cv",
            "match_id": "m3",
            "period": 1,
            "frame_t": 30,
            "cluster_id": 2,
            "latent_residual_score": 0.1,
        },
        {
            "rank_source": "low_latent_residual_cv",
            "match_id": "m4",
            "period": 1,
            "frame_t": 40,
            "cluster_id": 3,
            "latent_residual_score": 0.2,
        },
    ]
    selected = module.select_rows_with_low_residual_controls(
        rows,
        positive_rows=2,
        controls_per_positive=1,
        shuffle_seed=None,
    )
    assert [row["positive_control"] for row in selected] == [True, False, True, False]
    assert selected[1]["control_match_reason"] == "same_cluster"
    assert selected[3]["control_group"] == "group_00001"

    annotator, key = module.write_blinded_annotation_files(
        selected,
        tmp_path / "annotator.csv",
        tmp_path / "key.csv",
    )
    annotator_rows = list(csv.DictReader(annotator.open("r", encoding="utf-8")))
    key_rows = list(csv.DictReader(key.open("r", encoding="utf-8")))
    assert "positive_control" not in annotator_rows[0]
    assert "control_group" not in annotator_rows[0]
    assert [row["positive_control"] for row in key_rows] == ["True", "False", "True", "False"]


def test_blinded_renderer_attaches_processed_window_gif(tmp_path):
    module = _load_renderer()
    n_entities = 3
    past = torch.zeros((1, 2, n_entities, len(FEATURE_NAMES)), dtype=torch.float32)
    future_xy = torch.zeros((1, 2, n_entities, 2), dtype=torch.float32)
    past_mask = torch.ones((1, 2, n_entities), dtype=torch.bool)
    future_mask = torch.ones((1, 2, n_entities), dtype=torch.bool)
    team_id = torch.tensor([[0, TEAM_HOME, TEAM_AWAY]], dtype=torch.long)
    entity_type = torch.tensor([[0, 1, 1]], dtype=torch.long)

    past_xy_m = torch.tensor(
        [
            [[52.5, 34.0], [30.0, 20.0], [75.0, 45.0]],
            [[53.0, 34.5], [31.0, 20.5], [74.0, 44.5]],
        ],
        dtype=torch.float32,
    )
    future_xy_m = torch.tensor(
        [
            [[54.0, 35.0], [32.0, 21.0], [73.0, 44.0]],
            [[55.0, 35.5], [33.0, 21.5], [72.0, 43.5]],
        ],
        dtype=torch.float32,
    )
    past[0, :, :, 0:2] = normalize_xy_from_meters(past_xy_m)
    future_xy[0] = normalize_xy_from_meters(future_xy_m)
    windows = TrackingWindowTensorData(
        past=past,
        future_xy=future_xy,
        past_mask=past_mask,
        future_mask=future_mask,
        entity_type=entity_type,
        team_id=team_id,
        match_id=["m1"],
        period=[1],
        start_frame=[10],
        feature_names=list(FEATURE_NAMES),
        fps=5.0,
        context_seconds=0.4,
        horizon_seconds=0.4,
        stride_seconds=0.2,
    )

    rows, stats = module.attach_window_clip_paths(
        [
            {
                "match_id": "m1",
                "period": 1,
                "frame_t": 10,
                "cluster_id": 7,
                "latent_residual_score": 2.5,
            }
        ],
        windows,
        tmp_path / "media",
        fps=2.0,
    )
    assert stats["rendered_clips"] == 1
    assert stats["missing_windows"] == 0
    clip_path = Path(str(rows[0]["clip_path"]))
    assert clip_path.exists()
    assert clip_path.suffix == ".gif"

    annotator, key = module.write_blinded_annotation_files(
        rows,
        tmp_path / "annotator.csv",
        tmp_path / "key.csv",
    )
    annotator_rows = list(csv.DictReader(annotator.open("r", encoding="utf-8")))
    key_rows = list(csv.DictReader(key.open("r", encoding="utf-8")))
    assert "cluster_id" not in annotator_rows[0]
    assert annotator_rows[0]["clip_path"].endswith(".gif")
    assert key_rows[0]["latent_residual_score"] == "2.5"

    rows_reused, stats_reused = module.attach_window_clip_paths(
        [
            {
                "match_id": "m1",
                "period": 1,
                "frame_t": 10,
                "cluster_id": 7,
                "latent_residual_score": 2.5,
            }
        ],
        windows,
        tmp_path / "media",
        fps=2.0,
        reuse_existing=True,
    )
    assert stats_reused["rendered_clips"] == 0
    assert stats_reused["reused_clips"] == 1
    assert rows_reused[0]["clip_path"] == rows[0]["clip_path"]

    manifest = module.write_render_manifest(
        tmp_path / "render_manifest.json",
        examples_csv=tmp_path / "examples.csv",
        windows_path=tmp_path / "windows.pt",
        annotator_csv=annotator,
        key_csv=key,
        rows=rows,
        stats=stats,
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["claim_status"] == "diagnostic_only"
    assert manifest_payload["rows_with_clip_path"] == 1
    assert "cluster_id" not in manifest_payload["annotator_fields"]
    assert "latent_residual_score" in manifest_payload["private_key_fields"]


def test_blinded_renderer_resolves_rows_across_multiple_window_files(tmp_path):
    module = _load_renderer()
    n_entities = 3

    def tiny_windows(match_id: str, period: int, start_frame: int) -> TrackingWindowTensorData:
        past = torch.zeros((1, 2, n_entities, len(FEATURE_NAMES)), dtype=torch.float32)
        future_xy = torch.zeros((1, 2, n_entities, 2), dtype=torch.float32)
        past_mask = torch.ones((1, 2, n_entities), dtype=torch.bool)
        future_mask = torch.ones((1, 2, n_entities), dtype=torch.bool)
        team_id = torch.tensor([[0, TEAM_HOME, TEAM_AWAY]], dtype=torch.long)
        entity_type = torch.tensor([[0, 1, 1]], dtype=torch.long)
        past_xy_m = torch.tensor(
            [
                [[52.5, 34.0], [30.0, 20.0], [75.0, 45.0]],
                [[53.0, 34.5], [31.0, 20.5], [74.0, 44.5]],
            ],
            dtype=torch.float32,
        )
        future_xy_m = torch.tensor(
            [
                [[54.0, 35.0], [32.0, 21.0], [73.0, 44.0]],
                [[55.0, 35.5], [33.0, 21.5], [72.0, 43.5]],
            ],
            dtype=torch.float32,
        )
        past[0, :, :, 0:2] = normalize_xy_from_meters(past_xy_m)
        future_xy[0] = normalize_xy_from_meters(future_xy_m)
        return TrackingWindowTensorData(
            past=past,
            future_xy=future_xy,
            past_mask=past_mask,
            future_mask=future_mask,
            entity_type=entity_type,
            team_id=team_id,
            match_id=[match_id],
            period=[period],
            start_frame=[start_frame],
            feature_names=list(FEATURE_NAMES),
            fps=5.0,
            context_seconds=0.4,
            horizon_seconds=0.4,
            stride_seconds=0.2,
        )

    windows_a = save_windows_pt(tiny_windows("m1", 1, 10), tmp_path / "windows_a.pt")
    windows_b = save_windows_pt(tiny_windows("m2", 2, 20), tmp_path / "windows_b.pt")

    rows, stats = module.attach_window_clip_paths_from_files(
        [
            {"match_id": "m1", "period": 1, "frame_t": 10},
            {"match_id": "m2", "period": 2, "frame_t": 20},
        ],
        [windows_a, windows_b],
        tmp_path / "media_multi",
        fps=2.0,
    )

    assert stats["rendered_clips"] == 2
    assert stats["missing_windows"] == 0
    assert len(stats["windows_paths"]) == 2
    assert all(Path(str(row["clip_path"])).exists() for row in rows)


def test_blinded_renderer_expands_window_globs(tmp_path):
    module = _load_renderer()
    first = tmp_path / "a_windows.pt"
    second = tmp_path / "b_windows.pt"
    first.write_text("placeholder", encoding="utf-8")
    second.write_text("placeholder", encoding="utf-8")

    expanded = module._expand_windows_paths([tmp_path / "*_windows.pt"])

    assert expanded == [first, second]
