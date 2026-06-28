import json

import pandas as pd
import pytest
import torch

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.discovery.clustering import cluster_transition_file
from footballq.discovery.transitions import (
    TransitionDatasetData,
    build_transition_dataset,
    save_transition_dataset,
)
from footballq.probes.dataset import build_probe_dataset
from footballq.repro.splits import load_split_manifest
from footballq.synthetic.generate import generate_synthetic_tracking


def _split_manifest(tmp_path, match_ids):
    unique = sorted(set(str(value) for value in match_ids))
    payload = {
        "name": "synthetic_scientific",
        "version": 1,
        "dataset": "synthetic",
        "protocol": "inductive",
        "train_match_ids": [unique[0]],
        "val_match_ids": [unique[1]],
        "test_match_ids": [unique[2]],
        "all_match_ids": unique[:3],
        "expected_count": 3,
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, load_split_manifest(path).sha256


def _windows_and_embedding_payload(tmp_path):
    frames = []
    for idx in range(3):
        frames.append(
            generate_synthetic_tracking(
                match_id=f"scientific_m{idx}",
                duration_s=5.0,
                fps=5.0,
                seed=idx,
            )
        )
    windows = build_tracking_windows(
        pd.concat(frames, ignore_index=True),
        fps_out=5.0,
        context_seconds=1.0,
        horizon_seconds=1.0,
        stride_seconds=0.2,
    )
    windows_path = save_windows_pt(windows, tmp_path / "windows.pt")
    payload = {
        "z": torch.randn(
            len(windows.match_id),
            4,
            generator=torch.Generator().manual_seed(17),
        ),
        "match_id": windows.match_id,
        "period": windows.period,
        "frame_t": windows.start_frame,
        "sample_id": windows.sample_id,
        "source_split": ["train" for _ in windows.match_id],
        "feature_view": "geometry_only",
        "objective_mode": "future_nonoverlap_context_only",
    }
    return windows, windows_path, payload


def test_scientific_probe_requires_explicit_period(tmp_path):
    windows, windows_path, payload = _windows_and_embedding_payload(tmp_path)
    manifest_path, split_hash = _split_manifest(tmp_path, windows.match_id)
    payload.pop("period")
    payload["split_manifest_sha256"] = split_hash
    embeddings_path = tmp_path / "embeddings_missing_period.pt"
    torch.save(payload, embeddings_path)

    with pytest.raises(ValueError, match="missing period"):
        build_probe_dataset(
            embeddings_path,
            windows_path,
            target_names=["future_ball_displacement_m"],
            split_manifest_path=manifest_path,
            scientific_mode=True,
        )


def test_scientific_probe_rejects_split_hash_mismatch(tmp_path):
    windows, windows_path, payload = _windows_and_embedding_payload(tmp_path)
    manifest_path, _ = _split_manifest(tmp_path, windows.match_id)
    payload["split_manifest_sha256"] = "bad_hash"
    embeddings_path = tmp_path / "embeddings_bad_hash.pt"
    torch.save(payload, embeddings_path)

    with pytest.raises(ValueError, match="split_manifest_sha256 mismatch"):
        build_probe_dataset(
            embeddings_path,
            windows_path,
            target_names=["future_ball_displacement_m"],
            split_manifest_path=manifest_path,
            scientific_mode=True,
        )


def test_scientific_transitions_require_explicit_period(tmp_path):
    windows, _, payload = _windows_and_embedding_payload(tmp_path)
    manifest_path, split_hash = _split_manifest(tmp_path, windows.match_id)
    payload.pop("period")
    payload.pop("sample_id")
    payload["split_manifest_sha256"] = split_hash
    embeddings_path = tmp_path / "transition_embeddings_missing_period.pt"
    torch.save(payload, embeddings_path)

    with pytest.raises(ValueError, match="missing period"):
        build_transition_dataset(
            embeddings_path,
            delta_steps=[1],
            split_manifest_path=manifest_path,
            scientific_mode=True,
        )


def test_scientific_discovery_clustering_requires_train_fit(tmp_path):
    delta_z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    data = TransitionDatasetData(
        examples={
            "z_t": torch.zeros_like(delta_z),
            "z_next": delta_z,
            "z_prev": torch.zeros_like(delta_z),
            "has_prev": torch.tensor([False, True, True]),
            "delta_z": delta_z,
            "delta_seconds": torch.ones(3),
            "match_id": ["m0", "m1", "m2"],
            "source_split": ["val", "test", "test"],
        },
        features={
            "normalized_delta_z": delta_z,
            "delta_norm": torch.linalg.vector_norm(delta_z, dim=1),
        },
        metadata={"scientific_mode": True},
    )
    dataset_path = save_transition_dataset(data, tmp_path / "transitions.pt")

    with pytest.raises(ValueError, match="requires at least k train examples"):
        cluster_transition_file(dataset_path, tmp_path / "clusters", k_values=[2])
