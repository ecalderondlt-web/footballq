import json
import sys
from pathlib import Path

import torch
from test_pff import _record
from test_pff_shards import _split

from footballq.data.pff_td_shards import (
    finalize_pff_td_jepa_manifest,
    prepare_pff_td_jepa_shards,
)
from footballq.data.sharded_td_dataset import (
    ShardedTDJEPADataset,
    ShardGroupedSampler,
    ShardTemperatureSampler,
    temperature_shard_allocations,
)
from footballq.data.td_jepa_dataset import load_td_jepa_data
from footballq.data.td_jepa_projection import project_td_jepa_feature_view
from footballq.io.pff_shards import prepare_pff_dataset_shards
from footballq.training.eval_td_jepa import evaluate_td_checkpoint
from footballq.training.train_td_jepa import train_td_jepa_from_config
from scripts.run_td_falsification_controls import main as run_falsification_controls


def _manifest(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for match_id in ("10502", "10503", "10504"):
        records = [{**_record(frame), "gameRefId": int(match_id)} for frame in range(30)]
        (raw / f"{match_id}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records), encoding="utf-8"
        )
    split_path = tmp_path / "split.json"
    _split(split_path)
    canonical = tmp_path / "canonical"
    prepare_pff_dataset_shards(raw, canonical, split_path, frames_per_shard=15, hash_source=False)
    out = tmp_path / "td"
    prepare_pff_td_jepa_shards(
        canonical,
        out,
        split_path,
        context_seconds=0.2,
        delta_seconds=0.1,
        stride_seconds=0.1,
        prediction_gap_seconds=0.1,
    )
    manifest_path = out / "all_available" / "dataset_manifest.json"
    finalize_pff_td_jepa_manifest(manifest_path)
    return manifest_path


def test_sharded_td_dataset_loads_all_examples_lazily(tmp_path):
    manifest_path = _manifest(tmp_path)
    dataset = ShardedTDJEPADataset(manifest_path, "train", verify_hashes_on_load=True)

    assert len(dataset) > 0
    assert dataset[0]["match_id"] == "10502"
    assert dataset[-1]["sample_id"]
    assert len(dataset._cache) == 1


def test_shard_grouped_sampler_is_complete_and_seeded(tmp_path):
    dataset = ShardedTDJEPADataset(_manifest(tmp_path), "train")
    first = list(ShardGroupedSampler(dataset, shuffle=True, seed=7))
    second = list(ShardGroupedSampler(dataset, shuffle=True, seed=7))

    assert sorted(first) == list(range(len(dataset)))
    assert first == second
    assert torch.is_tensor(dataset[0]["state_t"])


def test_temperature_shard_allocations_flatten_source_imbalance():
    allocations = temperature_shard_allocations(
        [100, 25, 1],
        num_samples=160,
        temperature=0.5,
    )

    assert allocations == [100, 50, 10]
    assert sum(allocations) == 160


def test_shard_temperature_sampler_is_fixed_length_grouped_and_seeded(tmp_path):
    dataset = ShardedTDJEPADataset(_manifest(tmp_path), "train")
    first_sampler = ShardTemperatureSampler(
        dataset,
        num_samples=25,
        temperature=0.5,
        seed=7,
    )
    second_sampler = ShardTemperatureSampler(
        dataset,
        num_samples=25,
        temperature=0.5,
        seed=7,
    )
    first = list(first_sampler)
    second = list(second_sampler)

    assert len(first) == 25
    assert first == second
    assert all(0 <= index < len(dataset) for index in first)
    assert sum(first_sampler.allocations) == 25


def test_sharded_td_manifest_trains_and_evaluates_without_combining_files(tmp_path):
    manifest_path = _manifest(tmp_path)
    cfg = {
        "experiment": "pff_sharded_smoke",
        "seed": 7,
        "data": {"path": str(manifest_path), "source": "pff_fc"},
        "split": {"manifest_path": str(tmp_path / "split.json")},
        "model": {
            "z_dim": 8,
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "dropout": 0.0,
            "motion_hidden_dim": 16,
        },
        "ema": {"momentum": 0.9},
        "loss": {"variance_weight": 0.05, "variance_threshold": 0.2},
        "training": {
            "batch_size": 4,
            "max_epochs": 1,
            "max_train_batches": 1,
            "validation_curve_steps": [1],
            "validation_curve_max_batches": 1,
            "num_workers": 0,
            "device": "cpu",
            "run_root": str(tmp_path / "runs"),
        },
    }

    result = train_td_jepa_from_config(cfg)
    metrics = evaluate_td_checkpoint(result["latest_checkpoint"], split="test", device="cpu")
    checkpoint = torch.load(result["latest_checkpoint"], weights_only=False)

    assert checkpoint["split_indices"]["train"]["mode"] == "sharded_manifest"
    assert checkpoint["split_indices"]["test"]["num_examples"] > 0
    curve = json.loads(
        (result["run_dir"] / "metrics_val_curve.jsonl").read_text(encoding="utf-8")
    )
    assert curve["step"] == 1
    assert torch.isfinite(torch.tensor(metrics["metrics"]["total_loss"]))


def test_feature_projection_reads_only_included_shard_splits(tmp_path, monkeypatch):
    manifest_path = _manifest(tmp_path)
    loaded_paths: list[Path] = []

    def tracked_load(path):
        loaded_paths.append(Path(path))
        return load_td_jepa_data(path)

    monkeypatch.setattr("footballq.data.td_jepa_projection.load_td_jepa_data", tracked_load)
    projected_path = project_td_jepa_feature_view(
        manifest_path,
        tmp_path / "projected",
        target_feature_view="position_only",
        included_splits={"train", "val"},
    )
    projected = json.loads(projected_path.read_text(encoding="utf-8"))

    assert projected["included_splits"] == ["train", "val"]
    assert projected["feature_names"] == [
        "x_norm",
        "y_norm",
        "is_ball",
        "is_home",
        "is_away",
    ]
    assert {shard["split"] for shard in projected["shards"]} == {"train", "val"}
    assert projected["excluded_source_shard_counts"]["test"] > 0
    assert loaded_paths
    assert all("test" not in path.parts for path in loaded_paths)


def test_train_only_manifest_can_disable_validation_and_embedding_access(tmp_path):
    source_manifest = _manifest(tmp_path)
    manifest_path = project_td_jepa_feature_view(
        source_manifest,
        tmp_path / "train_only",
        target_feature_view="position_only",
        included_splits={"train"},
    )
    cfg = {
        "experiment": "train_only_sharded_smoke",
        "seed": 7,
        "data": {"path": str(manifest_path), "source": "pff_fc"},
        "split": {"manifest_path": str(tmp_path / "split.json")},
        "model": {
            "z_dim": 8,
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "dropout": 0.0,
            "motion_hidden_dim": 16,
        },
        "ema": {"momentum": 0.9},
        "loss": {"variance_weight": 0.05, "variance_threshold": 0.2},
        "training": {
            "batch_size": 4,
            "max_epochs": 1,
            "max_train_batches": 1,
            "num_workers": 0,
            "device": "cpu",
            "run_root": str(tmp_path / "runs"),
            "validation_split": None,
        },
    }

    result = train_td_jepa_from_config(cfg)
    run_dir = result["run_dir"]
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert result["best_metric"] is None
    assert result["latest_checkpoint"].exists()
    assert result["best_checkpoint"].exists()
    assert not (run_dir / "metrics_val.jsonl").exists()
    assert not (run_dir / "embeddings_sample.pt").exists()
    assert run_manifest["data_access"] == {
        "loaded_tensor_splits": ["train"],
        "validation_split": None,
        "embedding_sample_split": None,
    }


def test_sharded_td_manifest_runs_falsification_controls(tmp_path, monkeypatch):
    manifest_path = _manifest(tmp_path)
    cfg = {
        "experiment": "pff_sharded_falsification_smoke",
        "seed": 7,
        "data": {"path": str(manifest_path), "source": "pff_fc"},
        "split": {"manifest_path": str(tmp_path / "split.json")},
        "model": {
            "z_dim": 8,
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "dropout": 0.0,
            "motion_hidden_dim": 16,
        },
        "ema": {"momentum": 0.9},
        "loss": {"variance_weight": 0.05, "variance_threshold": 0.2},
        "training": {
            "batch_size": 4,
            "max_epochs": 1,
            "max_train_batches": 1,
            "num_workers": 0,
            "device": "cpu",
            "run_root": str(tmp_path / "runs"),
        },
    }
    result = train_td_jepa_from_config(cfg)
    out = tmp_path / "falsification"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_td_falsification_controls.py",
            "--checkpoint",
            str(result["best_checkpoint"]),
            "--data",
            str(manifest_path),
            "--out",
            str(out),
            "--split",
            "test",
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--max-batches",
            "1",
            "--conditions",
            "correct_temporal_pairing",
            "no_motion_predictor",
        ],
    )

    run_falsification_controls()

    summary = json.loads((out / "td_falsification_summary.json").read_text(encoding="utf-8"))
    assert summary["split"] == "test"
    assert summary["results"]["correct_temporal_pairing"]["num_examples"] == 4
    assert summary["results"]["no_motion_predictor"]["num_examples"] == 4
