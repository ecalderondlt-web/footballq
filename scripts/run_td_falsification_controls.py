"""Evaluate TD-JEPA falsification controls on a checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.td_jepa_dataset import TDJEPADataset  # noqa: E402
from footballq.repro.falsification import (  # noqa: E402
    CONTROL_CONDITIONS,
    apply_td_falsification_control,
)
from footballq.repro.manifest import build_run_manifest, write_run_manifest  # noqa: E402
from footballq.training.eval_td_jepa import load_td_checkpoint_model  # noqa: E402
from footballq.training.td_jepa_losses import td_jepa_loss  # noqa: E402
from footballq.training.train_td_jepa import td_batch_to_device  # noqa: E402

DEFAULT_CONDITIONS = [
    "correct_temporal_pairing",
    "shuffled_future_within_batch",
    "future_from_another_match",
    "reversed_time_context",
    "masked_ball",
    "team_swap",
    "pitch_reflection",
    "consistent_player_slot_permutation",
    "no_motion_predictor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _empty_totals() -> dict[str, float]:
    return {}


def _add_metrics(
    totals: dict[str, float],
    metrics: dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item()) * batch_size


def _finalize(totals: dict[str, float], num_examples: int, skipped_batches: int) -> dict[str, Any]:
    if num_examples == 0:
        return {"num_examples": 0, "skipped_batches": skipped_batches}
    return {
        key: value / num_examples for key, value in totals.items()
    } | {"num_examples": num_examples, "skipped_batches": skipped_batches}


def _condition_losses(
    model: torch.nn.Module,
    batch: dict[str, Any],
    condition: str,
    loss_cfg: dict[str, Any],
    seed: int,
) -> dict[str, torch.Tensor]:
    if condition == "no_motion_predictor":
        outputs = model(batch)
        return td_jepa_loss(
            outputs["z_t"],
            outputs["z_target"],
            outputs["z_t"],
            variance_weight=float(loss_cfg.get("variance_weight", 0.1)),
            variance_threshold=float(loss_cfg.get("variance_threshold", 1.0)),
        )
    controlled = apply_td_falsification_control(batch, condition, seed=seed)
    outputs = model(controlled)
    return td_jepa_loss(
        outputs["z_pred"],
        outputs["z_target"],
        outputs["z_t"],
        variance_weight=float(loss_cfg.get("variance_weight", 0.1)),
        variance_threshold=float(loss_cfg.get("variance_threshold", 1.0)),
    )


def main() -> None:
    args = parse_args()
    invalid = sorted(set(args.conditions) - (CONTROL_CONDITIONS | {"no_motion_predictor"}))
    if invalid:
        raise ValueError(f"Unknown falsification conditions: {', '.join(invalid)}")

    model, data, cfg, payload, torch_device = load_td_checkpoint_model(
        args.checkpoint,
        data_path=args.data,
        device=args.device,
    )
    split_indices = payload.get("split_indices", {})
    if args.split not in split_indices:
        raise ValueError(f"Split {args.split!r} not found. Available: {sorted(split_indices)}")

    batch_size = args.batch_size or int(cfg.get("training", {}).get("batch_size", 64))
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    loader = DataLoader(
        TDJEPADataset(data, indices=split_indices[args.split]),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    loss_cfg = cfg.get("loss", {})
    totals = {condition: _empty_totals() for condition in args.conditions}
    counts = {condition: 0 for condition in args.conditions}
    skipped = {condition: 0 for condition in args.conditions}

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = td_batch_to_device(batch, torch_device)
            batch_size_now = int(batch["state_t"].shape[0])
            for condition in args.conditions:
                try:
                    losses = _condition_losses(
                        model,
                        batch,
                        condition,
                        loss_cfg,
                        seed=int(args.seed) + batch_idx,
                    )
                except ValueError:
                    skipped[condition] += 1
                    continue
                _add_metrics(totals[condition], losses, batch_size_now)
                counts[condition] += batch_size_now
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break

    results = {
        condition: _finalize(totals[condition], counts[condition], skipped[condition])
        for condition in args.conditions
    }
    summary = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "split": args.split,
        "max_batches": args.max_batches,
        "feature_view": data.feature_view,
        "objective_mode": data.objective_mode,
        "split_manifest_sha256": (data.metadata or {}).get("split_manifest_sha256"),
        "results": results,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "td_falsification_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    split_manifest_path = (data.metadata or {}).get("split_manifest_path")
    if split_manifest_path:
        manifest_path = args.out / "run_manifest.json"
        config_path = Path(payload.get("run_dir", args.checkpoint.parent)) / "config.yaml"
        manifest = build_run_manifest(
            command=sys.argv,
            config_path=config_path if config_path.exists() else None,
            split_manifest_path=split_manifest_path,
            evaluation_protocol="inductive",
            feature_view=data.feature_view,
            objective_mode=data.objective_mode,
            dataset_paths={"checkpoint": args.checkpoint, "td_jepa": args.data},
            output_paths={"summary": summary_path, "run_manifest": manifest_path},
            warnings=[],
        )
        write_run_manifest(manifest_path, manifest)

    print(f"summary: {summary_path}")
    for condition, metrics in results.items():
        print(
            f"{condition}: total_loss={metrics.get('total_loss')} "
            f"td_loss={metrics.get('td_loss')} cosine={metrics.get('cosine_similarity')}"
        )


if __name__ == "__main__":
    main()
