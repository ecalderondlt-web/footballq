"""Sample future latent rollouts from a trained latent flow checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.latent_flow.dataset import LatentRolloutDataset, load_latent_rollout_dataset  # noqa: E402
from footballq.latent_flow.baselines import denormalize_residual  # noqa: E402
from footballq.latent_flow.dataset import ensure_residual_targets, residual_normalization_stats  # noqa: E402
from footballq.latent_flow.flow_matching import sample_latent_flow  # noqa: E402
from footballq.latent_flow.models import create_latent_model  # noqa: E402
from footballq.training.train import resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    data = load_latent_rollout_dataset(args.dataset)
    model_name = str(payload["model_name"])
    if model_name == "residual_latent_flow_mlp":
        residual_mode = str(
            payload.get("residual_mode")
            or cfg.get("flow", {}).get("residual_mode", "last_latent")
        )
        data = ensure_residual_targets(data, residual_mode)
    indices = data.splits[f"{args.split}_indices"][: args.num_examples]
    loader = DataLoader(LatentRolloutDataset(data, indices=indices), batch_size=args.num_examples)
    batch = next(iter(loader))
    device = resolve_device(args.device)
    model = create_latent_model(
        cfg,
        latent_dim=int(payload["latent_dim"]),
        context_steps=int(payload["context_steps"]),
        horizon_steps=int(payload["horizon_steps"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(device)
    past_z = batch["past_z"].to(device)
    flow_cfg = cfg.get("flow", {})
    steps = int(
        args.num_steps
        or flow_cfg.get("num_sampling_steps", cfg.get("sampling", {}).get("num_steps", 20))
    )
    noise_scale = float(flow_cfg.get("noise_scale", cfg.get("sampling", {}).get("noise_scale", 1.0)))
    if bool(flow_cfg.get("deterministic_mean_eval", False)):
        noise_scale = 0.0
    samples = sample_latent_flow(
        model,
        past_z,
        horizon_steps=data.horizon_steps,
        latent_dim=data.latent_dim,
        num_samples=args.num_samples,
        num_steps=steps,
        noise_scale=noise_scale,
    )
    if model_name == "residual_latent_flow_mlp":
        normalization = payload.get("normalization") or data.metadata.get("normalization", {})
        if "residual_mean" in normalization and "residual_std" in normalization:
            residual_mean = normalization["residual_mean"].float().to(device)
            residual_std = normalization["residual_std"].float().to(device)
        else:
            mean, std = residual_normalization_stats(data)
            residual_mean = mean.to(device)
            residual_std = std.to(device)
        residual = denormalize_residual(samples, residual_mean, residual_std)
        samples = batch["baseline_future_z"].to(device).unsqueeze(1) + residual
    samples = samples.cpu()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "samples": samples,
            "past_z": batch["past_z"],
            "future_z": batch["future_z"],
            "future_mask": batch["future_mask"],
            "match_id": list(batch["match_id"]),
            "frame_t": [int(value) for value in batch["frame_t"]],
            "layout": "[num_examples, num_samples, horizon_steps, latent_dim]",
        },
        args.out,
    )
    print(f"samples: {args.out}")
    print(f"shape: {tuple(samples.shape)}")


if __name__ == "__main__":
    main()
