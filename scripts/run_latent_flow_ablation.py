"""Run stochastic residual latent-flow ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.latent_flow.ablation import run_latent_flow_ablation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--noise-scales", nargs="*", type=float, default=[0.0, 0.01, 0.03, 0.05, 0.1])
    parser.add_argument("--num-steps", nargs="*", type=int, default=[5, 10, 20])
    parser.add_argument("--num-samples", nargs="*", type=int, default=[4, 8, 16])
    parser.add_argument("--max-mean-ade-multiplier", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_latent_flow_ablation(
        args.base_config,
        args.out,
        checkpoint=args.checkpoint,
        noise_scales=args.noise_scales,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        split=args.split,
        device=args.device,
        max_mean_ade_multiplier=args.max_mean_ade_multiplier,
    )
    summary = result["summary"]
    print(f"results_csv: {result['results_csv']}")
    print(f"summary_json: {result['summary_json']}")
    print(f"decision: {summary['decision']}")
    best = summary.get("best_residual_flow_config_by_minADE")
    if best:
        print(
            "best_residual_minADE: "
            f"noise={best['noise_scale']} steps={best['num_sampling_steps']} "
            f"samples={best['num_samples']} minADE={best['minADE']}"
        )


if __name__ == "__main__":
    main()
