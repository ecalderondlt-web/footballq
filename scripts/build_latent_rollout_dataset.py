"""Build latent rollout datasets from exported TD-JEPA embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.latent_flow.dataset import build_latent_rollout_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context-steps", type=int, default=5)
    parser.add_argument("--horizon-steps", type=int, default=5)
    parser.add_argument("--stride-steps", type=int, default=1)
    parser.add_argument(
        "--residual-mode",
        choices=["none", "last_latent", "constant_latent_velocity"],
        default="constant_latent_velocity",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_latent_rollout_dataset(
        embeddings_path=args.embeddings,
        out=args.out,
        context_steps=args.context_steps,
        horizon_steps=args.horizon_steps,
        stride_steps=args.stride_steps,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        residual_mode=args.residual_mode,
    )
    print(f"latent_rollout_dataset: {args.out}")
    print(f"examples: {data.num_examples}")
    print(f"past_z: {tuple(data.examples['past_z'].shape)}")
    print(f"future_z: {tuple(data.examples['future_z'].shape)}")
    print(
        "split_match_ids: "
        f"train={data.splits['train_match_ids']} "
        f"val={data.splits['val_match_ids']} "
        f"test={data.splits['test_match_ids']}"
    )
    for warning in data.metadata.get("warnings", []):
        print(f"warning: {warning}")
    if "normalization" in data.metadata:
        print(f"residual_mode: {data.metadata['normalization']['residual_mode']}")


if __name__ == "__main__":
    main()
