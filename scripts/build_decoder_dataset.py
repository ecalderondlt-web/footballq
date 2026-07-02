"""Build an Experiment 4C coordinate-decoder dataset."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.decoding.dataset import build_decoder_dataset  # noqa: E402
from footballq.repro.manifest import build_run_manifest, write_run_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--context-z-steps", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--allow-legacy-alignment", action="store_true")
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--scientific-mode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_decoder_dataset(
        args.embeddings,
        args.windows,
        out=args.out,
        horizon_steps=args.horizon_steps,
        context_z_steps=args.context_z_steps,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        allow_legacy_alignment=args.allow_legacy_alignment,
        split_manifest_path=args.split_manifest,
        scientific_mode=args.scientific_mode,
    )
    if args.scientific_mode:
        manifest_path = args.out.with_name(f"{args.out.stem}_run_manifest.json")
        manifest = build_run_manifest(
            command=sys.argv,
            config_path=None,
            split_manifest_path=args.split_manifest,
            evaluation_protocol="inductive",
            feature_view=str(data.metadata.get("feature_view", "unknown")),
            objective_mode=str(data.metadata.get("objective_mode", "coordinate_decoder_dataset")),
            dataset_paths={"embeddings": args.embeddings, "windows": args.windows},
            output_paths={"decoder_dataset": args.out, "run_manifest": manifest_path},
            warnings=list(data.metadata.get("warnings", [])),
        )
        write_run_manifest(manifest_path, manifest)
    print(f"decoder_dataset: {args.out}")
    print(f"examples: {data.num_examples}")
    print(f"latent_dim: {data.latent_dim}")
    print(f"horizon_steps: {data.horizon_steps}")
    print(f"rollout_steps: {data.rollout_steps}")
    print(f"alignment: {data.metadata['alignment']}")
    match_counts = Counter(str(value) for value in data.examples["match_id"])
    print(f"matches: {len(match_counts)}")
    print("examples_per_match:")
    for match_id in sorted(match_counts):
        print(f"- {match_id}: {match_counts[match_id]}")
    print("split_match_ids:")
    for split in ["train", "val", "test"]:
        print(f"- {split}: {', '.join(str(value) for value in data.splits[f'{split}_match_ids'])}")
    if data.metadata.get("warnings"):
        print("warnings:")
        for warning in data.metadata["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
