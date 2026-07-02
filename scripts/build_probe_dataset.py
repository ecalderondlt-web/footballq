"""Build Experiment 3 frozen-probe datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.probes.dataset import build_probe_dataset, save_probe_dataset  # noqa: E402
from footballq.repro.manifest import build_run_manifest, write_run_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--allow-legacy-alignment", action="store_true")
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--scientific-mode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_probe_dataset(
        embeddings_path=args.embeddings,
        windows_path=args.windows,
        target_names=args.targets,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        allow_legacy_alignment=args.allow_legacy_alignment,
        split_manifest_path=args.split_manifest,
        scientific_mode=args.scientific_mode,
    )
    save_probe_dataset(data, args.out)
    if args.scientific_mode:
        manifest_path = args.out.with_name(f"{args.out.stem}_run_manifest.json")
        manifest = build_run_manifest(
            command=sys.argv,
            config_path=None,
            split_manifest_path=args.split_manifest,
            evaluation_protocol="inductive",
            feature_view=str(data.metadata.get("feature_view", "unknown")),
            objective_mode=str(data.metadata.get("objective_mode", "downstream_probe")),
            dataset_paths={"embeddings": args.embeddings, "windows": args.windows},
            output_paths={"probe_dataset": args.out, "run_manifest": manifest_path},
            warnings=list(data.metadata.get("warnings", [])),
        )
        write_run_manifest(manifest_path, manifest)
    print(f"probe_dataset: {args.out}")
    print(f"examples: {data.metadata['num_examples']}")
    print(f"targets: {', '.join(data.metadata['targets'])}")
    skipped = data.metadata.get("skipped_targets", [])
    if skipped:
        print(f"skipped_targets: {', '.join(skipped)}")
    for warning in data.metadata.get("warnings", []):
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
