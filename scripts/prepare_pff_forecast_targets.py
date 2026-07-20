"""Prepare train/validation-only PFF multi-horizon forecast targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.pff_forecasting import prepare_pff_forecast_targets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--horizons", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--match-ids", nargs="*", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    manifest = prepare_pff_forecast_targets(
        args.source_manifest,
        args.out,
        args.split_manifest,
        horizons_seconds=tuple(args.horizons),
        included_splits=tuple(args.splits),
        match_ids=args.match_ids,
        resume=not args.no_resume,
    )
    print(f"prepared_examples: {manifest['example_count']}")
    print(f"valid_endpoints: {manifest['valid_endpoint_count']}")
    print(f"included_splits: {','.join(manifest['included_splits'])}")
    print(f"test_included: {manifest['test_included']}")
    print(f"manifest: {args.out / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
