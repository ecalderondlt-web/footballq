"""Build latent transition examples from TD-JEPA embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.discovery.transitions import (  # noqa: E402
    build_transition_dataset,
    transition_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delta-steps", nargs="+", type=int, default=[2, 5, 10])
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--scientific-mode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_transition_dataset(
        args.embeddings,
        args.windows,
        out=args.out,
        delta_steps=args.delta_steps,
        fps=args.fps,
        split_manifest_path=args.split_manifest,
        scientific_mode=args.scientific_mode,
    )
    summary = transition_summary(data)
    summary_path = args.out.with_name(f"{args.out.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"transition_dataset: {args.out}")
    print(f"summary: {summary_path}")
    print(f"num_examples: {data.num_examples}")
    print(f"num_matches: {summary['num_matches']}")
    print(f"delta_seconds: {', '.join(str(value) for value in summary['requested_delta_seconds'])}")
    if summary.get("missing_metadata_fields"):
        print(f"missing_metadata_fields: {', '.join(summary['missing_metadata_fields'])}")


if __name__ == "__main__":
    main()
