"""Prepare split-aware GRF TD-JEPA shards from a frozen collection plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.gfootball_td_shards import (  # noqa: E402
    VELOCITY_MODES,
    prepare_gfootball_td_jepa_shards,
)
from footballq.repro.feature_views import FEATURE_VIEW_NAMES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--visibility-profile", type=Path, default=None)
    parser.add_argument("--visibility-seed", type=int, default=20260713)
    parser.add_argument("--fps-out", type=float, default=10.0)
    parser.add_argument("--context-seconds", type=float, default=1.0)
    parser.add_argument("--delta-seconds", type=float, default=0.2)
    parser.add_argument("--stride-seconds", type=float, default=0.2)
    parser.add_argument("--prediction-gap-seconds", type=float, default=1.0)
    parser.add_argument("--velocity-mode", choices=sorted(VELOCITY_MODES), default="provider")
    parser.add_argument(
        "--feature-view", choices=sorted(FEATURE_VIEW_NAMES), default="geometry_only"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    path = prepare_gfootball_td_jepa_shards(
        args.plan,
        args.raw_root,
        args.out,
        args.split_manifest,
        visibility_profile_path=args.visibility_profile,
        visibility_seed=args.visibility_seed,
        fps_out=args.fps_out,
        context_seconds=args.context_seconds,
        delta_seconds=args.delta_seconds,
        stride_seconds=args.stride_seconds,
        prediction_gap_seconds=args.prediction_gap_seconds,
        velocity_mode=args.velocity_mode,
        feature_view=args.feature_view,
        included_splits=set(args.splits),
        resume_existing=args.resume_existing,
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(f"dataset_manifest: {path}")
    print(f"examples: {manifest['example_count']}")
    print(f"unique_sample_ids: {manifest['unique_sample_id_count']}")
    print(f"split_examples: {manifest['split_example_counts']}")
    print(f"manifest_sha256: {manifest['manifest_payload_sha256']}")


if __name__ == "__main__":
    main()
