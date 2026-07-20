"""Run the frozen train-only GRF position-discontinuity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.analysis.gfootball_discontinuity import (  # noqa: E402
    run_gfootball_position_discontinuity_audit,
)

FROZEN_COLLECTION_PLAN_SHA256 = (
    "cba4b38f44ed78dce9b14fcf8d67cb2170552952ae9b44195ec29e0b97d7ee90"
)
FROZEN_SPLIT_MANIFEST_SHA256 = (
    "55b5db0bb003ee3ee4b11180903403ddf1da3df0e22e451311054cca58368e71"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection-manifest",
        type=Path,
        default=Path("data/raw/gfootball/v2_pilot/collection_manifest.json"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("splits/gfootball_v2_pilot_episode_split.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/GRF_POSITION_DISCONTINUITY_AUDIT_PROTOCOL_V1.md"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/integrity/grf_position_discontinuity_audit_v1.json"),
    )
    args = parser.parse_args()

    result = run_gfootball_position_discontinuity_audit(
        args.collection_manifest,
        args.split_manifest,
        repo_root=ROOT,
        expected_collection_plan_sha256=FROZEN_COLLECTION_PLAN_SHA256,
        expected_split_manifest_sha256=FROZEN_SPLIT_MANIFEST_SHA256,
    )
    result["protocol_path"] = str(args.protocol)
    result["protocol_sha256"] = _sha256(args.protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    players = result["global_metrics"]["players"]
    attribution = result["global_event_attribution"]["players"]
    print(f"status: {result['status']}")
    print(f"out: {args.out}")
    print(f"train_jobs: {result['inputs']['train_job_count']}")
    print(f"train_episodes: {result['inputs']['train_episode_count']}")
    print(f"train_frames: {result['inputs']['train_frame_count']}")
    print(
        "player_acceleration: "
        f"mean={players['causal_acceleration_mps2']['mean']:.3f}, "
        f"p99={players['causal_acceleration_mps2']['p99']:.3f}, "
        f"max={players['causal_acceleration_mps2']['max']:.3f}"
    )
    print(f"extreme_player_accelerations: {attribution['extreme_count']}")
    print(f"event_mass_share: {attribution['event_proximate_mass_share']:.6f}")
    print(f"jump_mass_share: {attribution['jump_associated_mass_share']:.6f}")
    print(f"decision: {result['decision']['selected_next_candidate']}")


if __name__ == "__main__":
    main()
