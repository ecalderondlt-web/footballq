"""Build a train-only PFF visibility profile for synthetic-domain masking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.synthetic_visibility import (  # noqa: E402
    build_pff_visibility_profile,
    write_visibility_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    args = parser.parse_args()
    profile = build_pff_visibility_profile(
        args.canonical_root,
        split="train",
        frame_stride=args.frame_stride,
    )
    path = write_visibility_profile(profile, args.out)
    print(f"profile: {path}")
    print(f"sampled_frames: {profile['sampled_frame_count']}")
    print(f"player_observed_rate: {profile['player_observed_rate']:.6f}")
    print(f"ball_observed_rate: {profile['ball_observed_rate']:.6f}")
    print(f"profile_sha256: {profile['profile_payload_sha256']}")


if __name__ == "__main__":
    main()
