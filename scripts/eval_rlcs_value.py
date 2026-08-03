from __future__ import annotations

import argparse
from pathlib import Path

from footballq.training.eval_rlcs_value import (
    evaluate_sealed_value_test,
    evaluate_value_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen RLCS V2 critical-value models.")
    parser.add_argument("--config", default="configs/rlcs_player_matchup_value_v2.yaml")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("internal_development", "validation", "test"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unlock", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "test":
        if args.unlock is None:
            parser.error("--unlock is required for the sealed test")
        paths = evaluate_sealed_value_test(
            args.config,
            bundle_path=args.bundle,
            unlock_path=args.unlock,
            output_dir=args.output,
            resume=args.resume,
        )
    else:
        if args.unlock is not None:
            parser.error("--unlock is only valid for the sealed test")
        paths = evaluate_value_bundle(
            args.config,
            bundle_path=args.bundle,
            stage=args.stage,
            output_dir=args.output,
        )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
