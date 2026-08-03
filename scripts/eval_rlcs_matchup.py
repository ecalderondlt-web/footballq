from __future__ import annotations

import argparse
from pathlib import Path

from footballq.training.eval_matchup import evaluate_sealed_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-time sealed RLCS test evaluation.")
    parser.add_argument("--config", default="configs/rlcs_identity_matchup_v1.yaml")
    parser.add_argument("--unlock", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/rlcs_identity_matchup_v1/sealed_test")
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    paths = evaluate_sealed_test(
        args.config,
        unlock_path=args.unlock,
        output_dir=args.output,
        resume=args.resume,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
