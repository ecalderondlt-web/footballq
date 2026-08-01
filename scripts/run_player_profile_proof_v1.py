from __future__ import annotations

import argparse
from pathlib import Path

from footballq.analysis.player_profile_proof import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen chronological player-profile proof."
    )
    parser.add_argument(
        "--config",
        default="configs/player_profile_proof_v1.yaml",
    )
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    paths = run_experiment(
        args.config,
        workspace_root=Path(args.workspace_root).resolve(),
        device=args.device,
        rebuild_cache=args.rebuild_cache,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
