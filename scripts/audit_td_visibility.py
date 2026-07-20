"""Report visibility statistics from a finalized sharded TD-JEPA dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.parent.parent
    player_visible = 0
    ball_visible = 0
    frame_count = 0
    count_histogram = torch.zeros(23, dtype=torch.long)
    shard_count = 0
    scenario_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"frames": 0, "players": 0, "ball": 0}
    )

    for shard in manifest["shards"]:
        if shard["split"] != args.split:
            continue
        payload = torch.load(root / shard["path"], map_location="cpu", weights_only=False)
        mask = payload["mask_t"].bool()
        player_counts = mask[:, :, 1:].sum(dim=-1)
        count_histogram += torch.bincount(player_counts.reshape(-1), minlength=23)
        player_visible += int(player_counts.sum())
        ball_visible += int(mask[:, :, 0].sum())
        frame_count += int(mask.shape[0] * mask.shape[1])
        shard_count += 1
        scenario = str(shard.get("scenario", "unknown"))
        scenario_totals[scenario]["frames"] += int(mask.shape[0] * mask.shape[1])
        scenario_totals[scenario]["players"] += int(player_counts.sum())
        scenario_totals[scenario]["ball"] += int(mask[:, :, 0].sum())

    if frame_count == 0:
        raise ValueError(f"Manifest contains no context frames for split {args.split!r}.")
    nonzero_counts = torch.nonzero(count_histogram, as_tuple=False).flatten()
    print(f"split: {args.split}")
    print(f"shards: {shard_count}")
    print(f"context_frames: {frame_count}")
    print(f"mean_visible_players: {player_visible / frame_count:.6f}")
    print(f"ball_visible_rate: {ball_visible / frame_count:.6f}")
    print(f"min_visible_players: {int(nonzero_counts[0])}")
    print(f"max_visible_players: {int(nonzero_counts[-1])}")
    print("scenario_visibility:")
    for scenario, totals in sorted(scenario_totals.items()):
        frames = totals["frames"]
        print(
            f"  {scenario}: frames={frames}, "
            f"mean_players={totals['players'] / frames:.6f}, "
            f"ball_rate={totals['ball'] / frames:.6f}"
        )


if __name__ == "__main__":
    main()
