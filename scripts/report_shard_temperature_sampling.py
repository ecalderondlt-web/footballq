"""Write the source-count-derived allocation for a sharded training manifest."""

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

from footballq.data.sharded_td_dataset import temperature_shard_allocations  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    shards = [shard for shard in manifest["shards"] if shard["split"] == args.split]
    counts = [int(shard["example_count"]) for shard in shards]
    allocations = temperature_shard_allocations(
        counts,
        num_samples=args.num_samples,
        temperature=args.temperature,
    )
    source_total = sum(counts)
    rows = []
    for shard, count, allocation in zip(shards, counts, allocations, strict=True):
        rows.append(
            {
                "job_id": shard["job_id"],
                "scenario": shard["scenario"],
                "source_examples": count,
                "source_share": count / source_total,
                "allocated_samples": allocation,
                "allocated_share": allocation / args.num_samples,
                "exposure_ratio": allocation / count,
            }
        )
    report = {
        "status": "frozen_before_training",
        "method": "shard_example_count_power_temperature",
        "manifest_path": str(args.manifest),
        "manifest_file_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "split": args.split,
        "temperature": args.temperature,
        "num_samples": args.num_samples,
        "source_examples": source_total,
        "allocations": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {args.out}")
    print(f"allocated_samples: {sum(allocations)}")
    for row in rows:
        print(
            f"{row['job_id']}: source={row['source_examples']}, "
            f"allocated={row['allocated_samples']}, exposure={row['exposure_ratio']:.3f}x"
        )


if __name__ == "__main__":
    main()
