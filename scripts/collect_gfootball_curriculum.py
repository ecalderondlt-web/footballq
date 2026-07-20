"""Execute an immutable multi-scenario GRF collection plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.io.gfootball_curriculum import (  # noqa: E402
    collection_plan_sha256,
    job_match_prefix,
    load_collection_plan,
    write_episode_split_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    plan = load_collection_plan(args.plan)
    output_root = args.output_root or Path(plan["output_root"])
    split_path = args.split_manifest or Path(plan["split_manifest_path"])
    write_episode_split_manifest(plan, split_path)
    print(f"plan_sha256: {collection_plan_sha256(plan)}")
    print(f"split_manifest: {split_path}")
    print(f"jobs: {len(plan['jobs'])}")
    if args.dry_run:
        print("dry_run: collection skipped")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    collector = ROOT / "scripts" / "collect_gfootball_tracking.py"
    for job in plan["jobs"]:
        out = output_root / job["split"] / f"{job['id']}.jsonl"
        if out.exists() and not args.overwrite:
            raise FileExistsError(f"Collection output already exists: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(collector),
            "--out",
            str(out),
            "--env-name",
            str(job["env_name"]),
            "--episodes",
            str(job["episodes"]),
            "--max-steps",
            str(job["max_steps"]),
            "--seed",
            str(job["seed"]),
            "--action-policy",
            str(job.get("action_policy", "builtin_ai")),
            "--action-set",
            str(job.get("action_set", "full")),
            "--perturbation-rate",
            str(job.get("perturbation_rate", 0.05)),
            "--match-prefix",
            job_match_prefix(plan, job),
            "--collection-job-id",
            str(job["id"]),
            "--split",
            str(job["split"]),
        ]
        for player in job.get("extra_players", []):
            command.extend(["--extra-player", str(player)])
        subprocess.run(command, check=True, cwd=ROOT)
        results.append(
            {
                "id": job["id"],
                "split": job["split"],
                "path": str(out),
                "frames": _line_count(out),
                "sha256": _sha256(out),
            }
        )

    manifest = {
        "status": "complete",
        "version": 1,
        "dataset": "gfootball",
        "collection_plan_path": str(args.plan),
        "collection_plan_sha256": collection_plan_sha256(plan),
        "split_manifest_path": str(split_path),
        "jobs": results,
        "total_frames": sum(item["frames"] for item in results),
    }
    manifest_path = output_root / "collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"collection_manifest: {manifest_path}")
    print(f"total_frames: {manifest['total_frames']}")


if __name__ == "__main__":
    main()
