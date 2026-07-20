"""Derive an episode-prefix subset from a larger frozen GRF collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

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


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _job_signature(job: dict[str, Any]) -> dict[str, Any]:
    return {
        key: job.get(key)
        for key in (
            "id",
            "split",
            "env_name",
            "max_steps",
            "seed",
            "action_policy",
            "action_set",
            "perturbation_rate",
            "extra_players",
        )
    }


def derive_collection_subset(
    master_plan_path: str | Path,
    subset_plan_path: str | Path,
    master_root: str | Path,
    output_root: str | Path,
    split_manifest_path: str | Path,
) -> Path:
    """Copy exact JSONL lines for each job's frozen episode prefix."""

    master_plan_path = Path(master_plan_path)
    subset_plan_path = Path(subset_plan_path)
    master = load_collection_plan(master_plan_path)
    subset = load_collection_plan(subset_plan_path)
    if master["name"] != subset["name"]:
        raise ValueError("Master and subset plans must share a match-identity namespace.")

    master_jobs = {str(job["id"]): job for job in master["jobs"]}
    subset_jobs = {str(job["id"]): job for job in subset["jobs"]}
    if set(master_jobs) != set(subset_jobs):
        raise ValueError("Master and subset plans must contain the same jobs.")
    for job_id, subset_job in subset_jobs.items():
        master_job = master_jobs[job_id]
        if _job_signature(master_job) != _job_signature(subset_job):
            raise ValueError(f"Collection job settings differ for {job_id!r}.")
        if int(subset_job["episodes"]) > int(master_job["episodes"]):
            raise ValueError(f"Subset job {job_id!r} requests more episodes than the master.")

    master_root = Path(master_root)
    output_root = Path(output_root)
    results = []
    for job in subset["jobs"]:
        split = str(job["split"])
        source_path = master_root / split / f"{job['id']}.jsonl"
        target_path = output_root / split / f"{job['id']}.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing master collection output: {source_path}")
        if target_path.exists():
            raise FileExistsError(f"Subset collection output already exists: {target_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        episode_limit = int(job["episodes"])
        frames = 0
        observed_episodes: set[int] = set()
        expected_prefix = job_match_prefix(subset, job)
        with source_path.open("rb") as source, target_path.open("wb") as target:
            for line in source:
                record = json.loads(line)
                episode_id = int(record["episode_id"])
                if episode_id >= episode_limit:
                    continue
                expected_match_id = f"{expected_prefix}_episode_{episode_id}"
                if str(record.get("match_id")) != expected_match_id:
                    raise ValueError(
                        "Master match identity does not match subset namespace: "
                        f"{expected_match_id}"
                    )
                target.write(line)
                frames += 1
                observed_episodes.add(episode_id)
        if observed_episodes != set(range(episode_limit)):
            raise ValueError(f"Subset job {job['id']!r} is missing planned episodes.")
        results.append(
            {
                "id": job["id"],
                "split": split,
                "path": str(target_path),
                "frames": frames,
                "sha256": _sha256(target_path),
                "source_path": str(source_path),
                "source_sha256": _sha256(source_path),
            }
        )

    write_episode_split_manifest(subset, split_manifest_path)
    manifest = {
        "status": "complete",
        "version": 1,
        "dataset": "gfootball",
        "derivation": "episode_prefix_subset",
        "collection_plan_path": str(subset_plan_path),
        "collection_plan_sha256": collection_plan_sha256(subset),
        "master_collection_plan_path": str(master_plan_path),
        "master_collection_plan_sha256": collection_plan_sha256(master),
        "master_collection_manifest_sha256": _sha256(
            master_root / "collection_manifest.json"
        ),
        "split_manifest_path": str(split_manifest_path),
        "jobs": results,
        "total_frames": sum(item["frames"] for item in results),
    }
    manifest_path = output_root / "collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-plan", type=Path, required=True)
    parser.add_argument("--subset-plan", type=Path, required=True)
    parser.add_argument("--master-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = derive_collection_subset(
        args.master_plan,
        args.subset_plan,
        args.master_root,
        args.out,
        args.split_manifest,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"collection_manifest: {manifest_path}")
    print(f"total_frames: {manifest['total_frames']}")


if __name__ == "__main__":
    main()
