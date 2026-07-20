"""Validation and split lineage for GRF curriculum collection plans."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from footballq.repro.splits import stable_json_bytes, validate_split_manifest

JOB_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
ALLOWED_POLICIES = {"builtin_ai", "builtin_ai_perturbed", "idle", "random"}
ALLOWED_SPLITS = {"train", "val", "test"}


def load_collection_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_collection_plan(plan)
    return plan


def validate_collection_plan(plan: dict[str, Any]) -> None:
    required = ["name", "version", "dataset", "jobs", "creation_timestamp_utc"]
    missing = [key for key in required if key not in plan]
    if missing:
        raise ValueError(f"Collection plan is missing: {', '.join(missing)}")
    if plan["dataset"] != "gfootball":
        raise ValueError("GRF collection plan dataset must be 'gfootball'.")
    if not isinstance(plan["jobs"], list) or not plan["jobs"]:
        raise ValueError("Collection plan must contain at least one job.")

    seen: set[str] = set()
    for job in plan["jobs"]:
        job_id = str(job.get("id", ""))
        if not JOB_ID_PATTERN.fullmatch(job_id):
            raise ValueError(f"Invalid collection job id: {job_id!r}")
        if job_id in seen:
            raise ValueError(f"Duplicate collection job id: {job_id}")
        seen.add(job_id)
        if job.get("split") not in ALLOWED_SPLITS:
            raise ValueError(f"Job {job_id} has invalid split.")
        if job.get("action_policy", "builtin_ai") not in ALLOWED_POLICIES:
            raise ValueError(f"Job {job_id} has invalid action_policy.")
        for field in ("episodes", "max_steps", "seed"):
            if int(job.get(field, 0)) <= 0:
                raise ValueError(f"Job {job_id} requires positive {field}.")
        if not str(job.get("env_name", "")):
            raise ValueError(f"Job {job_id} requires env_name.")
        rate = float(job.get("perturbation_rate", 0.05))
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Job {job_id} perturbation_rate must lie in [0, 1].")


def collection_plan_sha256(plan: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_bytes(plan)).hexdigest()


def job_match_prefix(plan: dict[str, Any], job: dict[str, Any]) -> str:
    return f"{plan['name']}_{job['id']}"


def job_match_ids(plan: dict[str, Any], job: dict[str, Any]) -> list[str]:
    prefix = job_match_prefix(plan, job)
    return [f"{prefix}_episode_{episode}" for episode in range(int(job["episodes"]))]


def build_episode_split_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    validate_collection_plan(plan)
    by_split = {split: [] for split in ("train", "val", "test")}
    for job in plan["jobs"]:
        by_split[str(job["split"])].extend(job_match_ids(plan, job))
    all_ids = sorted([*by_split["train"], *by_split["val"], *by_split["test"]])
    payload = {
        "name": f"{plan['name']}_episode_split",
        "version": 1,
        "dataset": "gfootball",
        "protocol": "inductive_episode_holdout",
        "train_match_ids": by_split["train"],
        "val_match_ids": by_split["val"],
        "test_match_ids": by_split["test"],
        "all_match_ids": all_ids,
        "source": "frozen_gfootball_collection_plan",
        "source_collection_plan_sha256": collection_plan_sha256(plan),
        "creation_timestamp_utc": plan["creation_timestamp_utc"],
        "expected_count": len(all_ids),
        "notes": "Frozen before GRF V2 collection or model training.",
    }
    validate_split_manifest(payload)
    return payload


def write_episode_split_manifest(plan: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_episode_split_manifest(plan), indent=2), encoding="utf-8")
    return target
