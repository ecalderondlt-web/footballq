"""Run provenance metadata for scientific artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from footballq.repro.splits import load_split_manifest

REQUIRED_RUN_FIELDS = [
    "created_at_utc",
    "command",
    "git",
    "config_path",
    "config_sha256",
    "split_manifest_path",
    "split_manifest_sha256",
    "evaluation_protocol",
    "feature_view",
    "objective_mode",
    "dataset_paths",
    "output_paths",
    "warnings",
    "python",
]


def file_sha256(path: str | Path) -> str:
    """Hash a file with SHA-256."""

    path_obj = Path(path)
    digest = hashlib.sha256()
    with path_obj.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return file_sha256(path)


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_metadata() -> dict[str, Any]:
    """Return local git provenance without contacting remotes."""

    dirty = _git_value(["status", "--porcelain"])
    return {
        "remote_url": _git_value(["config", "--get", "remote.origin.url"]),
        "branch": _git_value(["branch", "--show-current"]),
        "commit": _git_value(["rev-parse", "HEAD"]),
        "dirty": bool(dirty),
    }


def build_run_manifest(
    *,
    command: list[str] | str,
    config_path: str | Path | None,
    split_manifest_path: str | Path,
    evaluation_protocol: str,
    feature_view: str,
    objective_mode: str,
    dataset_paths: dict[str, str | Path],
    output_paths: dict[str, str | Path] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build validated run provenance for a scientific artifact."""

    split = load_split_manifest(split_manifest_path)
    manifest = {
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "command": command if isinstance(command, str) else " ".join(command),
        "git": git_metadata(),
        "config_path": None if config_path is None else str(config_path),
        "config_sha256": _optional_file_sha256(config_path),
        **split.metadata(),
        "evaluation_protocol": str(evaluation_protocol),
        "feature_view": str(feature_view),
        "objective_mode": str(objective_mode),
        "dataset_paths": {key: str(value) for key, value in dataset_paths.items()},
        "output_paths": {key: str(value) for key, value in (output_paths or {}).items()},
        "warnings": list(warnings or []),
        "python": {
            "version": sys.version,
            "platform": platform.platform(),
        },
    }
    validate_run_manifest(manifest)
    return manifest


def write_run_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Write a validated run manifest as JSON."""

    validate_run_manifest(manifest)
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest_path


def validate_run_manifest(manifest: dict[str, Any]) -> None:
    """Validate required provenance fields."""

    missing = [field for field in REQUIRED_RUN_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"Run manifest is missing required fields: {', '.join(missing)}")
    git = manifest.get("git", {})
    for field in ["remote_url", "branch", "commit", "dirty"]:
        if field not in git:
            raise ValueError(f"Run manifest git metadata is missing {field!r}.")
    if not manifest.get("split_manifest_sha256"):
        raise ValueError("Run manifest requires split_manifest_sha256.")
    if not manifest.get("feature_view"):
        raise ValueError("Run manifest requires feature_view.")
    if not manifest.get("objective_mode"):
        raise ValueError("Run manifest requires objective_mode.")
