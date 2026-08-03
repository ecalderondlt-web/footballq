"""Immutable split manifest loading, validation, and hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_SPLIT_DATASETS = {
    "footpass",
    "gfootball",
    "idsse",
    "metrica",
    "pff_fc",
    "rlcs",
    "skillcorner",
    "statsbomb",
    "synthetic",
    "wyscout",
}


@dataclass(frozen=True)
class SplitManifest:
    """Validated match split manifest."""

    path: Path
    payload: dict[str, Any]
    sha256: str

    @property
    def name(self) -> str:
        return str(self.payload["name"])

    @property
    def train_match_ids(self) -> list[str]:
        return [str(value) for value in self.payload["train_match_ids"]]

    @property
    def val_match_ids(self) -> list[str]:
        return [str(value) for value in self.payload["val_match_ids"]]

    @property
    def test_match_ids(self) -> list[str]:
        return [str(value) for value in self.payload["test_match_ids"]]

    @property
    def all_match_ids(self) -> list[str]:
        return [str(value) for value in self.payload["all_match_ids"]]

    def metadata(self) -> dict[str, Any]:
        return {
            "split_manifest_path": str(self.path),
            "split_manifest_name": self.name,
            "split_manifest_sha256": self.sha256,
            "split_protocol": self.payload.get("protocol"),
            "split_dataset": self.payload.get("dataset"),
            "train_match_ids": self.train_match_ids,
            "val_match_ids": self.val_match_ids,
            "test_match_ids": self.test_match_ids,
        }


def stable_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize JSON in a stable form for hashing."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def split_manifest_sha256(payload: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for the manifest payload."""

    return hashlib.sha256(stable_json_bytes(payload)).hexdigest()


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def validate_split_manifest(payload: dict[str, Any], min_scientific_matches: int = 3) -> None:
    """Validate split integrity for scientific runs."""

    required = [
        "name",
        "version",
        "dataset",
        "protocol",
        "train_match_ids",
        "val_match_ids",
        "test_match_ids",
        "all_match_ids",
        "expected_count",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Split manifest is missing required fields: {', '.join(missing)}")

    dataset = str(payload["dataset"])
    if dataset not in ALLOWED_SPLIT_DATASETS:
        allowed = ", ".join(sorted(ALLOWED_SPLIT_DATASETS))
        raise ValueError(f"Split manifest dataset {dataset!r} is not allowed. Expected: {allowed}.")

    train = [str(value) for value in payload["train_match_ids"]]
    val = [str(value) for value in payload["val_match_ids"]]
    test = [str(value) for value in payload["test_match_ids"]]
    all_ids = [str(value) for value in payload["all_match_ids"]]

    duplicate_ids = sorted(
        set(_duplicates(train) + _duplicates(val) + _duplicates(test) + _duplicates(all_ids))
    )
    if duplicate_ids:
        raise ValueError(f"Split manifest contains duplicate match IDs: {', '.join(duplicate_ids)}")

    train_set = set(train)
    val_set = set(val)
    test_set = set(test)
    if not train_set.isdisjoint(val_set):
        raise ValueError("Split manifest train and val match IDs overlap.")
    if not train_set.isdisjoint(test_set):
        raise ValueError("Split manifest train and test match IDs overlap.")
    if not val_set.isdisjoint(test_set):
        raise ValueError("Split manifest val and test match IDs overlap.")

    union = train_set | val_set | test_set
    if union != set(all_ids):
        raise ValueError("Split manifest all_match_ids must equal train/val/test union.")
    if len(all_ids) != int(payload["expected_count"]):
        raise ValueError("Split manifest expected_count does not match all_match_ids length.")
    if len(union) < int(min_scientific_matches):
        raise ValueError(
            f"Scientific split requires at least {min_scientific_matches} distinct matches."
        )


def load_split_manifest(path: str | Path, min_scientific_matches: int = 3) -> SplitManifest:
    """Load, validate, and hash a split manifest."""

    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_split_manifest(payload, min_scientific_matches=min_scientific_matches)
    return SplitManifest(path=path_obj, payload=payload, sha256=split_manifest_sha256(payload))


def split_manifest_metadata(
    split_manifest_path: str | Path | None,
    scientific_mode: bool = False,
) -> dict[str, Any]:
    """Return split metadata or fail when scientific mode requires it."""

    if split_manifest_path is None:
        if scientific_mode:
            raise ValueError("scientific_mode=True requires a split_manifest_path.")
        return {
            "scientific_mode": False,
            "split_manifest_path": None,
            "split_manifest_sha256": None,
        }
    split = load_split_manifest(split_manifest_path)
    return {
        **split.metadata(),
        "scientific_mode": bool(scientific_mode),
    }


def _split_hash_from_mapping(mapping: dict[str, Any]) -> str | None:
    value = mapping.get("split_manifest_sha256")
    if value:
        return str(value)
    for key in ("repro_metadata", "metadata"):
        nested = mapping.get(key)
        if isinstance(nested, dict):
            value = _split_hash_from_mapping(nested)
            if value:
                return value
    return None


def payload_split_manifest_sha256(payload: dict[str, Any]) -> str | None:
    """Return a split hash recorded on an artifact payload, if present."""

    value = _split_hash_from_mapping(payload)
    if value:
        return value
    data_meta = payload.get("data_meta")
    if isinstance(data_meta, dict):
        return _split_hash_from_mapping(data_meta)
    return None


def assert_split_hash_compatible(
    source_payload: dict[str, Any],
    expected_metadata: dict[str, Any],
    *,
    source_name: str = "source artifact",
    require_source_hash: bool = False,
) -> None:
    """Fail if a downstream artifact is built from missing or mismatched split lineage."""

    expected = expected_metadata.get("split_manifest_sha256")
    if not expected:
        return
    actual = payload_split_manifest_sha256(source_payload)
    if actual is None:
        if require_source_hash:
            raise ValueError(
                f"{source_name} is missing split_manifest_sha256 required for scientific mode."
            )
        return
    if str(actual) != str(expected):
        raise ValueError(
            f"{source_name} split_manifest_sha256 mismatch: expected {expected}, got {actual}."
        )


def split_indices_from_manifest(
    match_ids: list[str],
    split_manifest_path: str | Path,
) -> dict[str, list[int]]:
    """Assign row indices using an immutable match split manifest."""

    split = load_split_manifest(split_manifest_path)
    train = set(split.train_match_ids)
    val = set(split.val_match_ids)
    test = set(split.test_match_ids)
    seen = set(str(value) for value in match_ids)
    manifest_ids = train | val | test
    missing = sorted(seen - manifest_ids)
    if missing:
        raise ValueError("Rows contain match IDs absent from split manifest: " + ", ".join(missing))
    return {
        "train": [idx for idx, match_id in enumerate(match_ids) if str(match_id) in train],
        "val": [idx for idx, match_id in enumerate(match_ids) if str(match_id) in val],
        "test": [idx for idx, match_id in enumerate(match_ids) if str(match_id) in test],
    }


def named_split_indices_from_manifest(
    match_ids: list[str],
    split_manifest_path: str | Path,
) -> dict[str, Any]:
    """Return train/val/test indices and manifest match IDs with dataset-style keys."""

    split = load_split_manifest(split_manifest_path)
    indices = split_indices_from_manifest(match_ids, split_manifest_path)
    return {
        "train_indices": indices["train"],
        "val_indices": indices["val"],
        "test_indices": indices["test"],
        "train_match_ids": split.train_match_ids,
        "val_match_ids": split.val_match_ids,
        "test_match_ids": split.test_match_ids,
    }
