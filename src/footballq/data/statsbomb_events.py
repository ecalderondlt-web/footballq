"""StatsBomb Open Data catalog, split, and source-manifest helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from footballq.repro.splits import split_manifest_sha256, validate_split_manifest

STATSBOMB_OPEN_DATA_COMMIT = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
STATSBOMB_SPLIT_SEED = "statsbomb-open-data-b0bc9f2-match-inductive-v1"
STATSBOMB_SPLIT_NAME = "statsbomb_open_data_b0bc9f2_match_inductive_v1"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _stable_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_stable_bytes(payload)).hexdigest()


def statsbomb_event_payload(event: dict[str, Any]) -> dict[str, Any]:
    """Return the provider-specific payload nested below a StatsBomb event."""

    event_name = str((event.get("type") or {}).get("name") or "")
    key = re.sub(r"[^a-z0-9]+", "_", event_name.lower()).strip("_")
    aliases = {"goal_keeper": "goalkeeper"}
    for candidate in (key, aliases.get(key)):
        payload = event.get(candidate) if candidate else None
        if isinstance(payload, dict):
            return payload
    excluded = {
        "type",
        "team",
        "player",
        "position",
        "possession_team",
        "play_pattern",
    }
    candidates = [
        value
        for candidate_key, value in event.items()
        if candidate_key not in excluded
        and isinstance(value, dict)
        and any(field in value for field in ("end_location", "outcome", "type"))
    ]
    return candidates[0] if len(candidates) == 1 else {}


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it wholly into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_statsbomb_data_dir(raw_root: str | Path) -> Path:
    """Resolve either an Open Data repository root or its nested data directory."""

    root = Path(raw_root)
    for candidate in (root, root / "data"):
        if (candidate / "competitions.json").is_file() and (candidate / "events").is_dir():
            return candidate
    raise FileNotFoundError(f"StatsBomb Open Data tree not found below: {root}")


def _match_sort_key(match_id: str) -> tuple[int, int | str]:
    return (0, int(match_id)) if match_id.isdigit() else (1, match_id)


def load_statsbomb_match_catalog(raw_root: str | Path) -> list[dict[str, Any]]:
    """Load provider match metadata and verify one event and lineup file per match."""

    data_dir = resolve_statsbomb_data_dir(raw_root)
    catalog: dict[str, dict[str, Any]] = {}
    for path in sorted((data_dir / "matches").rglob("*.json")):
        for match in _read_json(path):
            match_id = str(match["match_id"])
            if match_id in catalog:
                raise ValueError(f"Duplicate StatsBomb match ID: {match_id}")
            competition = match.get("competition") or {}
            season = match.get("season") or {}
            home_team = match.get("home_team") or {}
            away_team = match.get("away_team") or {}
            catalog[match_id] = {
                "match_id": match_id,
                "competition_id": competition.get("competition_id"),
                "competition_name": competition.get("competition_name"),
                "country_name": competition.get("country_name"),
                "season_id": season.get("season_id"),
                "season_name": season.get("season_name"),
                "match_date": match.get("match_date"),
                "home_team_id": home_team.get("home_team_id"),
                "away_team_id": away_team.get("away_team_id"),
                "match_status": match.get("match_status"),
                "match_status_360": match.get("match_status_360"),
                "has_360": (data_dir / "three-sixty" / f"{match_id}.json").is_file(),
                "metadata_available": True,
                "match_metadata_path": path.relative_to(data_dir).as_posix(),
            }

    catalog_ids = set(catalog)
    event_ids = {path.stem for path in (data_dir / "events").glob("*.json")}
    lineup_ids = {path.stem for path in (data_dir / "lineups").glob("*.json")}
    if event_ids != lineup_ids:
        missing_lineups = sorted(event_ids - lineup_ids, key=_match_sort_key)
        missing_events = sorted(lineup_ids - event_ids, key=_match_sort_key)
        raise ValueError(
            "StatsBomb event/lineup mismatch: "
            f"missing_lineups={missing_lineups}, missing_events={missing_events}"
        )
    missing_events = sorted(catalog_ids - event_ids, key=_match_sort_key)
    if missing_events:
        raise ValueError(f"StatsBomb metadata matches missing event files: {missing_events}")
    for match_id in sorted(event_ids - catalog_ids, key=_match_sort_key):
        catalog[match_id] = {
            "match_id": match_id,
            "competition_id": None,
            "competition_name": None,
            "country_name": None,
            "season_id": None,
            "season_name": None,
            "match_date": None,
            "home_team_id": None,
            "away_team_id": None,
            "match_status": None,
            "match_status_360": None,
            "has_360": (data_dir / "three-sixty" / f"{match_id}.json").is_file(),
            "metadata_available": False,
            "match_metadata_path": None,
        }
    return [catalog[key] for key in sorted(catalog, key=_match_sort_key)]


def _allocate_counts(size: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = [size * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remainder = size - sum(counts)
    priority = sorted(range(3), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in priority[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def build_statsbomb_split(
    catalog: list[dict[str, Any]],
    *,
    seed: str = STATSBOMB_SPLIT_SEED,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, Any]:
    """Create a split stratified by 360 and match-metadata availability."""

    if not catalog:
        raise ValueError("StatsBomb catalog is empty.")
    if any(ratio <= 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("StatsBomb split ratios must be positive and sum to one.")

    split_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    strata: dict[str, dict[str, int]] = {}
    stratum_keys = sorted(
        {(bool(row["metadata_available"]), bool(row["has_360"])) for row in catalog}
    )
    for metadata_available, has_360 in stratum_keys:
        metadata_label = "metadata" if metadata_available else "orphaned"
        label = f"{metadata_label}_{'with_360' if has_360 else 'without_360'}"
        ids = [
            str(row["match_id"])
            for row in catalog
            if bool(row["metadata_available"]) == metadata_available
            and bool(row["has_360"]) == has_360
        ]
        ids.sort(key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
        train_count, val_count, test_count = _allocate_counts(len(ids), ratios)
        split_ids["train"].extend(ids[:train_count])
        split_ids["val"].extend(ids[train_count : train_count + val_count])
        split_ids["test"].extend(ids[train_count + val_count :])
        strata[label] = {
            "total": len(ids),
            "train": train_count,
            "val": val_count,
            "test": test_count,
        }

    for ids in split_ids.values():
        ids.sort(key=_match_sort_key)
    all_ids = sorted(
        split_ids["train"] + split_ids["val"] + split_ids["test"],
        key=_match_sort_key,
    )
    payload = {
        "name": STATSBOMB_SPLIT_NAME,
        "version": 1,
        "dataset": "statsbomb",
        "protocol": "match_inductive_stratified_by_360_and_metadata_availability",
        "source_repository": "https://github.com/statsbomb/open-data",
        "source_commit": STATSBOMB_OPEN_DATA_COMMIT,
        "split_seed": seed,
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "strata": strata,
        "train_match_ids": split_ids["train"],
        "val_match_ids": split_ids["val"],
        "test_match_ids": split_ids["test"],
        "all_match_ids": all_ids,
        "expected_count": len(all_ids),
    }
    validate_split_manifest(payload)
    return payload


def build_statsbomb_source_manifest(
    raw_root: str | Path,
    catalog: list[dict[str, Any]],
    split_payload: dict[str, Any],
    *,
    archive_path: str | Path | None = None,
    hash_files: bool = True,
) -> dict[str, Any]:
    """Build a source manifest without parsing event, lineup, or 360 payloads."""

    data_dir = resolve_statsbomb_data_dir(raw_root)
    groups = {
        "competitions": [data_dir / "competitions.json"],
        "matches": sorted((data_dir / "matches").rglob("*.json")),
        "events": sorted((data_dir / "events").glob("*.json")),
        "lineups": sorted((data_dir / "lineups").glob("*.json")),
        "three_sixty": sorted((data_dir / "three-sixty").glob("*.json")),
    }
    files = []
    for group, paths in groups.items():
        for path in paths:
            row = {
                "group": group,
                "path": path.relative_to(data_dir).as_posix(),
                "size_bytes": path.stat().st_size,
            }
            if hash_files:
                row["sha256"] = file_sha256(path)
            files.append(row)

    archive = Path(archive_path) if archive_path is not None else None
    manifest = {
        "version": 1,
        "dataset": "statsbomb_open_data",
        "source_repository": "https://github.com/statsbomb/open-data",
        "source_commit": STATSBOMB_OPEN_DATA_COMMIT,
        "source_archive": (
            {
                "name": archive.name,
                "size_bytes": archive.stat().st_size,
                "sha256": file_sha256(archive),
            }
            if archive is not None
            else None
        ),
        "coverage": {
            "matches": len(catalog),
            "metadata_indexed_matches": sum(bool(row["metadata_available"]) for row in catalog),
            "orphaned_event_matches": sum(
                not bool(row["metadata_available"]) for row in catalog
            ),
            "matches_with_360": sum(bool(row["has_360"]) for row in catalog),
            "competition_season_files": len(groups["matches"]),
            "event_files": len(groups["events"]),
            "lineup_files": len(groups["lineups"]),
            "three_sixty_files": len(groups["three_sixty"]),
        },
        "split_manifest_name": split_payload["name"],
        "split_manifest_sha256": split_manifest_sha256(split_payload),
        "catalog_sha256": _stable_hash(catalog),
        "file_hashes_complete": bool(hash_files),
        "files": files,
    }
    manifest["manifest_payload_sha256"] = _stable_hash(manifest)
    return manifest


def _record_category(
    values: dict[int, Counter[str]],
    payload: object,
) -> None:
    if not isinstance(payload, dict) or payload.get("id") is None:
        return
    values[int(payload["id"])][str(payload.get("name") or "") ] += 1


def _category_vocab(values: dict[int, Counter[str]]) -> dict[str, Any]:
    entries = []
    for provider_id in sorted(values):
        names = values[provider_id]
        entries.append(
            {
                "index": len(entries) + 2,
                "provider_id": provider_id,
                "name": names.most_common(1)[0][0],
                "observed_names": sorted(names),
            }
        )
    return {
        "missing_index": 0,
        "unknown_index": 1,
        "size": len(entries) + 2,
        "entries": entries,
    }


def _location_is_valid(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and 0.0 <= float(value[0]) <= 120.0
        and 0.0 <= float(value[1]) <= 80.0
    )


def audit_statsbomb_training_schema(
    raw_root: str | Path,
    split_payload: dict[str, Any],
) -> dict[str, Any]:
    """Audit event and 360 schemas using training matches only."""

    validate_split_manifest(split_payload)
    data_dir = resolve_statsbomb_data_dir(raw_root)
    train_ids = [str(value) for value in split_payload["train_match_ids"]]
    categories: dict[str, dict[int, Counter[str]]] = {
        name: defaultdict(Counter)
        for name in ("event_type", "play_pattern", "position", "subtype", "outcome")
    }
    periods: Counter[int] = Counter()
    counts: Counter[str] = Counter()
    maxima = {"events_per_match": 0, "events_per_period": 0, "freeze_frame_players": 0}
    malformed_360 = []

    for match_id in train_ids:
        event_path = data_dir / "events" / f"{match_id}.json"
        events = _read_json(event_path)
        counts["events"] += len(events)
        maxima["events_per_match"] = max(maxima["events_per_match"], len(events))
        period_counts: Counter[int] = Counter()
        last_index: dict[int, int] = {}
        event_ids = set()
        for event in events:
            event_id = event.get("id")
            if event_id is not None:
                event_ids.add(str(event_id))
            period = int(event.get("period") or 0)
            periods[period] += 1
            period_counts[period] += 1
            event_index = int(event.get("index") or 0)
            if period in last_index and event_index <= last_index[period]:
                counts["non_increasing_event_indices"] += 1
            last_index[period] = event_index

            _record_category(categories["event_type"], event.get("type"))
            _record_category(categories["play_pattern"], event.get("play_pattern"))
            _record_category(categories["position"], event.get("position"))
            payload = statsbomb_event_payload(event)
            _record_category(categories["subtype"], payload.get("type"))
            _record_category(categories["outcome"], payload.get("outcome"))

            location = event.get("location")
            if isinstance(location, list) and len(location) >= 2:
                counts["events_with_location"] += 1
                if not _location_is_valid(location):
                    counts["out_of_bounds_locations"] += 1
            end_location = payload.get("end_location")
            if isinstance(end_location, list) and len(end_location) >= 2:
                counts["events_with_end_location"] += 1
                if not _location_is_valid(end_location):
                    counts["out_of_bounds_end_locations"] += 1
        maxima["events_per_period"] = max(
            maxima["events_per_period"],
            max(period_counts.values(), default=0),
        )

        frame_path = data_dir / "three-sixty" / f"{match_id}.json"
        if not frame_path.is_file():
            continue
        counts["three_sixty_files_present"] += 1
        try:
            frame_rows = _read_json(frame_path)
        except json.JSONDecodeError as exc:
            malformed_360.append(
                {
                    "match_id": match_id,
                    "path": frame_path.relative_to(data_dir).as_posix(),
                    "sha256": file_sha256(frame_path),
                    "error": str(exc),
                }
            )
            continue
        counts["three_sixty_files_parseable"] += 1
        counts["three_sixty_rows"] += len(frame_rows)
        for row in frame_rows:
            if str(row.get("event_uuid")) not in event_ids:
                counts["three_sixty_event_id_misses"] += 1
            freeze_frame = row.get("freeze_frame") or []
            maxima["freeze_frame_players"] = max(
                maxima["freeze_frame_players"], len(freeze_frame)
            )
            for player in freeze_frame:
                if not _location_is_valid(player.get("location")):
                    counts["out_of_bounds_freeze_frame_locations"] += 1

    vocabularies = {name: _category_vocab(values) for name, values in categories.items()}
    audit = {
        "version": 1,
        "dataset": "statsbomb_open_data",
        "source_commit": STATSBOMB_OPEN_DATA_COMMIT,
        "scope": "train_only",
        "loaded_splits": ["train"],
        "split_manifest_name": split_payload["name"],
        "split_manifest_sha256": split_manifest_sha256(split_payload),
        "train_match_count": len(train_ids),
        "counts": dict(sorted(counts.items())),
        "events_by_period": {str(key): value for key, value in sorted(periods.items())},
        "maxima": maxima,
        "malformed_three_sixty": malformed_360,
        "vocabularies": vocabularies,
        "vocabulary_payload_sha256": _stable_hash(vocabularies),
    }
    audit["audit_payload_sha256"] = _stable_hash(audit)
    return audit


def write_immutable_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write deterministic JSON, refusing to replace different existing content."""

    output = Path(path)
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != text:
        raise FileExistsError(
            f"Refusing to replace immutable JSON with different content: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output
