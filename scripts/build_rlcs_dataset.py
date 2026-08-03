from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from footballq.data.rlcs_ballchasing import read_inventory_parquet, sha256_file
from footballq.data.rlcs_replay import (
    PARSER_VERSION,
    IdentityResolutionError,
    IdentityResolver,
    ReplayParseError,
    ReplayQC,
    cache_parsed_replay,
    freeze_chronological_split_manifest,
    load_alias_registry,
    load_cached_replay,
    load_frozen_rlcs_split,
    parse_replay_file,
    quality_control_replay,
    roster_observations,
)
from footballq.data.rlcs_touch_windows import (
    build_replay_decisions,
    decision_arrow_schema,
    fit_identity_vocabulary,
    save_identity_vocabulary,
    write_dataset_manifest,
    write_decision_parquet_batches,
)
from footballq.repro.manifest import file_sha256


def _atomic_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _split_lookup(payload: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for replay_id in payload[f"{split}_match_ids"]:
            if str(replay_id) in lookup:
                raise ValueError(f"Replay {replay_id} occurs in multiple splits.")
            lookup[str(replay_id)] = split
    return lookup


def _validate_source_file(raw_dir: Path, row: dict[str, Any]) -> Path:
    replay_id = str(row["replay_id"])
    path = raw_dir / "replays" / f"{replay_id}.replay"
    if not path.exists():
        raise FileNotFoundError(f"Missing downloaded replay: {path}")
    expected_size = row.get("file_size_bytes")
    if expected_size is not None and int(expected_size) != path.stat().st_size:
        raise ValueError(f"Replay byte-size mismatch: {replay_id}")
    expected_hash = str(row.get("file_sha256") or "")
    if expected_hash and sha256_file(path) != expected_hash:
        raise ValueError(f"Replay SHA-256 mismatch: {replay_id}")
    return path


def _reuse_completed_split(
    path: Path, *, expected_replay_ids: set[str]
) -> Counter[str]:
    """Validate one completed split and recover replay-level sample counts."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RLCS split resume requires pyarrow.") from exc
    schema = pq.read_schema(path)
    expected_schema = decision_arrow_schema()

    def compatible_type(actual: Any, expected: Any) -> bool:
        if actual == expected:
            return True
        return bool(
            pa.types.is_fixed_size_list(actual)
            and pa.types.is_fixed_size_list(expected)
            and actual.list_size == expected.list_size
            and actual.value_type == expected.value_type
        )

    schema_matches = (
        schema.names == expected_schema.names
        and schema.metadata == expected_schema.metadata
        and all(
            actual.nullable == expected.nullable
            and compatible_type(actual.type, expected.type)
            for actual, expected in zip(schema, expected_schema, strict=True)
        )
    )
    if not schema_matches:
        raise ValueError(f"Completed split schema does not match the frozen schema: {path}")
    table = pq.read_table(path, columns=["sample_id", "replay_id"])
    sample_ids = [str(value) for value in table["sample_id"].to_pylist()]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Completed split contains duplicate sample IDs: {path}")
    replay_ids = [str(value) for value in table["replay_id"].to_pylist()]
    unexpected = sorted(set(replay_ids) - expected_replay_ids)
    if unexpected:
        raise ValueError(
            f"Completed split contains replay IDs outside the accepted split: {unexpected[:5]}"
        )
    if not replay_ids:
        raise ValueError(f"Refusing to reuse an empty completed split: {path}")
    return Counter(replay_ids)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = Path(args.raw)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = raw_dir / "replay_inventory.parquet"
    records = read_inventory_parquet(inventory_path)
    if not records:
        raise ValueError("Replay inventory is empty.")
    split_path = Path(args.split_manifest)
    split_template = json.loads(split_path.read_text(encoding="utf-8"))
    if split_template.get("status") != "frozen":
        freeze_chronological_split_manifest(
            records,
            template_path=split_path,
            inventory_sha256=file_sha256(inventory_path),
        )
    split_payload = load_frozen_rlcs_split(split_path)
    split_lookup = _split_lookup(split_payload)
    inventory_by_id = {str(row["replay_id"]): dict(row) for row in records}
    absent = sorted(set(split_lookup) - set(inventory_by_id))
    if absent:
        raise ValueError(
            "Frozen split contains replay IDs absent from inventory: " + ", ".join(absent)
        )

    ordered_ids = sorted(
        split_lookup,
        key=lambda replay_id: (
            {"train": 0, "val": 1, "test": 2}[split_lookup[replay_id]],
            str(inventory_by_id[replay_id].get("event_time_utc") or ""),
            replay_id,
        ),
    )
    if args.parse_limit is not None:
        ordered_ids = ordered_ids[: int(args.parse_limit)]
    cache_dir = output_dir / "parser_cache"
    observations_by_id = {}
    replay_report: list[dict[str, Any]] = []
    for index, replay_id in enumerate(ordered_ids, start=1):
        row = inventory_by_id[replay_id]
        split = split_lookup[replay_id]
        report: dict[str, Any] = {
            "replay_id": replay_id,
            "split": split,
            "series_id": row.get("series_id") or row.get("leaf_group_id"),
            "event_time_utc": row.get("event_time_utc"),
            "parse_success": False,
            "qc_accepted": False,
            "identity_accepted": False,
            "sample_count": 0,
        }
        try:
            replay_path = _validate_source_file(raw_dir, row)
            cache_path = cache_dir / replay_id / "metadata.json"
            if cache_path.exists() and not args.reparse:
                parsed = load_cached_replay(cache_dir, replay_id)
                parsed.qc = quality_control_replay(
                    parsed.frames,
                    parsed.events,
                    map_name=row.get("map_name"),
                    minimum_duration_seconds=args.minimum_duration_seconds,
                    maximum_duration_seconds=args.maximum_duration_seconds,
                    require_standard_3v3=args.require_standard_3v3,
                    expected_blue_score=row.get("blue_score"),
                    expected_orange_score=row.get("orange_score"),
                )
            else:
                parsed = parse_replay_file(
                    replay_path,
                    replay_id=replay_id,
                    workers=args.workers,
                    map_name=row.get("map_name"),
                    minimum_duration_seconds=args.minimum_duration_seconds,
                    maximum_duration_seconds=args.maximum_duration_seconds,
                    require_standard_3v3=args.require_standard_3v3,
                    expected_blue_score=row.get("blue_score"),
                    expected_orange_score=row.get("orange_score"),
                )
                cache_parsed_replay(parsed, cache_dir)
            observations = roster_observations(
                parsed.frames,
                replay_id=replay_id,
                split=split,
                event_time_utc=row.get("event_time_utc"),
                group_id=str(row.get("leaf_group_id") or "") or None,
            )
            observations_by_id[replay_id] = observations
            report.update(
                {
                    "parse_success": True,
                    "qc_accepted": parsed.qc.accepted,
                    "qc": asdict(parsed.qc),
                    "observed_roster": [
                        {
                            "prefix": item.prefix,
                            "team": item.team,
                            "handle": item.handle,
                            "platform": item.platform,
                            "platform_id": item.platform_id,
                        }
                        for item in observations
                    ],
                }
            )
        except (FileNotFoundError, ValueError, ReplayParseError, RuntimeError) as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        replay_report.append(report)
        if index % 25 == 0 or index == len(ordered_ids):
            print(f"parsed {index}/{len(ordered_ids)}", flush=True)

    aliases = load_alias_registry(args.aliases)
    report_by_id = {str(row["replay_id"]): row for row in replay_report}
    training_observations = [
        item
        for replay_id, items in observations_by_id.items()
        if split_lookup[replay_id] == "train" and report_by_id[replay_id]["qc_accepted"]
        for item in items
    ]
    audit_observations = [
        item
        for replay_id, items in observations_by_id.items()
        if report_by_id[replay_id]["qc_accepted"]
        for item in items
    ]
    resolver = IdentityResolver.fit_training(training_observations, aliases).with_collision_audit(
        audit_observations
    )
    resolver_path = _atomic_json(output_dir / "identity_resolver.json", resolver.to_dict())
    accepted_rosters: dict[str, list[list[str]]] = {"train": [], "val": [], "test": []}
    resolved_rosters_by_id: dict[str, dict[str, str]] = {}
    for replay_id, observations in observations_by_id.items():
        report = report_by_id[replay_id]
        if not report["qc_accepted"]:
            continue
        split = split_lookup[replay_id]
        try:
            roster_ids = resolver.resolve_roster(observations)
        except IdentityResolutionError as exc:
            report["identity_error"] = str(exc)
            continue
        report["identity_accepted"] = True
        report["canonical_roster"] = [roster_ids[item.prefix] for item in observations]
        resolved_rosters_by_id[replay_id] = roster_ids
        accepted_rosters[split].append(report["canonical_roster"])

    samples_by_split = {"train": 0, "val": 0, "test": 0}

    def write_quality_report() -> Path:
        summary = {
            "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "inventory_path": str(inventory_path),
            "inventory_sha256": file_sha256(inventory_path),
            "split_manifest_path": str(split_path),
            "split_manifest_sha256": file_sha256(split_path),
            "parser_package": "analyzerl-parser",
            "parser_version": PARSER_VERSION,
            "parse_limit": args.parse_limit,
            "counts": {
                "inventory": len(records),
                "attempted": len(ordered_ids),
                "parse_success": sum(bool(row["parse_success"]) for row in replay_report),
                "qc_accepted": sum(bool(row["qc_accepted"]) for row in replay_report),
                "identity_accepted": sum(
                    bool(row["identity_accepted"]) for row in replay_report
                ),
                "samples_by_split": dict(samples_by_split),
                "qc_rejections": dict(
                    Counter(
                        reason
                        for row in replay_report
                        for reason in row.get("qc", {}).get("reasons", [])
                    )
                ),
            },
            "replays": replay_report,
        }
        return _atomic_json(output_dir / "quality_report.json", summary)

    if args.audit_only:
        quality_path = write_quality_report()
        return {
            "quality_report": quality_path,
            "identity_resolver": resolver_path,
            "dataset_manifest": None,
        }

    vocabulary = fit_identity_vocabulary(accepted_rosters["train"])
    vocabulary_path = save_identity_vocabulary(
        vocabulary, output_dir / "identity_vocabulary.json"
    )
    split_paths: dict[str, Path] = {}

    def decision_batches(split: str):
        split_ids = [replay_id for replay_id in ordered_ids if split_lookup[replay_id] == split]
        for index, replay_id in enumerate(split_ids, start=1):
            report = report_by_id[replay_id]
            if not report["identity_accepted"]:
                continue
            parsed = load_cached_replay(cache_dir, replay_id)
            qc_payload = dict(report["qc"])
            qc_payload["reasons"] = tuple(qc_payload.get("reasons", []))
            parsed.qc = ReplayQC(**qc_payload)
            decisions = build_replay_decisions(
                parsed,
                inventory=inventory_by_id[replay_id],
                split=split,
                observations=observations_by_id[replay_id],
                roster_ids=resolved_rosters_by_id[replay_id],
                vocabulary=vocabulary,
                fps=args.fps,
                context_seconds=args.context_seconds,
                min_next_touch_dt=args.min_next_touch_dt,
                max_next_touch_dt=args.max_next_touch_dt,
                exclude_goal_reset_seconds=args.exclude_goal_reset_seconds,
            )
            report["sample_count"] = len(decisions)
            samples_by_split[split] += len(decisions)
            if index % 25 == 0 or index == len(split_ids):
                print(
                    f"built {split} {index}/{len(split_ids)} "
                    f"({samples_by_split[split]} samples)",
                    flush=True,
                )
            yield decisions

    for split in ("train", "val", "test"):
        destination = output_dir / f"{split}.parquet"
        expected_replay_ids = {
            replay_id
            for replay_id in ordered_ids
            if split_lookup[replay_id] == split
            and report_by_id[replay_id]["identity_accepted"]
        }
        if getattr(args, "resume_built_splits", False) and destination.exists():
            replay_counts = _reuse_completed_split(
                destination, expected_replay_ids=expected_replay_ids
            )
            samples_by_split[split] = sum(replay_counts.values())
            for replay_id in expected_replay_ids:
                report_by_id[replay_id]["sample_count"] = replay_counts[replay_id]
            split_paths[split] = destination
            print(
                f"reused {split} ({samples_by_split[split]} samples)",
                flush=True,
            )
        else:
            split_paths[split] = write_decision_parquet_batches(
                decision_batches(split), destination
            )
        gc.collect()
    quality_path = write_quality_report()
    dataset_manifest = write_dataset_manifest(
        output_dir=output_dir,
        split_paths=split_paths,
        vocabulary_path=vocabulary_path,
        split_manifest_path=split_path,
        parser_version=PARSER_VERSION,
        quality_report_path=quality_path,
    )
    return {
        "quality_report": quality_path,
        "identity_resolver": resolver_path,
        "identity_vocabulary": vocabulary_path,
        "dataset_manifest": dataset_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen RLCS touch-decision dataset.")
    parser.add_argument("--config", default="configs/rlcs_identity_matchup_v1.yaml")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/rlcs_2025"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/rlcs_identity_matchup_v1")
    )
    parser.add_argument(
        "--split-manifest", default="splits/rlcs_2025_chronological_v1.json"
    )
    parser.add_argument("--aliases", default="provenance/rlcs_identity_aliases_v1.csv")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--min-next-touch-dt", type=float, default=0.20)
    parser.add_argument("--max-next-touch-dt", type=float, default=4.00)
    parser.add_argument("--exclude-goal-reset-seconds", type=float, default=2.0)
    parser.add_argument("--minimum-duration-seconds", type=float, default=180.0)
    parser.add_argument("--maximum-duration-seconds", type=float, default=1800.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--require-standard-3v3", action="store_true", default=True)
    parser.add_argument("--allow-nonstandard", dest="require_standard_3v3", action="store_false")
    parser.add_argument("--parse-limit", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--reparse", action="store_true")
    parser.add_argument(
        "--resume-built-splits",
        action="store_true",
        help="Validate and reuse completed split Parquets after an interrupted build.",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if config_path.exists():
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = build_dataset(args)
    for name, path in outputs.items():
        if path is not None:
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
