from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from footballq.data.rlcs_player_profiles import (
    build_v2_split_frame,
    observations_and_roster,
)
from footballq.data.rlcs_value_windows import (
    build_replay_value_rows,
    value_arrow_schema,
    write_value_parquet,
)
from footballq.repro.manifest import file_sha256

STAGE_ORDER = ("train", "internal_development", "validation")


def _atomic_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _build_replay_shard(job: dict[str, Any]) -> dict[str, Any]:
    """Build one replay in an isolated process and persist its rows as a shard."""

    cache = Path(job["cache"])
    frames = pq.read_table(cache / "frames.parquet").to_pandas()
    events = pq.read_table(cache / "events.parquet").to_pandas()
    observations, roster = observations_and_roster(job["quality_record"])
    rows = build_replay_value_rows(
        frames,
        events,
        replay_id=job["replay_id"],
        inventory=job["inventory"],
        stage=job["stage"],
        observations=observations,
        roster_ids=roster,
        snapshots=job["snapshots"],
        fps=job["fps"],
        context_seconds=job["context_seconds"],
        horizon_touches=job["horizon_touches"],
        exclude_goal_reset_seconds=job["exclude_goal_reset_seconds"],
    )
    shard_path = Path(job["shard_path"])
    if rows:
        write_value_parquet(rows, shard_path)
    counts = Counter(int(row["outcome_label"]) for row in rows)
    return {
        "index": int(job["index"]),
        "replay_id": job["replay_id"],
        "series_id": str(job["inventory"].get("series_id") or ""),
        "shard_path": str(shard_path) if rows else None,
        "rows": len(rows),
        "label_counts": dict(counts),
    }


def _merge_shards(shards: list[Path], destination: Path, temporary_dir: Path) -> Path:
    """Merge replay shards without materializing the complete dataset in memory."""

    merged = temporary_dir / "merged.parquet"
    with pq.ParquetWriter(merged, value_arrow_schema(), compression="zstd") as writer:
        for shard in shards:
            writer.write_table(pq.read_table(shard))
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.replace(destination)
    return destination


def _manifest_payload(
    *,
    config_path: Path,
    data_cfg: dict[str, Any],
    audit_path: Path,
    manifest_stages: dict[str, Any],
) -> dict[str, Any]:
    opened = [stage for stage in STAGE_ORDER if stage in manifest_stages]
    return {
        "version": 2,
        "experiment": "rlcs_player_matchup_value_v2",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "split_manifest_path": str(data_cfg["split_manifest"]),
        "split_manifest_sha256": file_sha256(data_cfg["split_manifest"]),
        "profile_audit_path": str(audit_path),
        "profile_audit_sha256": file_sha256(audit_path),
        "opened_stages": opened,
        "test_loaded": False,
        "stages": manifest_stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build past-only ten-touch RLCS V2 value rows.")
    parser.add_argument("--config", default="configs/rlcs_player_matchup_value_v2.yaml")
    parser.add_argument(
        "--stage",
        action="append",
        choices=("train", "internal_development", "validation"),
        help="Defaults to train and internal development. Sealed test is never available here.",
    )
    parser.add_argument(
        "--frozen-bundle",
        type=Path,
        help="Required before opening Split 2 Regional 1 validation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Replay-construction worker processes (default: up to four local cores).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    stages = set(args.stage or ["train", "internal_development"])
    if "test" in stages:
        raise PermissionError("The sealed V2 test is unavailable to ordinary dataset builders.")
    audit_path = Path(data_cfg["profile_audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not bool(audit.get("all_gates_pass")):
        raise RuntimeError("V2 profile gate failed; value-dataset construction is forbidden.")
    if "validation" in stages:
        if args.frozen_bundle is None:
            raise PermissionError("Validation construction requires --frozen-bundle.")
        bundle = json.loads(args.frozen_bundle.read_text(encoding="utf-8"))
        if not bool(bundle.get("architecture_frozen")):
            raise PermissionError("The V2 architecture is not frozen after internal development.")

    inventory = pd.read_parquet(data_cfg["inventory"])
    split_frame = build_v2_split_frame(
        inventory,
        split1_fractions=(
            float(config["chronology"]["split1_profile_support_fraction"]),
            float(config["chronology"]["split1_training_fraction"]),
            float(config["chronology"]["split1_internal_development_fraction"]),
        ),
    )
    quality = json.loads(Path(data_cfg["v1_quality_report"]).read_text(encoding="utf-8"))
    accepted = {
        str(record["replay_id"]): record
        for record in quality.get("replays", quality.get("records", []))
        if bool(record.get("qc_accepted")) and bool(record.get("identity_accepted"))
    }
    snapshots = pd.read_parquet(data_cfg["profile_snapshots"])
    snapshot_groups = {
        str(replay_id): {
            str(row.player_id): {
                "profile": row.profile,
                "uncertainty": row.uncertainty,
                "n_prior_games": row.n_prior_games,
                "effective_sample_size": row.effective_sample_size,
                "prior_win_rate": row.prior_win_rate,
                "prior_goal_diff": row.prior_goal_diff,
            }
            for row in rows.itertuples(index=False)
        }
        for replay_id, rows in snapshots.groupby("replay_id", sort=False)
    }
    parser_cache = Path(data_cfg["parser_cache"])
    output_dir = Path(data_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = Path(data_cfg["dataset_manifest"])
    manifest_stages: dict[str, Any] = {}
    if existing_manifest_path.exists():
        old = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        same_lineage = (
            old.get("config_sha256") == file_sha256(config_path)
            and old.get("split_manifest_sha256") == file_sha256(data_cfg["split_manifest"])
            and old.get("profile_audit_sha256") == file_sha256(audit_path)
        )
        if same_lineage:
            manifest_stages.update(old.get("stages", {}))

    for stage in (candidate for candidate in STAGE_ORDER if candidate in stages):
        selected = split_frame.loc[split_frame["v2_stage"] == stage]
        selected = selected.loc[selected["replay_id"].astype(str).isin(accepted)]
        failures: list[dict[str, str]] = []
        records = selected.to_dict(orient="records")
        total_rows = 0
        label_counts: Counter[int] = Counter()
        series_ids: set[str] = set()
        shard_by_index: dict[int, Path] = {}
        with tempfile.TemporaryDirectory(
            prefix=f"{stage}_value_shards_", dir=output_dir
        ) as temporary_name:
            temporary_dir = Path(temporary_name)
            jobs: list[dict[str, Any]] = []
            for index, inventory_row in enumerate(records):
                replay_id = str(inventory_row["replay_id"])
                if replay_id not in snapshot_groups:
                    failures.append(
                        {"replay_id": replay_id, "error": "missing_profile_snapshots"}
                    )
                    continue
                jobs.append(
                    {
                        "index": index,
                        "replay_id": replay_id,
                        "inventory": inventory_row,
                        "quality_record": accepted[replay_id],
                        "snapshots": snapshot_groups[replay_id],
                        "cache": str(parser_cache / replay_id),
                        "stage": stage,
                        "fps": float(data_cfg["fps"]),
                        "context_seconds": float(data_cfg["context_seconds"]),
                        "horizon_touches": int(data_cfg["future_distinct_touches"]),
                        "exclude_goal_reset_seconds": float(
                            data_cfg["exclude_goal_reset_seconds"]
                        ),
                        "shard_path": str(temporary_dir / f"{index:05d}.parquet"),
                    }
                )
            completed = 0
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(_build_replay_shard, job): job for job in jobs
                }
                for future in as_completed(futures):
                    job = futures[future]
                    replay_id = str(job["replay_id"])
                    try:
                        result = future.result()
                        total_rows += int(result["rows"])
                        label_counts.update(
                            {
                                int(label): int(count)
                                for label, count in result["label_counts"].items()
                            }
                        )
                        if result["shard_path"]:
                            shard_by_index[int(result["index"])] = Path(
                                result["shard_path"]
                            )
                            series_ids.add(str(result["series_id"]))
                    except Exception as exc:
                        failures.append(
                            {
                                "replay_id": replay_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    completed += 1
                    if completed == len(jobs) or completed % 25 == 0:
                        print(
                            f"{stage}: built {completed}/{len(jobs)} replays; "
                            f"rows={total_rows}",
                            flush=True,
                        )
            if failures:
                failure_path = output_dir / f"{stage}_build_failures.json"
                _atomic_json(failure_path, {"stage": stage, "failures": failures})
                raise RuntimeError(
                    f"V2 {stage} construction failed closed for {len(failures)} replays; "
                    f"see {failure_path}."
                )
            if not shard_by_index:
                raise RuntimeError(f"V2 {stage} produced no eligible value rows.")
            ordered_shards = [shard_by_index[index] for index in sorted(shard_by_index)]
            path = _merge_shards(
                ordered_shards, output_dir / f"{stage}.parquet", temporary_dir
            )
        manifest_stages[stage] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "rows": total_rows,
            "series": len(series_ids),
            "label_counts": {
                "no_goal": int(label_counts.get(0, 0)),
                "score": int(label_counts.get(1, 0)),
                "concede": int(label_counts.get(2, 0)),
            },
        }
        manifest = _manifest_payload(
            config_path=config_path,
            data_cfg=data_cfg,
            audit_path=audit_path,
            manifest_stages=manifest_stages,
        )
        _atomic_json(existing_manifest_path, manifest)
        if stage == "train":
            train_counts = manifest_stages["train"]["label_counts"]
            required_score = int(config["evaluation"]["minimum_training_score_rows"])
            required_concede = int(config["evaluation"]["minimum_training_concede_rows"])
            passed = (
                int(train_counts.get("score", 0)) >= required_score
                and int(train_counts.get("concede", 0)) >= required_concede
            )
            print(
                f"label-count gate: {'PASS' if passed else 'STOP'}; {train_counts}",
                flush=True,
            )
            if not passed:
                raise RuntimeError("V2 label-count gate failed; outcome training is forbidden.")
    print(f"dataset manifest: {existing_manifest_path}")


if __name__ == "__main__":
    main()
