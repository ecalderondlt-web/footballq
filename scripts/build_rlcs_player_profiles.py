from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

from footballq.data.rlcs_player_profiles import (
    PROFILE_EVENT_COLUMNS,
    build_profile_snapshots,
    build_v2_split_frame,
    compute_player_game_profiles,
    fit_profile_priors,
    profile_frame_columns,
    split_manifest_payload,
    write_profile_parquet,
)
from footballq.repro.manifest import file_sha256


def _atomic_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build chronology-safe RLCS V2 player profiles from the V1 parser cache."
    )
    parser.add_argument("--config", default="configs/rlcs_player_matchup_value_v2.yaml")
    parser.add_argument(
        "--stage",
        action="append",
        choices=("profile_support", "train", "internal_development", "validation"),
        help="Open stage to profile. Defaults to profile_support only; sealed test is unavailable.",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    profile_cfg = config["profiles"]
    output_dir = Path(data_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = pd.read_parquet(data_cfg["inventory"])
    split_frame = build_v2_split_frame(
        inventory,
        split1_fractions=(
            float(config["chronology"]["split1_profile_support_fraction"]),
            float(config["chronology"]["split1_training_fraction"]),
            float(config["chronology"]["split1_internal_development_fraction"]),
        ),
    )
    split_payload = split_manifest_payload(split_frame)
    split_payload["inventory_path"] = str(data_cfg["inventory"])
    split_payload["inventory_sha256"] = file_sha256(data_cfg["inventory"])
    split_path = _atomic_json(Path(data_cfg["split_manifest"]), split_payload)

    quality = json.loads(Path(data_cfg["v1_quality_report"]).read_text(encoding="utf-8"))
    records = quality.get("replays", quality.get("records", []))
    accepted = {
        str(record["replay_id"]): record
        for record in records
        if bool(record.get("qc_accepted")) and bool(record.get("identity_accepted"))
    }
    stages = set(args.stage or ["profile_support"])
    if "test" in stages:
        raise PermissionError("The sealed V2 test cannot be opened by the profile builder.")
    selected = split_frame.loc[split_frame["v2_stage"].isin(stages)].copy()
    selected = selected.loc[selected["replay_id"].astype(str).isin(accepted)]
    parser_cache = Path(data_cfg["parser_cache"])

    rows: list[dict] = []
    failures: list[dict[str, str]] = []
    total = len(selected)
    for number, inventory_row in enumerate(selected.to_dict(orient="records"), start=1):
        replay_id = str(inventory_row["replay_id"])
        record = accepted[replay_id]
        cache = parser_cache / replay_id
        try:
            frame_columns = profile_frame_columns(record)
            available_frames = set(pq.ParquetFile(cache / "frames.parquet").schema_arrow.names)
            frame_columns = [column for column in frame_columns if column in available_frames]
            frames = pq.read_table(cache / "frames.parquet", columns=frame_columns).to_pandas()
            available_events = set(pq.ParquetFile(cache / "events.parquet").schema_arrow.names)
            event_columns = [
                column for column in PROFILE_EVENT_COLUMNS if column in available_events
            ]
            events = pq.read_table(cache / "events.parquet", columns=event_columns).to_pandas()
            rows.extend(
                compute_player_game_profiles(
                    frames,
                    events,
                    quality_record=record,
                    inventory_row=inventory_row,
                    stage=str(inventory_row["v2_stage"]),
                    sample_seconds=float(data_cfg["profile_frame_sample_seconds"]),
                )
            )
        except Exception as exc:  # fail closed but retain an auditable ledger
            failures.append({"replay_id": replay_id, "error": f"{type(exc).__name__}: {exc}"})
        if number == total or number % 25 == 0:
            print(f"profiled {number}/{total} accepted replays; failures={len(failures)}")

    failure_path = output_dir / "profile_build_failures.json"
    if failures:
        _atomic_json(
            failure_path,
            {
                "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "status": "failed_closed",
                "failures": failures,
            },
        )
        raise RuntimeError(
            f"Profile construction failed closed for {len(failures)} replays; see {failure_path}."
        )
    _atomic_json(
        failure_path,
        {
            "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "status": "passed",
            "failures": [],
        },
    )
    if not rows:
        raise RuntimeError("No accepted replay profiles were built.")

    games = pd.DataFrame(rows)
    games_path = write_profile_parquet(games, data_cfg["profile_games"])
    priors = fit_profile_priors(
        games,
        minimum_prior_games=float(profile_cfg["empirical_bayes_minimum_prior_games"]),
        maximum_prior_games=float(profile_cfg["empirical_bayes_maximum_prior_games"]),
    )
    priors["profile_games_sha256"] = file_sha256(games_path)
    priors_path = _atomic_json(Path(data_cfg["profile_priors"]), priors)
    snapshots = build_profile_snapshots(games, priors)
    snapshots_path = write_profile_parquet(snapshots, data_cfg["profile_snapshots"])
    build_manifest = {
        "version": 2,
        "experiment": "rlcs_player_matchup_value_v2",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "opened_stages": sorted(stages),
        "sealed_test_loaded": False,
        "accepted_replays": int(games["replay_id"].nunique()),
        "player_game_rows": int(len(games)),
        "players": int(games["player_id"].nunique()),
        "split_manifest": str(split_path),
        "split_manifest_sha256": file_sha256(split_path),
        "profile_games": str(games_path),
        "profile_games_sha256": file_sha256(games_path),
        "profile_priors": str(priors_path),
        "profile_priors_sha256": file_sha256(priors_path),
        "profile_snapshots": str(snapshots_path),
        "profile_snapshots_sha256": file_sha256(snapshots_path),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
    }
    manifest_path = _atomic_json(output_dir / "profile_build_manifest.json", build_manifest)
    print(f"profiles: {games_path}")
    print(f"snapshots: {snapshots_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
