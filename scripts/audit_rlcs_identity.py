from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from footballq.data.rlcs_ballchasing import read_inventory_parquet
from footballq.repro.manifest import file_sha256


def _matchup_key(roster: list[str]) -> str:
    if len(roster) != 6:
        return ""
    blue = "\n".join(sorted(roster[:3]))
    orange = "\n".join(sorted(roster[3:]))
    teams = sorted([blue, orange])
    return hashlib.sha256("\n---\n".join(teams).encode("utf-8")).hexdigest()


def _test_metrics(test_path: Path) -> dict[str, Any]:
    table = pq.read_table(
        test_path,
        columns=[
            "series_id",
            "score_diff_actor",
            "seconds_remaining",
            "overtime",
            "player_known_mask",
        ],
    )
    frame = table.to_pandas()
    critical = (frame["score_diff_actor"].abs() <= 1) & (
        (frame["seconds_remaining"] <= 120.0) | frame["overtime"]
    )
    critical_frame = frame.loc[critical]
    all_known = critical_frame["player_known_mask"].map(lambda values: all(bool(v) for v in values))
    return {
        "test_samples": int(len(frame)),
        "critical_test_samples": int(critical.sum()),
        "test_series": int(critical_frame["series_id"].nunique()),
        "critical_all_known_samples": int(all_known.sum()),
        "critical_all_known_coverage": (
            float(all_known.mean()) if len(all_known) else 0.0
        ),
    }


def estimate_validation_series_power(
    train_path: Path,
    validation_path: Path,
    *,
    relative_effect: float = 0.05,
    alpha: float = 0.01,
    simulation_trials: int = 1_000,
    sign_flip_permutations: int = 10_000,
    seed: int = 20_250_802,
) -> dict[str, Any]:
    """Estimate paired series power without opening sealed-test targets."""

    if simulation_trials <= 0 or sign_flip_permutations <= 0:
        raise ValueError("Power simulation counts must be positive.")
    train = pq.read_table(
        train_path, columns=["next_touch_entity", "next_touch_zone"]
    ).to_pandas()
    validation = pq.read_table(
        validation_path,
        columns=[
            "series_id",
            "score_diff_actor",
            "seconds_remaining",
            "overtime",
            "player_known_mask",
            "next_touch_entity",
            "next_touch_zone",
        ],
    ).to_pandas()
    entity_counts = np.bincount(
        train["next_touch_entity"].to_numpy(dtype=np.int64), minlength=6
    )
    zone_counts = np.bincount(
        train["next_touch_zone"].to_numpy(dtype=np.int64), minlength=18
    )
    entity_probability = (entity_counts + 1.0) / (len(train) + 6.0)
    zone_probability = (zone_counts + 1.0) / (len(train) + 18.0)
    critical = (validation["score_diff_actor"].abs() <= 1) & (
        (validation["seconds_remaining"] <= 120.0) | validation["overtime"]
    )
    all_known = validation["player_known_mask"].map(
        lambda values: all(bool(value) for value in values)
    )
    primary = validation.loc[critical & all_known].copy()
    if primary.empty:
        raise ValueError("Validation has no critical all-identities-known samples.")
    entity_target = primary["next_touch_entity"].to_numpy(dtype=np.int64)
    zone_target = primary["next_touch_zone"].to_numpy(dtype=np.int64)
    primary["anonymous_joint_nll"] = -np.log(entity_probability[entity_target]) - np.log(
        zone_probability[zone_target]
    )
    series_nll = primary.groupby("series_id", sort=True)["anonymous_joint_nll"].mean()
    if len(series_nll) < 8:
        raise ValueError("Power simulation requires at least eight validation series.")
    baseline_mean = float(series_nll.mean())
    residuals = series_nll.to_numpy(dtype=np.float64) - baseline_mean
    effect_absolute = float(relative_effect) * baseline_mean
    rng = np.random.default_rng(int(seed))
    series_count = len(residuals)
    first = rng.choice(residuals, size=(simulation_trials, series_count), replace=True)
    second = rng.choice(residuals, size=(simulation_trials, series_count), replace=True)
    differences = (effect_absolute + first - second).astype(np.float32)
    signs = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(sign_flip_permutations, series_count),
        replace=True,
    )
    rejections = 0
    batch_size = 50
    for start in range(0, simulation_trials, batch_size):
        batch = differences[start : start + batch_size]
        observed = batch.mean(axis=1)
        permuted = signs @ batch.T / float(series_count)
        exceedances = (permuted >= observed[np.newaxis, :]).sum(axis=0)
        p_values = (exceedances + 1.0) / (sign_flip_permutations + 1.0)
        rejections += int((p_values < float(alpha)).sum())
    estimated_power = rejections / float(simulation_trials)
    return {
        "status": "passed" if estimated_power >= 0.80 else "failed",
        "method": "validation_series_bootstrap_conservative_sign_flip",
        "required_power": 0.80,
        "estimated_power": estimated_power,
        "relative_effect": float(relative_effect),
        "absolute_nll_effect": effect_absolute,
        "alpha": float(alpha),
        "simulation_trials": int(simulation_trials),
        "sign_flip_permutations": int(sign_flip_permutations),
        "seed": int(seed),
        "validation_primary_samples": int(len(primary)),
        "validation_series": int(series_count),
        "train_laplace_anonymous_mean_nll": baseline_mean,
        "validation_series_nll_sd": float(series_nll.std(ddof=1)),
        "assumed_difference_sd": float(np.sqrt(2.0) * series_nll.std(ddof=1)),
        "assumption": (
            "Two independent bootstrap draws from centered validation-series anonymous "
            "NLL residuals; this is conservative for paired models on identical series."
        ),
    }


def audit_identity_dataset(
    *,
    inventory_path: Path,
    quality_report_path: Path,
    dataset_manifest_path: Path,
    power_trials: int = 1_000,
    power_permutations: int = 10_000,
    power_seed: int = 20_250_802,
) -> dict[str, Any]:
    inventory = read_inventory_parquet(inventory_path)
    quality = json.loads(quality_report_path.read_text(encoding="utf-8"))
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    replays = quality["replays"]
    accepted = [row for row in replays if row.get("identity_accepted")]
    games_by_player: dict[str, set[str]] = defaultdict(set)
    matchups_by_split: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        roster = [str(value) for value in row.get("canonical_roster", [])]
        if row.get("split") == "train":
            for player in roster:
                games_by_player[player].add(str(row["replay_id"]))
        key = _matchup_key(roster)
        if key:
            matchups_by_split[str(row["split"])].add(key)
    players_with_history = sum(len(games) >= 20 for games in games_by_player.values())
    repeated_matchups = len(matchups_by_split["train"] & matchups_by_split["test"])
    train_path = Path(dataset["splits"]["train"]["path"])
    validation_path = Path(dataset["splits"]["val"]["path"])
    test_path = Path(dataset["splits"]["test"]["path"])
    power = estimate_validation_series_power(
        train_path,
        validation_path,
        simulation_trials=power_trials,
        sign_flip_permutations=power_permutations,
        seed=power_seed,
    )
    test = _test_metrics(test_path)
    downloaded = sum(
        str(row.get("download_status")) == "complete"
        and bool(row.get("file_sha256"))
        and int(row.get("file_size_bytes") or 0) > 0
        for row in inventory
    )
    parsed = sum(bool(row.get("parse_success")) for row in replays)
    attempted = len(replays)
    qc_accepted = sum(bool(row.get("qc_accepted")) for row in replays)
    identity_accepted = len(accepted)
    total_samples = sum(int(split["rows"]) for split in dataset["splits"].values())
    metrics = {
        "downloaded_hashed_replays": downloaded,
        "inventory_replays": len(inventory),
        "parser_success_rate": parsed / attempted if attempted else 0.0,
        "qc_accepted_replays": qc_accepted,
        "identity_accepted_replays": identity_accepted,
        "unresolved_identity_replay_rate": (
            (qc_accepted - identity_accepted) / qc_accepted if qc_accepted else 1.0
        ),
        "clean_decision_samples": total_samples,
        "players_with_20_earlier_games": players_with_history,
        "repeated_train_test_matchups": repeated_matchups,
        **test,
    }
    gates = {
        "downloaded_replays_at_least_1595": downloaded >= 1595,
        "accepted_replays_at_least_1400": identity_accepted >= 1400,
        "clean_samples_at_least_75000": total_samples >= 75_000,
        "critical_test_samples_at_least_10000": test["critical_test_samples"] >= 10_000,
        "test_series_at_least_80": test["test_series"] >= 80,
        "players_with_history_at_least_30": players_with_history >= 30,
        "repeated_matchups_at_least_10": repeated_matchups >= 10,
        "critical_known_coverage_at_least_60pct": (
            test["critical_all_known_coverage"] >= 0.60
        ),
        "parser_success_at_least_95pct": metrics["parser_success_rate"] >= 0.95,
        "unresolved_identity_rate_at_most_2pct": (
            metrics["unresolved_identity_replay_rate"] <= 0.02
        ),
        "series_power_at_least_80pct": power["estimated_power"] >= 0.80,
    }
    return {
        "version": 1,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "inputs": {
            "inventory_sha256": file_sha256(inventory_path),
            "quality_report_sha256": file_sha256(quality_report_path),
            "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        },
        "metrics": metrics,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "power_gate": power,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RLCS corpus identity and volume gates.")
    parser.add_argument(
        "--inventory", type=Path, default=Path("data/raw/rlcs_2025/replay_inventory.parquet")
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("data/processed/rlcs_identity_matchup_v1/quality_report.json"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("data/processed/rlcs_identity_matchup_v1/dataset_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/rlcs_identity_matchup_v1/identity_audit.json"),
    )
    parser.add_argument("--allow-gate-failure", action="store_true")
    parser.add_argument("--power-trials", type=int, default=1_000)
    parser.add_argument("--power-permutations", type=int, default=10_000)
    parser.add_argument("--power-seed", type=int, default=20_250_802)
    args = parser.parse_args()
    payload = audit_identity_dataset(
        inventory_path=args.inventory,
        quality_report_path=args.quality_report,
        dataset_manifest_path=args.dataset_manifest,
        power_trials=args.power_trials,
        power_permutations=args.power_permutations,
        power_seed=args.power_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"identity_audit: {args.output}")
    print(json.dumps(payload["metrics"], indent=2))
    if not payload["all_gates_pass"] and not args.allow_gate_failure:
        failed = [name for name, passed in payload["gates"].items() if not passed]
        raise SystemExit("Corpus gates failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
