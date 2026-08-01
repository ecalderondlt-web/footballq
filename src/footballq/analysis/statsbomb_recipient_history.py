"""Chronology-safe pass-recipient prediction from prior player histories."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from footballq.analysis.statsbomb_player_history_signal import (
    START_X_BINS,
    START_Y_BINS,
    Cohort,
    MatchRecord,
    _clock_seconds,
    _query_player_ids,
    _read_json,
    _relevant_match_ids,
    _role_shuffle_map,
    _stable_hash,
    _zone_index,
    active_lineup_player_ids,
    broad_role,
    is_open_play_pass,
    load_cohorts,
    load_match_records,
)
from footballq.data.statsbomb_events import resolve_statsbomb_data_dir
from footballq.repro.splits import load_split_manifest

ROLE_NAMES = ("goalkeeper", "defender", "midfielder", "forward", "unknown")
SPLIT_MANIFEST_KEYS = {
    "train": "train",
    "validation": "val",
    "development_test": "test",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def recipient_cohort_for_match(
    record: MatchRecord,
    cohorts: list[Cohort],
) -> Cohort | None:
    for cohort in cohorts:
        if (
            record.competition_name == cohort.competition_name
            and record.season_name == cohort.season_name
            and (
                cohort.focal_team_name == "*"
                or cohort.focal_team_name
                in {record.home_team_name, record.away_team_name}
            )
        ):
            return cohort
    return None


def _event_sample_id(match_id: str, period: int, event_id: str) -> str:
    return f"{match_id}:{period}:{event_id}"


def _last_scalar_sum(
    history: list[tuple[str, str, int]],
    support_size: int,
    *,
    support_before: str | None = None,
) -> int:
    eligible = [
        row
        for row in history
        if support_before is None or row[0] < support_before
    ]
    return int(sum(value for _date, _match_id, value in eligible[-support_size:]))


def _last_zone_sum(
    history: list[tuple[str, str, np.ndarray]],
    support_size: int,
    *,
    support_before: str | None = None,
) -> np.ndarray:
    total = np.zeros(START_X_BINS * START_Y_BINS, dtype=np.float64)
    eligible = [
        row
        for row in history
        if support_before is None or row[0] < support_before
    ]
    for _date, _match_id, values in eligible[-support_size:]:
        total += values
    return total


def _support_count(
    history: list[tuple[str, str, Any]],
    support_size: int,
    *,
    support_before: str | None = None,
) -> int:
    return min(
        support_size,
        sum(
            support_before is None or match_date < support_before
            for match_date, _match_id, _value in history
        ),
    )


def _lineup_roles(
    lineup_team: dict[str, Any],
    event: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    active = set(active_lineup_player_ids(lineup_team, event))
    roles: dict[str, str] = {}
    positions_by_player: dict[str, str] = {}
    event_period = int(event.get("period", 1))
    event_time = float(event.get("minute", 0)) * 60.0 + float(
        event.get("second", 0)
    )
    for player in lineup_team.get("lineup") or []:
        player_id = str(player["player_id"])
        if player_id not in active:
            continue
        positions = player.get("positions") or []
        selected_position = "unknown"
        for position in positions:
            from_period = int(position.get("from_period") or 1)
            to_period_raw = position.get("to_period")
            to_period = int(to_period_raw) if to_period_raw is not None else 99
            if not from_period <= event_period <= to_period:
                continue
            if (
                event_period == from_period
                and event_time < _clock_seconds(position.get("from"))
            ):
                continue
            if (
                to_period_raw is not None
                and event_period == to_period
                and position.get("to") is not None
                and event_time >= _clock_seconds(position.get("to"))
            ):
                continue
            selected_position = str(position.get("position") or "unknown")
            break
        positions_by_player[player_id] = selected_position
        roles[player_id] = broad_role(selected_position)
    return roles, positions_by_player


def _appeared_player_ids_by_team(
    lineups: list[dict[str, Any]],
    eligible_player_ids: set[str],
) -> dict[str, list[str]]:
    appeared: dict[str, list[str]] = {}
    for team in lineups:
        team_name = str(team.get("team_name") or "")
        appeared[team_name] = sorted(
            str(player["player_id"])
            for player in team.get("lineup") or []
            if (
                str(player.get("player_id")) in eligible_player_ids
                and bool(player.get("positions"))
            )
        )
    return appeared


def _freeze_event_ids(path: Path) -> tuple[set[str], bool]:
    try:
        rows = _read_json(path)
    except json.JSONDecodeError:
        return set(), True
    return {str(row["event_uuid"]) for row in rows}, False


def _history_components(
    *,
    candidates: list[str],
    actor_id: str,
    start_zone: int,
    support_size: int,
    receiver_history: dict[str, list[tuple[str, str, int]]],
    zone_history: dict[str, list[tuple[str, str, np.ndarray]]],
    pair_history: dict[tuple[str, str], list[tuple[str, str, int]]],
    support_before: str | None = None,
) -> dict[str, list[float]]:
    return {
        "global": [
            float(
                _last_scalar_sum(
                    receiver_history[player_id],
                    support_size,
                    support_before=support_before,
                )
            )
            for player_id in candidates
        ],
        "zone": [
            float(
                _last_zone_sum(
                    zone_history[player_id],
                    support_size,
                    support_before=support_before,
                )[start_zone]
            )
            for player_id in candidates
        ],
        "pair": [
            float(
                _last_scalar_sum(
                    pair_history[(actor_id, player_id)],
                    support_size,
                    support_before=support_before,
                )
            )
            for player_id in candidates
        ],
        "appearance_count": [
            float(
                _support_count(
                    receiver_history[player_id],
                    support_size,
                    support_before=support_before,
                )
            )
            for player_id in candidates
        ],
        "pair_coappearance_count": [
            float(
                _support_count(
                    pair_history[(actor_id, player_id)],
                    support_size,
                    support_before=support_before,
                )
            )
            for player_id in candidates
        ],
    }


def build_development_cache(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_dir = resolve_statsbomb_data_dir(config["data"]["statsbomb_root"])
    records = load_match_records(data_dir)
    cohorts = load_cohorts(config)
    cohort_config = {
        str(row["name"]): row
        for row in config["development_cohorts"]
    }
    support_policy = str(config["profiles"]["support_policy"])
    if support_policy not in {
        "online_strictly_prior_appearances",
        "cohort_frozen_before_start",
    }:
        raise ValueError(f"Unknown player-history support policy: {support_policy}")
    split_manifest = load_split_manifest(config["data"]["split_manifest"])
    query_records: dict[str, Cohort] = {
        record.match_id: cohort
        for record in records
        if (cohort := recipient_cohort_for_match(record, cohorts)) is not None
    }
    actual_split_ids = {
        manifest_key: sorted(
            match_id
            for match_id, cohort in query_records.items()
            if SPLIT_MANIFEST_KEYS[cohort.split] == manifest_key
        )
        for manifest_key in ("train", "val", "test")
    }
    expected_split_ids = {
        "train": sorted(split_manifest.train_match_ids),
        "val": sorted(split_manifest.val_match_ids),
        "test": sorted(split_manifest.test_match_ids),
    }
    if actual_split_ids != expected_split_ids:
        raise ValueError(
            "Configured StatsBomb cohorts do not match the immutable split manifest."
        )
    if any(cohort.focal_team_name == "*" for cohort in cohorts):
        query_player_ids = set()
        for match_id, cohort in query_records.items():
            for team in _read_json(data_dir / "lineups" / f"{match_id}.json"):
                if (
                    cohort.focal_team_name == "*"
                    or str(team.get("team_name") or "")
                    == cohort.focal_team_name
                ):
                    query_player_ids.update(
                        str(player["player_id"])
                        for player in team.get("lineup") or []
                    )
    else:
        query_player_ids = _query_player_ids(data_dir, records, cohorts)
    relevant_match_ids = _relevant_match_ids(
        data_dir,
        records,
        query_player_ids,
    )
    max_query_date = max(
        record.match_date
        for record in records
        if record.match_id in query_records
    )
    by_date: dict[str, list[MatchRecord]] = defaultdict(list)
    for record in records:
        if record.match_date > max_query_date:
            continue
        if record.match_id in relevant_match_ids or record.match_id in query_records:
            by_date[record.match_date].append(record)

    receiver_history: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    zone_history: dict[str, list[tuple[str, str, np.ndarray]]] = defaultdict(list)
    pair_history: dict[tuple[str, str], list[tuple[str, str, int]]] = defaultdict(list)
    support_sizes = [int(value) for value in config["profiles"]["support_sizes"]]
    require_360 = bool(config["data"]["require_360_query"])
    rows: list[dict[str, Any]] = []
    malformed_query_360: list[str] = []
    cohort_match_counts: dict[str, int] = defaultdict(int)
    broad_shuffle_assignments = 0
    fine_shuffle_assignments = 0
    candidate_assignments = 0

    for match_date in sorted(by_date):
        for record in by_date[match_date]:
            cohort = query_records.get(record.match_id)
            if cohort is None:
                continue
            events = _read_json(data_dir / "events" / f"{record.match_id}.json")
            lineups = _read_json(data_dir / "lineups" / f"{record.match_id}.json")
            lineup_by_team = {
                str(team.get("team_name") or ""): team
                for team in lineups
            }
            freeze_ids: set[str] | None = None
            if require_360:
                freeze_ids, malformed = _freeze_event_ids(
                    data_dir / "three-sixty" / f"{record.match_id}.json"
                )
                if malformed:
                    malformed_query_360.append(record.match_id)
                    continue
            cohort_match_counts[cohort.name] += 1
            for event in events:
                event_team_name = str(
                    (event.get("team") or {}).get("name") or ""
                )
                if (
                    (
                        cohort.focal_team_name != "*"
                        and event_team_name != cohort.focal_team_name
                    )
                    or not is_open_play_pass(event)
                    or (freeze_ids is not None and str(event["id"]) not in freeze_ids)
                ):
                    continue
                pass_payload = event.get("pass") or {}
                recipient = pass_payload.get("recipient") or {}
                recipient_id = str(recipient.get("id") or "")
                actor_id = str((event.get("player") or {}).get("id") or "")
                if not recipient_id or not actor_id:
                    continue
                lineup_team = lineup_by_team.get(event_team_name, {})
                candidate_roles, candidate_positions = _lineup_roles(
                    lineup_team,
                    event,
                )
                candidates = sorted(
                    player_id
                    for player_id in candidate_roles
                    if player_id != actor_id
                )
                if recipient_id not in candidates or len(candidates) < 2:
                    continue
                start_zone = _zone_index(
                    event["location"],
                    x_bins=START_X_BINS,
                    y_bins=START_Y_BINS,
                )
                support_before = None
                if support_policy == "cohort_frozen_before_start":
                    support_before_raw = cohort_config[cohort.name].get(
                        "support_before"
                    )
                    if not support_before_raw:
                        raise ValueError(
                            f"Cohort {cohort.name!r} requires support_before "
                            "under cohort_frozen_before_start."
                        )
                    support_before = str(support_before_raw)
                shuffle_map = _role_shuffle_map(
                    {
                        role: {
                            player_id
                            for player_id, candidate_role in candidate_roles.items()
                            if player_id in candidates and candidate_role == role
                        }
                        for role in ROLE_NAMES
                    },
                    seed=(
                        f"{config['evaluation']['shuffle_seed']}:"
                        f"{record.match_id}:{event['id']}"
                    ),
                )
                position_shuffle_map = _role_shuffle_map(
                    {
                        position: {
                            player_id
                            for player_id, candidate_position in (
                                candidate_positions.items()
                            )
                            if (
                                player_id in candidates
                                and candidate_position == position
                            )
                        }
                        for position in sorted(set(candidate_positions.values()))
                    },
                    seed=(
                        f"{config['evaluation']['shuffle_seed']}:fine:"
                        f"{record.match_id}:{event['id']}"
                    ),
                )
                candidate_assignments += len(candidates)
                broad_shuffle_assignments += sum(
                    shuffle_map.get(player_id, player_id) != player_id
                    for player_id in candidates
                )
                fine_shuffle_assignments += sum(
                    position_shuffle_map.get(player_id, player_id) != player_id
                    for player_id in candidates
                )
                row: dict[str, Any] = {
                    "match_id": record.match_id,
                    "match_date": record.match_date,
                    "cohort": cohort.name,
                    "split": cohort.split,
                    "event_id": str(event["id"]),
                    "period": int(event.get("period", 1)),
                    "sample_id": _event_sample_id(
                        record.match_id,
                        int(event.get("period", 1)),
                        str(event["id"]),
                    ),
                    "actor_id": actor_id,
                    "recipient_id": recipient_id,
                    "start_zone": start_zone,
                    "under_pressure": bool(event.get("under_pressure")),
                    "candidates": candidates,
                    "candidate_roles": [
                        candidate_roles[player_id]
                        for player_id in candidates
                    ],
                    "candidate_positions": [
                        candidate_positions[player_id]
                        for player_id in candidates
                    ],
                    "target_index": candidates.index(recipient_id),
                    "support": {},
                }
                for support_size in support_sizes:
                    components = _history_components(
                        candidates=candidates,
                        actor_id=actor_id,
                        start_zone=start_zone,
                        support_size=support_size,
                        receiver_history=receiver_history,
                        zone_history=zone_history,
                        pair_history=pair_history,
                        support_before=support_before,
                    )
                    shuffled_candidates = [
                        shuffle_map.get(player_id, player_id)
                        for player_id in candidates
                    ]
                    components["shuffled_zone"] = [
                        float(
                            _last_zone_sum(
                                zone_history[player_id],
                                support_size,
                                support_before=support_before,
                            )[start_zone]
                        )
                        for player_id in shuffled_candidates
                    ]
                    components["shuffled_global"] = [
                        float(
                            _last_scalar_sum(
                                receiver_history[player_id],
                                support_size,
                                support_before=support_before,
                            )
                        )
                        for player_id in shuffled_candidates
                    ]
                    components["shuffled_position_zone"] = [
                        float(
                            _last_zone_sum(
                                zone_history[
                                    position_shuffle_map.get(
                                        player_id,
                                        player_id,
                                    )
                                ],
                                support_size,
                                support_before=support_before,
                            )[start_zone]
                        )
                        for player_id in candidates
                    ]
                    components["shuffled_position_global"] = [
                        float(
                            _last_scalar_sum(
                                receiver_history[
                                    position_shuffle_map.get(
                                        player_id,
                                        player_id,
                                    )
                                ],
                                support_size,
                                support_before=support_before,
                            )
                        )
                        for player_id in candidates
                    ]
                    row["support"][support_size] = components
                rows.append(row)

        # Same-day matches cannot enter one another's support.
        for record in by_date[match_date]:
            if record.match_id not in relevant_match_ids:
                continue
            events = _read_json(data_dir / "events" / f"{record.match_id}.json")
            lineups = _read_json(data_dir / "lineups" / f"{record.match_id}.json")
            appeared_by_team = _appeared_player_ids_by_team(
                lineups,
                query_player_ids,
            )
            receiver_counts: dict[str, int] = defaultdict(int)
            receiver_zones: dict[str, np.ndarray] = defaultdict(
                lambda: np.zeros(
                    START_X_BINS * START_Y_BINS,
                    dtype=np.float64,
                )
            )
            pair_counts: dict[tuple[str, str], int] = defaultdict(int)
            for event in events:
                if not is_open_play_pass(event):
                    continue
                pass_payload = event.get("pass") or {}
                recipient = pass_payload.get("recipient") or {}
                recipient_id = str(recipient.get("id") or "")
                actor_id = str((event.get("player") or {}).get("id") or "")
                if not recipient_id or recipient_id not in query_player_ids:
                    continue
                receiver_counts[recipient_id] += 1
                start_zone = _zone_index(
                    event["location"],
                    x_bins=START_X_BINS,
                    y_bins=START_Y_BINS,
                )
                receiver_zones[recipient_id][start_zone] += 1.0
                if actor_id in query_player_ids:
                    pair_counts[(actor_id, recipient_id)] += 1
            appeared_player_ids = {
                player_id
                for team_player_ids in appeared_by_team.values()
                for player_id in team_player_ids
            }
            for player_id in sorted(appeared_player_ids):
                receiver_history[player_id].append(
                    (
                        match_date,
                        record.match_id,
                        receiver_counts.get(player_id, 0),
                    )
                )
                zone_history[player_id].append(
                    (
                        match_date,
                        record.match_id,
                        receiver_zones[player_id],
                    )
                )
            for team_player_ids in appeared_by_team.values():
                for actor_id in team_player_ids:
                    for recipient_id in team_player_ids:
                        if actor_id == recipient_id:
                            continue
                        pair = (actor_id, recipient_id)
                        pair_history[pair].append(
                            (
                                match_date,
                                record.match_id,
                                pair_counts.get(pair, 0),
                            )
                        )

    if not rows:
        raise ValueError("No pass-recipient development examples were built.")
    audit = {
        "source_root": str(data_dir),
        "source_commit": str(config["data"]["source_commit"]),
        "config_sha256": _stable_hash(config),
        **split_manifest.metadata(),
        "query_requires_360": require_360,
        "development_examples": len(rows),
        "query_roster_players": len(query_player_ids),
        "relevant_support_matches": len(relevant_match_ids),
        "support_max_date": max_query_date,
        "cohort_match_counts": dict(sorted(cohort_match_counts.items())),
        "split_example_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "development_test")
        },
        "query_match_ids_by_split": {
            key: value
            for key, value in actual_split_ids.items()
        },
        "sample_identity": "match_id:period:event_uuid",
        "feature_view": str(config["feature_view"]),
        "objective_mode": str(config["objective_mode"]),
        "mean_candidate_count": float(
            np.mean([len(row["candidates"]) for row in rows])
        ),
        "broad_role_shuffle_assignment_fraction": (
            broad_shuffle_assignments / candidate_assignments
        ),
        "fine_position_shuffle_assignment_fraction": (
            fine_shuffle_assignments / candidate_assignments
        ),
        "malformed_query_360_match_ids": sorted(malformed_query_360),
        "support_sizes": support_sizes,
        "support_policy": support_policy,
        "cohort_support_before": {
            name: row.get("support_before")
            for name, row in cohort_config.items()
        },
        "chronology_rule": str(config["profiles"]["chronology_rule"]),
        "support_unit": (
            "player appearance, including zero-reception matches; "
            "pair support uses same-team co-appearance"
        ),
        "sealed_test_loaded": False,
    }
    return {"rows": rows, "audit": audit}, audit


def _normalize(values: list[float] | np.ndarray, alpha: float = 1.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64) + float(alpha)
    return array / array.sum()


def _fit_train_priors(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    role_zone_selected: dict[tuple[int, str], int] = defaultdict(int)
    position_zone_selected: dict[tuple[int, str], int] = defaultdict(int)
    position_selected: dict[str, int] = defaultdict(int)
    static_selected: dict[str, int] = defaultdict(int)
    static_available: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["split"] != "train":
            continue
        for player_id in row["candidates"]:
            static_available[player_id] += 1
        recipient_id = row["recipient_id"]
        target_role = row["candidate_roles"][row["target_index"]]
        target_position = row["candidate_positions"][row["target_index"]]
        role_zone_selected[(int(row["start_zone"]), target_role)] += 1
        position_zone_selected[(int(row["start_zone"]), target_position)] += 1
        position_selected[target_position] += 1
        static_selected[recipient_id] += 1
    return {
        "role_zone_selected": role_zone_selected,
        "position_zone_selected": position_zone_selected,
        "position_selected": position_selected,
        "static_selected": static_selected,
        "static_available": static_available,
    }


def _role_probabilities(
    row: dict[str, Any],
    priors: dict[str, Any],
) -> np.ndarray:
    role_counts: dict[str, int] = defaultdict(int)
    for role in row["candidate_roles"]:
        role_counts[role] += 1
    scores = []
    for role in row["candidate_roles"]:
        selected = priors["role_zone_selected"].get(
            (int(row["start_zone"]), role),
            0,
        )
        scores.append((selected + 1.0) / role_counts[role])
    return _normalize(scores, alpha=0.0)


def _static_probabilities(
    row: dict[str, Any],
    priors: dict[str, Any],
) -> np.ndarray:
    scores = [
        (
            priors["static_selected"].get(player_id, 0) + 1.0
        )
        / (
            priors["static_available"].get(player_id, 0) + 2.0
        )
        for player_id in row["candidates"]
    ]
    return _normalize(scores, alpha=0.0)


def _position_probabilities(
    row: dict[str, Any],
    priors: dict[str, Any],
) -> np.ndarray:
    position_counts: dict[str, int] = defaultdict(int)
    for position in row["candidate_positions"]:
        position_counts[position] += 1
    scores = []
    for position in row["candidate_positions"]:
        selected = priors["position_zone_selected"].get(
            (int(row["start_zone"]), position),
            0,
        )
        scores.append((selected + 1.0) / position_counts[position])
    return _normalize(scores, alpha=0.0)


def _profile_likelihood_probabilities(
    row: dict[str, Any],
    priors: dict[str, Any],
    components: dict[str, list[float]],
    *,
    prior_strength: float,
    shuffled_zone: bool,
    shuffled_position_zone: bool,
) -> np.ndarray:
    if shuffled_zone:
        zone_values = components["shuffled_zone"]
        global_values = components["shuffled_global"]
    elif shuffled_position_zone:
        zone_values = components["shuffled_position_zone"]
        global_values = components["shuffled_position_global"]
    else:
        zone_values = components["zone"]
        global_values = components["global"]
    ratios = []
    zone_count = START_X_BINS * START_Y_BINS
    for position, observed_zone, observed_global in zip(
        row["candidate_positions"],
        zone_values,
        global_values,
        strict=True,
    ):
        selected = priors["position_zone_selected"].get(
            (int(row["start_zone"]), position),
            0,
        )
        total = priors["position_selected"].get(position, 0)
        prior_rate = (selected + 1.0) / (total + zone_count)
        posterior_rate = (
            float(observed_zone) + float(prior_strength) * prior_rate
        ) / (float(observed_global) + float(prior_strength))
        ratios.append(posterior_rate / prior_rate)
    return _normalize(ratios, alpha=0.0)


def _row_probabilities(
    row: dict[str, Any],
    priors: dict[str, Any],
    *,
    support_size: int,
    position_weight: float = 0.0,
    static_weight: float = 0.0,
    global_weight: float = 0.0,
    zone_weight: float = 0.0,
    profile_prior_strength: float = 10.0,
    pair_weight: float = 0.0,
    shuffled_zone: bool = False,
    shuffled_position_zone: bool = False,
) -> np.ndarray:
    components = row["support"][support_size]
    log_score = np.log(_role_probabilities(row, priors))
    weighted = (
        (position_weight, _position_probabilities(row, priors)),
        (static_weight, _static_probabilities(row, priors)),
        (global_weight, _normalize(components["global"])),
        (
            zone_weight,
            _profile_likelihood_probabilities(
                row,
                priors,
                components,
                prior_strength=profile_prior_strength,
                shuffled_zone=shuffled_zone,
                shuffled_position_zone=shuffled_position_zone,
            ),
        ),
        (pair_weight, _normalize(components["pair"])),
    )
    for weight, probabilities in weighted:
        if weight:
            log_score += float(weight) * np.log(probabilities)
    log_score -= float(log_score.max())
    probabilities = np.exp(log_score)
    return probabilities / probabilities.sum()


def _ranking_metrics(
    rows: list[dict[str, Any]],
    probabilities: list[np.ndarray],
) -> dict[str, Any]:
    losses: list[float] = []
    top1: list[float] = []
    top3: list[float] = []
    reciprocal_ranks: list[float] = []
    for row, probability in zip(rows, probabilities, strict=True):
        target = int(row["target_index"])
        order = np.argsort(-probability)
        losses.append(-math.log(max(float(probability[target]), 1e-12)))
        top1.append(float(order[0] == target))
        top3.append(float(target in order[:3]))
        reciprocal_ranks.append(1.0 / (int(np.where(order == target)[0][0]) + 1))
    return {
        "examples": len(rows),
        "nll": float(np.mean(losses)),
        "top1_accuracy": float(np.mean(top1)),
        "top3_accuracy": float(np.mean(top3)),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
    }


def _evaluate_condition(
    rows: list[dict[str, Any]],
    priors: dict[str, Any],
    condition: dict[str, Any],
) -> tuple[dict[str, Any], list[np.ndarray]]:
    probabilities = [
        _row_probabilities(row, priors, **condition)
        for row in rows
    ]
    return _ranking_metrics(rows, probabilities), probabilities


def _select_weight(
    validation_rows: list[dict[str, Any]],
    priors: dict[str, Any],
    base_condition: dict[str, Any],
    *,
    parameter: str,
    values: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = []
    for value in values:
        condition = {**base_condition, parameter: float(value)}
        metrics, _probabilities = _evaluate_condition(
            validation_rows,
            priors,
            condition,
        )
        candidates.append((metrics["nll"], float(value), condition, metrics))
    _nll, selected_value, condition, metrics = min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return condition, {
        "selected_value": selected_value,
        "validation_metrics": metrics,
        "candidates": [
            {"value": value, "metrics": candidate_metrics}
            for _candidate_nll, value, _condition, candidate_metrics in candidates
        ],
    }


def _match_bootstrap_gain(
    rows: list[dict[str, Any]],
    baseline_probabilities: list[np.ndarray],
    profile_probabilities: list[np.ndarray],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_match: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_match[str(row["match_id"])].append(index)
    match_ids = sorted(by_match)
    rng = random.Random(seed)
    nll_gains: list[float] = []
    top3_gains: list[float] = []
    for _ in range(samples):
        selected_matches = [rng.choice(match_ids) for _ in match_ids]
        indices = [
            index
            for match_id in selected_matches
            for index in by_match[match_id]
        ]
        selected_rows = [rows[index] for index in indices]
        baseline = _ranking_metrics(
            selected_rows,
            [baseline_probabilities[index] for index in indices],
        )
        profile = _ranking_metrics(
            selected_rows,
            [profile_probabilities[index] for index in indices],
        )
        nll_gains.append(baseline["nll"] - profile["nll"])
        top3_gains.append(
            profile["top3_accuracy"] - baseline["top3_accuracy"]
        )
    return {
        "samples": samples,
        "match_count": len(match_ids),
        "nll_improvement": {
            "mean": float(np.mean(nll_gains)),
            "ci95": [
                float(np.quantile(nll_gains, 0.025)),
                float(np.quantile(nll_gains, 0.975)),
            ],
            "positive_fraction": float(np.mean(np.asarray(nll_gains) > 0.0)),
        },
        "top3_accuracy_gain": {
            "mean": float(np.mean(top3_gains)),
            "ci95": [
                float(np.quantile(top3_gains, 0.025)),
                float(np.quantile(top3_gains, 0.975)),
            ],
            "positive_fraction": float(np.mean(np.asarray(top3_gains) > 0.0)),
        },
    }


def evaluate_development_cache(
    cache: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = cache["rows"]
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "development_test")
    }
    priors = _fit_train_priors(rows)
    weights = [float(value) for value in config["model_selection"]["weights"]]
    support_sizes = [int(value) for value in config["profiles"]["support_sizes"]]
    validation_rows = split_rows["validation"]
    development_rows = split_rows["development_test"]

    role_condition = {
        "support_size": support_sizes[0],
    }
    position_condition, position_selection = _select_weight(
        validation_rows,
        priors,
        role_condition,
        parameter="position_weight",
        values=weights,
    )
    static_condition, static_selection = _select_weight(
        validation_rows,
        priors,
        position_condition,
        parameter="static_weight",
        values=weights,
    )

    rolling_candidates = []
    for support_size in support_sizes:
        condition, selection = _select_weight(
            validation_rows,
            priors,
            {**static_condition, "support_size": support_size},
            parameter="global_weight",
            values=weights,
        )
        rolling_candidates.append(
            (
                selection["validation_metrics"]["nll"],
                support_size,
                condition,
                selection,
            )
        )
    (
        _rolling_nll,
        selected_support,
        rolling_condition,
        rolling_selection,
    ) = min(rolling_candidates, key=lambda item: (item[0], item[1]))
    profile_prior_strengths = [
        float(value)
        for value in config["model_selection"]["profile_prior_strengths"]
    ]
    profile_candidates = []
    for prior_strength in profile_prior_strengths:
        condition, selection = _select_weight(
            validation_rows,
            priors,
            {
                **rolling_condition,
                "profile_prior_strength": prior_strength,
            },
            parameter="zone_weight",
            values=weights,
        )
        profile_candidates.append(
            (
                selection["validation_metrics"]["nll"],
                prior_strength,
                condition,
                selection,
            )
        )
    (
        _profile_nll,
        selected_profile_prior_strength,
        profile_condition,
        profile_selection,
    ) = min(
        profile_candidates,
        key=lambda item: (item[0], item[1]),
    )
    pair_condition, pair_selection = _select_weight(
        validation_rows,
        priors,
        profile_condition,
        parameter="pair_weight",
        values=weights,
    )
    shuffled_condition = {**profile_condition, "shuffled_zone": True}
    shuffled_position_condition = {
        **profile_condition,
        "shuffled_position_zone": True,
    }

    conditions = {
        "A_broad_role_prior": role_condition,
        "B_fine_position_prior": position_condition,
        "D_static_identity": static_condition,
        "E_rolling_involvement": rolling_condition,
        "F_history_target_by_origin_zone": profile_condition,
        "G_history_plus_pair": pair_condition,
        "same_broad_role_shuffled_history": shuffled_condition,
        "same_fine_position_shuffled_history": shuffled_position_condition,
    }
    condition_metrics: dict[str, Any] = {}
    condition_probabilities: dict[str, dict[str, list[np.ndarray]]] = {}
    for name, condition in conditions.items():
        condition_metrics[name] = {}
        condition_probabilities[name] = {}
        for split in ("validation", "development_test"):
            metrics, probabilities = _evaluate_condition(
                split_rows[split],
                priors,
                condition,
            )
            condition_metrics[name][split] = metrics
            condition_probabilities[name][split] = probabilities

    support_size_curve: dict[str, Any] = {}
    for support_size in support_sizes:
        rolling_at_k = {**rolling_condition, "support_size": support_size}
        profile_at_k = {**profile_condition, "support_size": support_size}
        support_size_curve[str(support_size)] = {}
        for split in ("validation", "development_test"):
            rolling_metrics_at_k, _ = _evaluate_condition(
                split_rows[split],
                priors,
                rolling_at_k,
            )
            profile_metrics_at_k, _ = _evaluate_condition(
                split_rows[split],
                priors,
                profile_at_k,
            )
            support_size_curve[str(support_size)][split] = {
                "rolling": rolling_metrics_at_k,
                "profile": profile_metrics_at_k,
                "profile_minus_rolling_nll_improvement": (
                    rolling_metrics_at_k["nll"]
                    - profile_metrics_at_k["nll"]
                ),
                "profile_minus_rolling_top3_gain": (
                    profile_metrics_at_k["top3_accuracy"]
                    - rolling_metrics_at_k["top3_accuracy"]
                ),
            }

    baseline_metrics = condition_metrics["E_rolling_involvement"]
    profile_metrics = condition_metrics["F_history_target_by_origin_zone"]
    shuffle_metrics = condition_metrics["same_broad_role_shuffled_history"]
    fine_shuffle_metrics = condition_metrics[
        "same_fine_position_shuffled_history"
    ]
    gains = {}
    for split in ("validation", "development_test"):
        gains[split] = {
            "profile_minus_rolling_nll_improvement": (
                baseline_metrics[split]["nll"] - profile_metrics[split]["nll"]
            ),
            "profile_minus_rolling_top1_gain": (
                profile_metrics[split]["top1_accuracy"]
                - baseline_metrics[split]["top1_accuracy"]
            ),
            "profile_minus_rolling_top3_gain": (
                profile_metrics[split]["top3_accuracy"]
                - baseline_metrics[split]["top3_accuracy"]
            ),
            "profile_minus_shuffle_nll_improvement": (
                shuffle_metrics[split]["nll"] - profile_metrics[split]["nll"]
            ),
            "profile_minus_shuffle_top3_gain": (
                profile_metrics[split]["top3_accuracy"]
                - shuffle_metrics[split]["top3_accuracy"]
            ),
            "profile_minus_fine_position_shuffle_nll_improvement": (
                fine_shuffle_metrics[split]["nll"]
                - profile_metrics[split]["nll"]
            ),
            "profile_minus_fine_position_shuffle_top3_gain": (
                profile_metrics[split]["top3_accuracy"]
                - fine_shuffle_metrics[split]["top3_accuracy"]
            ),
        }

    bootstrap = _match_bootstrap_gain(
        development_rows,
        condition_probabilities["E_rolling_involvement"]["development_test"],
        condition_probabilities["F_history_target_by_origin_zone"][
            "development_test"
        ],
        samples=int(config["evaluation"]["match_bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    result = {
        "development_only": True,
        "sealed_test_loaded": False,
        "selected_support_size": selected_support,
        "conditions": condition_metrics,
        "support_size_curve": support_size_curve,
        "incremental_gains": gains,
        "match_bootstrap_profile_vs_rolling": bootstrap,
        "model_selection": {
            "position": position_selection,
            "static": static_selection,
            "rolling": {
                **rolling_selection,
                "selected_support_size": selected_support,
                "support_candidates": [
                    {
                        "support_size": support_size,
                        "validation_metrics": selection["validation_metrics"],
                    }
                    for _nll, support_size, _condition, selection in rolling_candidates
                ],
            },
            "profile": profile_selection,
            "profile_prior_strength": {
                "selected_value": selected_profile_prior_strength,
                "candidates": [
                    {
                        "value": prior_strength,
                        "validation_metrics": selection["validation_metrics"],
                    }
                    for (
                        _nll,
                        prior_strength,
                        _condition,
                        selection,
                    ) in profile_candidates
                ],
            },
            "pair": pair_selection,
        },
        "provenance": {
            "config_sha256": cache["audit"]["config_sha256"],
            "split_manifest_path": cache["audit"]["split_manifest_path"],
            "split_manifest_sha256": cache["audit"]["split_manifest_sha256"],
            "source_commit": cache["audit"]["source_commit"],
            "sample_identity": cache["audit"]["sample_identity"],
            "feature_view": cache["audit"]["feature_view"],
            "objective_mode": cache["audit"]["objective_mode"],
            "support_policy": cache["audit"]["support_policy"],
            "chronology_rule": cache["audit"]["chronology_rule"],
        },
        "claim_status": "development_diagnostic_only",
    }
    development_gain = gains["development_test"]
    development_baseline_nll = baseline_metrics["development_test"]["nll"]
    gate_config = config["gates"]
    gate_checks = {
        "minimum_nll_improvement_fraction": (
            development_gain["profile_minus_rolling_nll_improvement"]
            / development_baseline_nll
            >= float(gate_config["minimum_nll_improvement_fraction"])
        ),
        "minimum_top3_gain": (
            development_gain["profile_minus_rolling_top3_gain"]
            >= float(gate_config["minimum_top3_gain"])
        ),
        "positive_match_bootstrap_mean": (
            bootstrap["nll_improvement"]["mean"] > 0.0
            if bool(gate_config["require_positive_match_bootstrap_mean"])
            else True
        ),
        "positive_match_bootstrap_ci": (
            bootstrap["nll_improvement"]["ci95"][0] > 0.0
            if bool(gate_config.get("require_positive_match_bootstrap_ci", False))
            else True
        ),
        "profile_better_than_same_role_shuffle": (
            development_gain["profile_minus_shuffle_nll_improvement"] > 0.0
            if bool(gate_config["require_profile_better_than_same_role_shuffle"])
            else True
        ),
    }
    result["development_gate"] = {
        "status": (
            "controls_passed"
            if all(gate_checks.values())
            else "blocked"
        ),
        "checks": gate_checks,
        "thresholds": gate_config,
        "note": (
            "This is an opened development gate and cannot support a "
            "confirmatory paper claim by itself."
        ),
    }
    result["result_payload_sha256"] = _stable_hash(result)
    return result


def frozen_recipient_conditions(
    development_result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Reconstruct the validation-selected conditions without retuning."""

    selection = development_result["model_selection"]
    support_size = int(development_result["selected_support_size"])
    rolling = {
        "support_size": support_size,
        "position_weight": float(selection["position"]["selected_value"]),
        "static_weight": float(selection["static"]["selected_value"]),
        "global_weight": float(selection["rolling"]["selected_value"]),
    }
    profile = {
        **rolling,
        "zone_weight": float(selection["profile"]["selected_value"]),
        "profile_prior_strength": float(
            selection["profile_prior_strength"]["selected_value"]
        ),
    }
    return {
        "rolling": rolling,
        "profile": profile,
        "same_broad_role_shuffled_profile": {
            **profile,
            "shuffled_zone": True,
        },
        "same_fine_position_shuffled_profile": {
            **profile,
            "shuffled_position_zone": True,
        },
    }


def _history_coverage(
    rows: list[dict[str, Any]],
    support_size: int,
) -> dict[str, Any]:
    target_counts = np.asarray(
        [
            row["support"][support_size]["appearance_count"][
                int(row["target_index"])
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    candidate_counts = np.asarray(
        [
            count
            for row in rows
            for count in row["support"][support_size]["appearance_count"]
        ],
        dtype=np.float64,
    )
    return {
        "query_examples": len(rows),
        "target_has_any_prior_match_fraction": float(
            np.mean(target_counts >= 1.0)
        ),
        "target_has_full_support_fraction": float(
            np.mean(target_counts >= support_size)
        ),
        "candidate_has_any_prior_match_fraction": float(
            np.mean(candidate_counts >= 1.0)
        ),
        "candidate_has_full_support_fraction": float(
            np.mean(candidate_counts >= support_size)
        ),
        "mean_target_prior_matches": float(np.mean(target_counts)),
        "mean_candidate_prior_matches": float(np.mean(candidate_counts)),
    }


def _frozen_cohort_evaluation(
    rows: list[dict[str, Any]],
    priors: dict[str, Any],
    conditions: dict[str, dict[str, Any]],
    *,
    support_sizes: list[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Frozen recipient cohort has no eligible query rows.")
    metrics: dict[str, Any] = {}
    probabilities: dict[str, list[np.ndarray]] = {}
    for name, condition in conditions.items():
        metrics[name], probabilities[name] = _evaluate_condition(
            rows,
            priors,
            condition,
        )
    rolling = metrics["rolling"]
    profile = metrics["profile"]
    broad_shuffle = metrics["same_broad_role_shuffled_profile"]
    fine_shuffle = metrics["same_fine_position_shuffled_profile"]
    bootstrap = _match_bootstrap_gain(
        rows,
        probabilities["rolling"],
        probabilities["profile"],
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    curve: dict[str, Any] = {}
    for support_size in support_sizes:
        rolling_at_k = {
            **conditions["rolling"],
            "support_size": support_size,
        }
        profile_at_k = {
            **conditions["profile"],
            "support_size": support_size,
        }
        rolling_metrics, _ = _evaluate_condition(
            rows,
            priors,
            rolling_at_k,
        )
        profile_metrics, _ = _evaluate_condition(
            rows,
            priors,
            profile_at_k,
        )
        curve[str(support_size)] = {
            "rolling": rolling_metrics,
            "profile": profile_metrics,
            "profile_minus_rolling_nll_improvement": (
                rolling_metrics["nll"] - profile_metrics["nll"]
            ),
            "profile_minus_rolling_top3_gain": (
                profile_metrics["top3_accuracy"]
                - rolling_metrics["top3_accuracy"]
            ),
            "history_coverage": _history_coverage(rows, support_size),
        }
    return {
        "match_count": len({str(row["match_id"]) for row in rows}),
        "example_count": len(rows),
        "metrics": metrics,
        "effects": {
            "profile_minus_rolling_nll_improvement": (
                rolling["nll"] - profile["nll"]
            ),
            "profile_minus_rolling_relative_nll_improvement": (
                rolling["nll"] - profile["nll"]
            )
            / rolling["nll"],
            "profile_minus_rolling_top1_gain": (
                profile["top1_accuracy"] - rolling["top1_accuracy"]
            ),
            "profile_minus_rolling_top3_gain": (
                profile["top3_accuracy"] - rolling["top3_accuracy"]
            ),
            "profile_minus_broad_role_shuffle_nll_improvement": (
                broad_shuffle["nll"] - profile["nll"]
            ),
            "profile_minus_fine_position_shuffle_nll_improvement": (
                fine_shuffle["nll"] - profile["nll"]
            ),
        },
        "match_bootstrap_profile_vs_rolling": bootstrap,
        "support_size_curve": curve,
    }


def evaluate_frozen_recipient_cache(
    cache: dict[str, Any],
    config: dict[str, Any],
    development_result: dict[str, Any],
) -> dict[str, Any]:
    """Score frozen tournament cohorts without model selection or retuning."""

    rows = cache["rows"]
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "development_test"]
    priors = _fit_train_priors(train_rows)
    conditions = frozen_recipient_conditions(development_result)
    validation_reproduction: dict[str, Any] = {}
    validation_matches = {
        "rolling": "E_rolling_involvement",
        "profile": "F_history_target_by_origin_zone",
    }
    validation_exact = True
    for condition_name, development_name in validation_matches.items():
        metrics, _ = _evaluate_condition(
            validation_rows,
            priors,
            conditions[condition_name],
        )
        expected = development_result["conditions"][development_name][
            "validation"
        ]
        deltas = {
            metric: float(metrics[metric]) - float(expected[metric])
            for metric in (
                "nll",
                "top1_accuracy",
                "top3_accuracy",
                "mean_reciprocal_rank",
            )
        }
        validation_reproduction[condition_name] = {
            "observed": metrics,
            "frozen_expected": expected,
            "deltas": deltas,
        }
        validation_exact &= all(abs(value) <= 1e-12 for value in deltas.values())

    support_sizes = [int(value) for value in config["profiles"]["support_sizes"]]
    bootstrap_samples = int(config["evaluation"]["match_bootstrap_samples"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    cohort_names = [
        str(value) for value in config["confirmatory"]["cohort_order"]
    ]
    by_cohort = {
        cohort_name: _frozen_cohort_evaluation(
            [row for row in test_rows if row["cohort"] == cohort_name],
            priors,
            conditions,
            support_sizes=support_sizes,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=seed + index,
        )
        for index, cohort_name in enumerate(cohort_names)
    }
    pooled = _frozen_cohort_evaluation(
        test_rows,
        priors,
        conditions,
        support_sizes=support_sizes,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=seed + 10_000,
    )
    primary_name = str(config["confirmatory"]["primary_cohort"])
    replication_names = [
        str(value)
        for value in config["confirmatory"]["external_replication_cohorts"]
    ]
    primary = by_cohort[primary_name]
    gate_config = config["confirmatory_gate"]
    checks = {
        "frozen_validation_reproduced_exactly": validation_exact,
        "primary_minimum_relative_nll_improvement": (
            float(
                primary["effects"][
                    "profile_minus_rolling_relative_nll_improvement"
                ]
            )
            >= float(
                gate_config["primary_minimum_relative_nll_improvement"]
            )
        ),
        "primary_positive_match_bootstrap_lower_bound": (
            float(
                primary["match_bootstrap_profile_vs_rolling"][
                    "nll_improvement"
                ]["ci95"][0]
            )
            > 0.0
        ),
        "primary_better_than_broad_role_shuffle": (
            float(
                primary["effects"][
                    "profile_minus_broad_role_shuffle_nll_improvement"
                ]
            )
            > 0.0
        ),
        "pooled_positive_match_bootstrap_lower_bound": (
            float(
                pooled["match_bootstrap_profile_vs_rolling"][
                    "nll_improvement"
                ]["ci95"][0]
            )
            > 0.0
        ),
        "external_replications_positive_point_nll": all(
            float(
                by_cohort[name]["effects"][
                    "profile_minus_rolling_nll_improvement"
                ]
            )
            > 0.0
            for name in replication_names
        ),
        "external_replications_better_than_broad_role_shuffle": all(
            float(
                by_cohort[name]["effects"][
                    "profile_minus_broad_role_shuffle_nll_improvement"
                ]
            )
            > 0.0
            for name in replication_names
        ),
    }
    result = {
        "experiment_protocol": str(config["experiment_protocol"]),
        "status": "confirmatory_metrics_opened_once",
        "claim_boundary": (
            "Strictly prior player receiving-location histories improve "
            "held-out pass-recipient probability ranking beyond current origin "
            "zone, role, fine position, static identity, and rolling reception "
            "frequency. This is an event-choice result, not tracking-based "
            "critical-event or matchup understanding."
        ),
        "frozen_development_result_payload_sha256": str(
            development_result["result_payload_sha256"]
        ),
        "frozen_conditions": conditions,
        "validation_reproduction": validation_reproduction,
        "validation_reproduced_exactly": validation_exact,
        "cohorts": by_cohort,
        "pooled": pooled,
        "gate": {
            "checks": checks,
            "passed": all(checks.values()),
            "thresholds": gate_config,
        },
        "provenance": {
            **cache["audit"],
            "confirmatory_metrics_loaded": True,
        },
    }
    result["result_payload_sha256"] = _stable_hash(result)
    return result
