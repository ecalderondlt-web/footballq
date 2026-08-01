"""Low-cost longitudinal player-history diagnostic on StatsBomb 360 passes."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from footballq.analysis.player_profile_proof import _extended_binary_metrics
from footballq.data.statsbomb_events import resolve_statsbomb_data_dir

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
PITCH_DIAGONAL = math.hypot(PITCH_LENGTH, PITCH_WIDTH)
START_X_BINS = 6
START_Y_BINS = 4
END_X_BINS = 6
END_Y_BINS = 4
DELTA_X_EDGES = (-20.0, -5.0, 5.0, 12.0, 25.0, 40.0)
ROLE_NAMES = ("goalkeeper", "defender", "midfielder", "forward", "unknown")
DEAD_BALL_PATTERNS = {
    "From Corner",
    "From Free Kick",
    "From Goal Kick",
    "From Kick Off",
    "From Throw In",
}
DEAD_BALL_PASS_TYPES = {
    "Corner",
    "Free Kick",
    "Goal Kick",
    "Kick Off",
    "Throw-in",
}

CURRENT_FEATURE_NAMES = (
    "start_x_norm",
    "start_y_norm",
    "minute_norm",
    "under_pressure",
    "visible_player_fraction",
    "visible_teammate_fraction",
    "visible_opponent_fraction",
    "nearest_teammate_distance_norm",
    "nearest_opponent_distance_norm",
    "teammates_ahead_fraction",
    "opponents_ahead_fraction",
    "teammate_mean_dx_norm",
    "opponent_mean_dx_norm",
    "teammate_mean_abs_dy_norm",
    "opponent_mean_abs_dy_norm",
    "teammates_in_box_fraction",
    "opponents_goal_side_fraction",
)

ROLLING_PROFILE_FEATURE_NAMES = (
    "profile_available",
    "support_match_count_log",
    "support_pass_count_log",
    "completion_rate",
    "progressive_rate",
    "turnover_5s_rate",
    "box_entry_rate",
    "shot_10s_rate",
    "under_pressure_rate",
    "mean_delta_x_norm",
    "std_delta_x_norm",
    "mean_pass_length_norm",
)


@dataclass(frozen=True)
class MatchRecord:
    match_id: str
    match_date: str
    competition_name: str
    season_name: str
    home_team_name: str
    away_team_name: str
    has_360: bool


@dataclass(frozen=True)
class Cohort:
    name: str
    split: str
    competition_name: str
    season_name: str
    focal_team_name: str


@dataclass
class PassSummary:
    match_count: int
    pass_count: int
    complete_count: int
    progressive_count: int
    turnover_5s_count: int
    box_entry_count: int
    shot_10s_count: int
    under_pressure_count: int
    delta_x_sum: float
    delta_x_square_sum: float
    length_sum: float
    start_zone_count: np.ndarray
    end_zone_count: np.ndarray
    delta_x_bin_count: np.ndarray
    pressure_count: np.ndarray
    pressure_progressive_count: np.ndarray
    pressure_turnover_count: np.ndarray
    start_end_count: np.ndarray
    start_pressure_count: np.ndarray
    start_pressure_progressive_count: np.ndarray
    start_pressure_turnover_count: np.ndarray

    @classmethod
    def empty(cls) -> PassSummary:
        return cls(
            match_count=0,
            pass_count=0,
            complete_count=0,
            progressive_count=0,
            turnover_5s_count=0,
            box_entry_count=0,
            shot_10s_count=0,
            under_pressure_count=0,
            delta_x_sum=0.0,
            delta_x_square_sum=0.0,
            length_sum=0.0,
            start_zone_count=np.zeros(START_X_BINS * START_Y_BINS, dtype=np.float64),
            end_zone_count=np.zeros(END_X_BINS * END_Y_BINS, dtype=np.float64),
            delta_x_bin_count=np.zeros(len(DELTA_X_EDGES) + 1, dtype=np.float64),
            pressure_count=np.zeros(2, dtype=np.float64),
            pressure_progressive_count=np.zeros(2, dtype=np.float64),
            pressure_turnover_count=np.zeros(2, dtype=np.float64),
            start_end_count=np.zeros(
                (START_X_BINS * START_Y_BINS, END_X_BINS * END_Y_BINS),
                dtype=np.float64,
            ),
            start_pressure_count=np.zeros(
                (START_X_BINS * START_Y_BINS, 2),
                dtype=np.float64,
            ),
            start_pressure_progressive_count=np.zeros(
                (START_X_BINS * START_Y_BINS, 2),
                dtype=np.float64,
            ),
            start_pressure_turnover_count=np.zeros(
                (START_X_BINS * START_Y_BINS, 2),
                dtype=np.float64,
            ),
        )

    def add(self, other: PassSummary) -> None:
        for name in (
            "match_count",
            "pass_count",
            "complete_count",
            "progressive_count",
            "turnover_5s_count",
            "box_entry_count",
            "shot_10s_count",
            "under_pressure_count",
        ):
            setattr(self, name, int(getattr(self, name)) + int(getattr(other, name)))
        for name in ("delta_x_sum", "delta_x_square_sum", "length_sum"):
            setattr(self, name, float(getattr(self, name)) + float(getattr(other, name)))
        for name in (
            "start_zone_count",
            "end_zone_count",
            "delta_x_bin_count",
            "pressure_count",
            "pressure_progressive_count",
            "pressure_turnover_count",
            "start_end_count",
            "start_pressure_count",
            "start_pressure_progressive_count",
            "start_pressure_turnover_count",
        ):
            getattr(self, name)[:] += getattr(other, name)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _stable_hash(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_match_records(raw_root: str | Path) -> list[MatchRecord]:
    data_dir = resolve_statsbomb_data_dir(raw_root)
    records: list[MatchRecord] = []
    for path in sorted((data_dir / "matches").rglob("*.json")):
        for match in _read_json(path):
            match_id = str(match["match_id"])
            records.append(
                MatchRecord(
                    match_id=match_id,
                    match_date=str(match["match_date"]),
                    competition_name=str(
                        (match.get("competition") or {}).get("competition_name") or ""
                    ),
                    season_name=str((match.get("season") or {}).get("season_name") or ""),
                    home_team_name=str(
                        (match.get("home_team") or {}).get("home_team_name") or ""
                    ),
                    away_team_name=str(
                        (match.get("away_team") or {}).get("away_team_name") or ""
                    ),
                    has_360=(data_dir / "three-sixty" / f"{match_id}.json").is_file(),
                )
            )
    return sorted(records, key=lambda row: (row.match_date, row.match_id))


def load_cohorts(config: dict[str, Any]) -> list[Cohort]:
    cohorts = [
        Cohort(
            name=str(row["name"]),
            split=str(row["split"]),
            competition_name=str(row["competition_name"]),
            season_name=str(row["season_name"]),
            focal_team_name=str(row["focal_team_name"]),
        )
        for row in config["development_cohorts"]
    ]
    split_names = {cohort.split for cohort in cohorts}
    expected = {"train", "validation", "development_test"}
    if split_names != expected:
        raise ValueError(
            f"Development cohorts must cover {sorted(expected)}; "
            f"got {sorted(split_names)}"
        )
    return cohorts


def cohort_for_match(record: MatchRecord, cohorts: Iterable[Cohort]) -> Cohort | None:
    for cohort in cohorts:
        if (
            record.competition_name == cohort.competition_name
            and record.season_name == cohort.season_name
            and cohort.focal_team_name
            in {record.home_team_name, record.away_team_name}
        ):
            return cohort
    return None


def broad_role(position_name: object) -> str:
    value = str(position_name or "").lower()
    if "goalkeeper" in value:
        return "goalkeeper"
    if any(token in value for token in ("back", "defender")):
        return "defender"
    if any(token in value for token in ("midfield", "wing back")):
        return "midfielder"
    if any(token in value for token in ("forward", "wing", "striker")):
        return "forward"
    return "unknown"


def timestamp_seconds(event: dict[str, Any]) -> float:
    return float(event.get("minute", 0)) * 60.0 + float(event.get("second", 0))


def is_open_play_pass(event: dict[str, Any]) -> bool:
    if str((event.get("type") or {}).get("name") or "") != "Pass":
        return False
    if not event.get("player") or not event.get("location"):
        return False
    payload = event.get("pass") or {}
    if not payload.get("end_location"):
        return False
    if str((payload.get("type") or {}).get("name") or "") in DEAD_BALL_PASS_TYPES:
        return False
    return str((event.get("play_pattern") or {}).get("name") or "") not in DEAD_BALL_PATTERNS


def _zone_index(
    location: list[float] | tuple[float, ...],
    *,
    x_bins: int,
    y_bins: int,
) -> int:
    x = min(max(float(location[0]), 0.0), PITCH_LENGTH - 1e-6)
    y = min(max(float(location[1]), 0.0), PITCH_WIDTH - 1e-6)
    x_index = min(int(x / PITCH_LENGTH * x_bins), x_bins - 1)
    y_index = min(int(y / PITCH_WIDTH * y_bins), y_bins - 1)
    return x_index * y_bins + y_index


def pass_labels(events: list[dict[str, Any]], index: int) -> dict[str, bool | int]:
    event = events[index]
    payload = event["pass"]
    start = event["location"]
    end = payload["end_location"]
    delta_x = float(end[0]) - float(start[0])
    start_in_box = float(start[0]) >= 102.0 and 18.0 <= float(start[1]) <= 62.0
    end_in_box = float(end[0]) >= 102.0 and 18.0 <= float(end[1]) <= 62.0
    current_time = timestamp_seconds(event)
    possession = event.get("possession")
    period = event.get("period")
    team_id = (event.get("team") or {}).get("id")
    turnover = bool(payload.get("outcome"))
    shot = False
    for future in events[index + 1 :]:
        if future.get("period") != period:
            break
        elapsed = timestamp_seconds(future) - current_time
        if elapsed > 10.0:
            break
        if elapsed <= 5.0 and future.get("possession") != possession:
            turnover = True
        if (
            future.get("possession") == possession
            and (future.get("team") or {}).get("id") == team_id
            and str((future.get("type") or {}).get("name") or "") == "Shot"
        ):
            shot = True
    end_lane = min(int(float(end[1]) / PITCH_WIDTH * 3), 2)
    if delta_x < -5.0:
        route = 0
    elif delta_x < 12.0:
        route = 1
    else:
        route = 2 + end_lane
    if turnover:
        pressure_response = 0
    elif delta_x >= 12.0:
        pressure_response = 1
    elif delta_x < -5.0:
        pressure_response = 2
    else:
        pressure_response = 3
    return {
        "complete": not bool(payload.get("outcome")),
        "progressive": delta_x >= 12.0,
        "turnover_5s": turnover,
        "box_entry": (not start_in_box) and end_in_box,
        "shot_10s": shot,
        "route": route,
        "end_zone": _zone_index(
            end,
            x_bins=END_X_BINS,
            y_bins=END_Y_BINS,
        ),
        "pressure_response": pressure_response,
    }


def summarize_player_passes(
    events: list[dict[str, Any]],
    *,
    allowed_player_ids: set[str] | None = None,
) -> dict[str, PassSummary]:
    summaries: dict[str, PassSummary] = {}
    for index, event in enumerate(events):
        if not is_open_play_pass(event):
            continue
        player_id = str((event.get("player") or {}).get("id") or "")
        if not player_id or (
            allowed_player_ids is not None and player_id not in allowed_player_ids
        ):
            continue
        summary = summaries.setdefault(player_id, PassSummary.empty())
        payload = event["pass"]
        start = event["location"]
        end = payload["end_location"]
        labels = pass_labels(events, index)
        delta_x = float(end[0]) - float(start[0])
        delta_y = float(end[1]) - float(start[1])
        pressure_index = int(bool(event.get("under_pressure")))
        start_zone = _zone_index(
            start,
            x_bins=START_X_BINS,
            y_bins=START_Y_BINS,
        )
        end_zone = _zone_index(
            end,
            x_bins=END_X_BINS,
            y_bins=END_Y_BINS,
        )
        summary.pass_count += 1
        summary.complete_count += int(labels["complete"])
        summary.progressive_count += int(labels["progressive"])
        summary.turnover_5s_count += int(labels["turnover_5s"])
        summary.box_entry_count += int(labels["box_entry"])
        summary.shot_10s_count += int(labels["shot_10s"])
        summary.under_pressure_count += pressure_index
        summary.delta_x_sum += delta_x
        summary.delta_x_square_sum += delta_x * delta_x
        summary.length_sum += math.hypot(delta_x, delta_y)
        summary.start_zone_count[start_zone] += 1.0
        summary.end_zone_count[end_zone] += 1.0
        summary.delta_x_bin_count[int(np.digitize(delta_x, DELTA_X_EDGES))] += 1.0
        summary.pressure_count[pressure_index] += 1.0
        summary.pressure_progressive_count[pressure_index] += float(labels["progressive"])
        summary.pressure_turnover_count[pressure_index] += float(labels["turnover_5s"])
        summary.start_end_count[start_zone, end_zone] += 1.0
        summary.start_pressure_count[start_zone, pressure_index] += 1.0
        summary.start_pressure_progressive_count[start_zone, pressure_index] += float(
            labels["progressive"]
        )
        summary.start_pressure_turnover_count[start_zone, pressure_index] += float(
            labels["turnover_5s"]
        )
    for summary in summaries.values():
        summary.match_count = 1
    return summaries


def aggregate_history(
    history: list[tuple[str, str, PassSummary]],
    support_size: int,
) -> PassSummary:
    selected = history[-int(support_size) :]
    aggregate = PassSummary.empty()
    for _date, _match_id, summary in selected:
        aggregate.add(summary)
    return aggregate


def _smoothed_distribution(values: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    denominator = float(values.sum()) + alpha * len(values)
    return (values + alpha) / denominator


def profile_vectors(summary: PassSummary) -> tuple[np.ndarray, np.ndarray]:
    if summary.pass_count == 0:
        rolling = np.zeros(len(ROLLING_PROFILE_FEATURE_NAMES), dtype=np.float32)
        rich_size = (
            len(ROLLING_PROFILE_FEATURE_NAMES)
            + START_X_BINS * START_Y_BINS
            + END_X_BINS * END_Y_BINS
            + len(DELTA_X_EDGES)
            + 1
            + 6
        )
        return rolling, np.zeros(rich_size, dtype=np.float32)
    count = float(summary.pass_count)
    mean_delta_x = summary.delta_x_sum / count
    variance_delta_x = max(
        summary.delta_x_square_sum / count - mean_delta_x * mean_delta_x,
        0.0,
    )
    rolling = np.asarray(
        [
            1.0,
            math.log1p(summary.match_count),
            math.log1p(summary.pass_count),
            summary.complete_count / count,
            summary.progressive_count / count,
            summary.turnover_5s_count / count,
            summary.box_entry_count / count,
            summary.shot_10s_count / count,
            summary.under_pressure_count / count,
            mean_delta_x / PITCH_LENGTH,
            math.sqrt(variance_delta_x) / PITCH_LENGTH,
            summary.length_sum / count / PITCH_DIAGONAL,
        ],
        dtype=np.float32,
    )
    pressure_rates: list[float] = []
    for pressure_index in range(2):
        pressure_count = float(summary.pressure_count[pressure_index])
        pressure_rates.extend(
            [
                math.log1p(pressure_count),
                (float(summary.pressure_progressive_count[pressure_index]) + 1.0)
                / (pressure_count + 2.0),
                (float(summary.pressure_turnover_count[pressure_index]) + 1.0)
                / (pressure_count + 2.0),
            ]
        )
    rich = np.concatenate(
        [
            rolling,
            _smoothed_distribution(summary.start_zone_count),
            _smoothed_distribution(summary.end_zone_count),
            _smoothed_distribution(summary.delta_x_bin_count),
            np.asarray(pressure_rates, dtype=np.float64),
        ]
    ).astype(np.float32)
    return rolling, rich


def query_conditioned_profile_vectors(
    summary: PassSummary,
    start_location: list[float] | tuple[float, ...],
    under_pressure: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rolling, rich = profile_vectors(summary)
    if summary.pass_count == 0:
        return rolling, np.concatenate(
            [
                rich,
                np.zeros(3 + END_X_BINS * END_Y_BINS, dtype=np.float32),
            ]
        )
    start_zone = _zone_index(
        start_location,
        x_bins=START_X_BINS,
        y_bins=START_Y_BINS,
    )
    pressure_index = int(bool(under_pressure))
    context_count = float(summary.start_pressure_count[start_zone, pressure_index])
    global_progressive = float(summary.progressive_count) / float(summary.pass_count)
    global_turnover = float(summary.turnover_5s_count) / float(summary.pass_count)
    shrinkage = 10.0
    conditioned = np.concatenate(
        [
            np.asarray(
                [
                    math.log1p(context_count),
                    (
                        float(
                            summary.start_pressure_progressive_count[
                                start_zone,
                                pressure_index,
                            ]
                        )
                        + shrinkage * global_progressive
                    )
                    / (context_count + shrinkage),
                    (
                        float(
                            summary.start_pressure_turnover_count[
                                start_zone,
                                pressure_index,
                            ]
                        )
                        + shrinkage * global_turnover
                    )
                    / (context_count + shrinkage),
                ],
                dtype=np.float32,
            ),
            _smoothed_distribution(
                summary.start_end_count[start_zone],
                alpha=0.5,
            ).astype(np.float32),
        ]
    )
    return rolling, np.concatenate([rich, conditioned]).astype(np.float32)


def _distance(
    source: tuple[float, float],
    targets: list[tuple[float, float]],
) -> list[float]:
    return [math.hypot(x - source[0], y - source[1]) for x, y in targets]


def freeze_frame_features(
    freeze_frame: list[dict[str, Any]],
    start_location: list[float] | tuple[float, ...],
) -> np.ndarray:
    actor = next(
        (
            row.get("location")
            for row in freeze_frame
            if bool(row.get("actor")) and row.get("location")
        ),
        start_location,
    )
    actor_xy = (float(actor[0]), float(actor[1]))
    teammates = [
        (float(row["location"][0]), float(row["location"][1]))
        for row in freeze_frame
        if bool(row.get("teammate"))
        and not bool(row.get("actor"))
        and row.get("location")
    ]
    opponents = [
        (float(row["location"][0]), float(row["location"][1]))
        for row in freeze_frame
        if not bool(row.get("teammate")) and row.get("location")
    ]

    def fraction_ahead(points: list[tuple[float, float]]) -> float:
        return (
            sum(float(x > actor_xy[0]) for x, _y in points) / len(points)
            if points
            else 0.0
        )

    def mean_dx(points: list[tuple[float, float]]) -> float:
        return (
            sum(x - actor_xy[0] for x, _y in points) / len(points) / PITCH_LENGTH
            if points
            else 0.0
        )

    def mean_abs_dy(points: list[tuple[float, float]]) -> float:
        return (
            sum(abs(y - actor_xy[1]) for _x, y in points) / len(points) / PITCH_WIDTH
            if points
            else 0.0
        )

    teammate_distances = _distance(actor_xy, teammates)
    opponent_distances = _distance(actor_xy, opponents)
    teammate_box_count = sum(x >= 102.0 and 18.0 <= y <= 62.0 for x, y in teammates)
    return np.asarray(
        [
            len(freeze_frame) / 22.0,
            len(teammates) / 10.0,
            len(opponents) / 11.0,
            min(teammate_distances, default=PITCH_DIAGONAL) / PITCH_DIAGONAL,
            min(opponent_distances, default=PITCH_DIAGONAL) / PITCH_DIAGONAL,
            fraction_ahead(teammates),
            fraction_ahead(opponents),
            mean_dx(teammates),
            mean_dx(opponents),
            mean_abs_dy(teammates),
            mean_abs_dy(opponents),
            teammate_box_count / 10.0,
            fraction_ahead(opponents),
        ],
        dtype=np.float32,
    )


def current_features(
    event: dict[str, Any],
    freeze_frame: list[dict[str, Any]],
) -> np.ndarray:
    start = event["location"]
    return np.concatenate(
        [
            np.asarray(
                [
                    float(start[0]) / PITCH_LENGTH,
                    float(start[1]) / PITCH_WIDTH,
                    min(timestamp_seconds(event) / (120.0 * 60.0), 1.0),
                    float(bool(event.get("under_pressure"))),
                ],
                dtype=np.float32,
            ),
            freeze_frame_features(freeze_frame, start),
        ]
    )


def role_features(position_name: object) -> np.ndarray:
    role = broad_role(position_name)
    return np.asarray([float(role == name) for name in ROLE_NAMES], dtype=np.float32)


def _clock_seconds(value: object) -> float:
    parts = str(value or "0:00").split(":")
    if len(parts) != 2:
        return 0.0
    return float(parts[0]) * 60.0 + float(parts[1])


def active_lineup_player_ids(
    lineup_team: dict[str, Any],
    event: dict[str, Any],
) -> list[str]:
    event_period = int(event.get("period", 1))
    event_time = timestamp_seconds(event)
    active: list[str] = []
    for player in lineup_team.get("lineup") or []:
        for position in player.get("positions") or []:
            from_period = int(position.get("from_period") or 1)
            to_period_raw = position.get("to_period")
            to_period = int(to_period_raw) if to_period_raw is not None else 99
            if not from_period <= event_period <= to_period:
                continue
            if event_period == from_period and event_time < _clock_seconds(position.get("from")):
                continue
            if (
                to_period_raw is not None
                and event_period == to_period
                and position.get("to") is not None
                and event_time >= _clock_seconds(position.get("to"))
            ):
                continue
            active.append(str(player["player_id"]))
            break
    return sorted(set(active))


def _query_player_ids(
    data_dir: Path,
    records: list[MatchRecord],
    cohorts: list[Cohort],
) -> set[str]:
    player_ids: set[str] = set()
    for record in records:
        cohort = cohort_for_match(record, cohorts)
        if cohort is None or not record.has_360:
            continue
        for team in _read_json(data_dir / "lineups" / f"{record.match_id}.json"):
            for player in team.get("lineup") or []:
                player_ids.add(str(player["player_id"]))
    return player_ids


def _relevant_match_ids(
    data_dir: Path,
    records: list[MatchRecord],
    player_ids: set[str],
) -> set[str]:
    relevant: set[str] = set()
    for record in records:
        lineup_path = data_dir / "lineups" / f"{record.match_id}.json"
        if not lineup_path.is_file():
            continue
        found = False
        for team in _read_json(lineup_path):
            for player in team.get("lineup") or []:
                if str(player.get("player_id")) in player_ids:
                    relevant.add(record.match_id)
                    found = True
                    break
            if found:
                break
    return relevant


def _load_freeze_frames(path: Path) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    try:
        rows = _read_json(path)
    except json.JSONDecodeError:
        return {}, True
    return {
        str(row["event_uuid"]): list(row.get("freeze_frame") or [])
        for row in rows
    }, False


def _role_shuffle_map(
    players_by_role: dict[str, set[str]],
    *,
    seed: str,
) -> dict[str, str]:
    output: dict[str, str] = {}
    for role, player_ids in players_by_role.items():
        ordered = sorted(player_ids)
        if len(ordered) < 2:
            output.update({player_id: player_id for player_id in ordered})
            continue
        rng = random.Random(f"{seed}:{role}")
        shift = rng.randrange(1, len(ordered))
        for index, player_id in enumerate(ordered):
            output[player_id] = ordered[(index + shift) % len(ordered)]
    return output


def _lineup_profile_vector(
    histories: dict[str, list[tuple[str, str, PassSummary]]],
    player_ids: list[str],
    support_size: int,
) -> np.ndarray:
    profiles: list[np.ndarray] = []
    available = 0
    for player_id in player_ids:
        summary = aggregate_history(histories[player_id], support_size)
        _rolling, rich = profile_vectors(summary)
        profiles.append(rich)
        available += int(summary.pass_count > 0)
    if not profiles:
        _rolling, empty_rich = profile_vectors(PassSummary.empty())
        return np.concatenate([empty_rich, np.zeros(1, dtype=np.float32)])
    return np.concatenate(
        [
            np.stack(profiles).mean(axis=0),
            np.asarray([available / len(profiles)], dtype=np.float32),
        ]
    ).astype(np.float32)


def build_development_examples(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_root = config["data"]["statsbomb_root"]
    data_dir = resolve_statsbomb_data_dir(raw_root)
    records = load_match_records(data_dir)
    cohorts = load_cohorts(config)
    query_records = {
        row.match_id: cohort_for_match(row, cohorts)
        for row in records
        if cohort_for_match(row, cohorts) is not None
    }
    player_ids = _query_player_ids(data_dir, records, cohorts)
    relevant_ids = _relevant_match_ids(data_dir, records, player_ids)
    support_sizes = [int(value) for value in config["profiles"]["support_sizes"]]
    histories: dict[str, list[tuple[str, str, PassSummary]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    malformed_360: list[str] = []
    cohort_match_counts: dict[str, int] = defaultdict(int)

    by_date: dict[str, list[MatchRecord]] = defaultdict(list)
    for record in records:
        if record.match_id in relevant_ids or record.match_id in query_records:
            by_date[record.match_date].append(record)

    for match_date in sorted(by_date):
        date_records = by_date[match_date]
        for record in date_records:
            cohort = query_records.get(record.match_id)
            if cohort is None:
                continue
            assert cohort is not None
            events = _read_json(data_dir / "events" / f"{record.match_id}.json")
            lineups = _read_json(data_dir / "lineups" / f"{record.match_id}.json")
            lineup_by_team = {
                str(team.get("team_name") or ""): team
                for team in lineups
            }
            frames, malformed = _load_freeze_frames(
                data_dir / "three-sixty" / f"{record.match_id}.json"
            )
            if malformed:
                malformed_360.append(record.match_id)
                continue
            cohort_match_counts[cohort.name] += 1
            players_by_role: dict[str, set[str]] = defaultdict(set)
            candidates: list[tuple[int, dict[str, Any], str, str]] = []
            for index, event in enumerate(events):
                if (
                    (event.get("team") or {}).get("name") != cohort.focal_team_name
                    or not is_open_play_pass(event)
                    or event["id"] not in frames
                ):
                    continue
                player_id = str(event["player"]["id"])
                role = broad_role((event.get("position") or {}).get("name"))
                players_by_role[role].add(player_id)
                candidates.append((index, event, player_id, role))
            shuffle_map = _role_shuffle_map(
                players_by_role,
                seed=f"{config['evaluation']['shuffle_seed']}:{record.match_id}",
            )
            lineup_profile_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
            for index, event, player_id, role in candidates:
                labels = pass_labels(events, index)
                own_team = str((event.get("team") or {}).get("name") or "")
                opponent_team = next(
                    (
                        name
                        for name in lineup_by_team
                        if name and name != own_team
                    ),
                    "",
                )
                own_active = active_lineup_player_ids(
                    lineup_by_team.get(own_team, {}),
                    event,
                )
                opponent_active = active_lineup_player_ids(
                    lineup_by_team.get(opponent_team, {}),
                    event,
                )
                row: dict[str, Any] = {
                    "match_id": record.match_id,
                    "match_date": record.match_date,
                    "cohort": cohort.name,
                    "split": cohort.split,
                    "player_id": player_id,
                    "role": role,
                    "current": current_features(event, frames[event["id"]]),
                    "role_features": role_features((event.get("position") or {}).get("name")),
                    "labels": labels,
                }
                shuffled_player = shuffle_map.get(player_id, player_id)
                for support_size in support_sizes:
                    own_summary = aggregate_history(histories[player_id], support_size)
                    shuffled_summary = aggregate_history(
                        histories[shuffled_player],
                        support_size,
                    )
                    own_rolling, own_rich = query_conditioned_profile_vectors(
                        own_summary,
                        event["location"],
                        bool(event.get("under_pressure")),
                    )
                    _shuffled_rolling, shuffled_rich = query_conditioned_profile_vectors(
                        shuffled_summary,
                        event["location"],
                        bool(event.get("under_pressure")),
                    )
                    row[f"rolling_k{support_size}"] = own_rolling
                    row[f"rich_k{support_size}"] = own_rich
                    row[f"shuffled_rich_k{support_size}"] = shuffled_rich
                main_support = int(config["profiles"]["main_support_size"])
                teammate_ids = [
                    active_id
                    for active_id in own_active
                    if active_id != player_id
                ]
                teammate_key = ("teammate", tuple(teammate_ids))
                opponent_key = ("opponent", tuple(opponent_active))
                if teammate_key not in lineup_profile_cache:
                    lineup_profile_cache[teammate_key] = _lineup_profile_vector(
                        histories,
                        teammate_ids,
                        main_support,
                    )
                if opponent_key not in lineup_profile_cache:
                    lineup_profile_cache[opponent_key] = _lineup_profile_vector(
                        histories,
                        opponent_active,
                        main_support,
                    )
                row["teammate_profile"] = lineup_profile_cache[teammate_key]
                row["opponent_profile"] = lineup_profile_cache[opponent_key]
                rows.append(row)

        # Strict chronology: no match on the same date can enter another query's support.
        for record in date_records:
            if record.match_id not in relevant_ids:
                continue
            events = _read_json(data_dir / "events" / f"{record.match_id}.json")
            summaries = summarize_player_passes(
                events,
                allowed_player_ids=player_ids,
            )
            for player_id, summary in summaries.items():
                histories[player_id].append((record.match_date, record.match_id, summary))

    if not rows:
        raise ValueError("No StatsBomb player-history development examples were built.")
    audit = {
        "source_root": str(data_dir),
        "query_players": len(player_ids),
        "relevant_support_matches": len(relevant_ids),
        "development_examples": len(rows),
        "cohort_match_counts": dict(sorted(cohort_match_counts.items())),
        "split_example_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "development_test")
        },
        "target_positive_counts": {
            target: {
                split: sum(
                    row["split"] == split and bool(row["labels"][target])
                    for row in rows
                )
                for split in ("train", "validation", "development_test")
            }
            for target in config["tasks"]
        },
        "malformed_query_360_match_ids": sorted(malformed_360),
        "support_sizes": support_sizes,
        "chronology_rule": "support_match_date_strictly_before_query_match_date",
        "sealed_test_loaded": False,
    }
    cache = {
        "rows": rows,
        "current_feature_names": list(CURRENT_FEATURE_NAMES),
        "role_names": list(ROLE_NAMES),
        "rolling_profile_feature_names": list(ROLLING_PROFILE_FEATURE_NAMES),
        "audit": audit,
    }
    return cache, audit


def _standardize(
    features: torch.Tensor,
    train_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train = features[train_indices].float()
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std < 1e-5, torch.ones_like(std), std)
    return (features.float() - mean) / std, mean, std


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def fit_logistic_model(
    features: torch.Tensor,
    labels: torch.Tensor,
    split_indices: dict[str, list[int]],
    *,
    device: str,
    max_iterations: int,
    history_size: int,
    l2_weight: float,
) -> dict[str, Any]:
    standardized, mean, std = _standardize(features, split_indices["train"])
    compute_device = _device(device)
    x_train = standardized[split_indices["train"]].to(compute_device)
    y_train = labels[split_indices["train"]].float().to(compute_device)
    if int(y_train.sum()) in {0, len(y_train)}:
        raise ValueError("Logistic training requires both target classes.")
    model = torch.nn.Linear(features.shape[1], 1).to(compute_device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=max_iterations,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train).view(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        loss = loss + 0.5 * float(l2_weight) * model.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.inference_mode():
        probabilities = torch.sigmoid(
            model(standardized.to(compute_device)).view(-1)
        ).cpu()
    return {
        "probabilities": probabilities,
        "metrics": {
            split: _extended_binary_metrics(labels[indices], probabilities[indices])
            for split, indices in split_indices.items()
        },
        "standardization": {
            "mean": mean.squeeze(0),
            "std": std.squeeze(0),
        },
    }


def static_identity_features(
    player_ids: list[str],
    train_indices: list[int],
) -> tuple[torch.Tensor, dict[str, int]]:
    vocabulary = {
        player_id: index
        for index, player_id in enumerate(
            sorted({player_ids[index] for index in train_indices})
        )
    }
    features = torch.zeros(len(player_ids), len(vocabulary), dtype=torch.float32)
    for row_index, player_id in enumerate(player_ids):
        column = vocabulary.get(player_id)
        if column is not None:
            features[row_index, column] = 1.0
    return features, vocabulary


def _jsonable_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {"metrics": result["metrics"]}


def evaluate_development_cache(
    cache: dict[str, Any],
    config: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    rows = cache["rows"]
    split_indices = {
        split: [index for index, row in enumerate(rows) if row["split"] == split]
        for split in ("train", "validation", "development_test")
    }
    current = torch.from_numpy(np.stack([row["current"] for row in rows]))
    role = torch.from_numpy(np.stack([row["role_features"] for row in rows]))
    player_ids = [str(row["player_id"]) for row in rows]
    identity, identity_vocabulary = static_identity_features(
        player_ids,
        split_indices["train"],
    )
    probe = config["probe"]
    common = {
        "device": device,
        "max_iterations": int(probe["max_iterations"]),
        "history_size": int(probe["history_size"]),
        "l2_weight": float(probe["l2_weight"]),
    }
    output: dict[str, Any] = {
        "development_only": True,
        "sealed_test_loaded": False,
        "identity_vocabulary_size": len(identity_vocabulary),
        "tasks": {},
        "support_size_curves": {},
    }
    main_support = int(config["profiles"]["main_support_size"])
    for target in config["tasks"]:
        labels = torch.tensor(
            [bool(row["labels"][target]) for row in rows],
            dtype=torch.long,
        )
        rolling = torch.from_numpy(
            np.stack([row[f"rolling_k{main_support}"] for row in rows])
        )
        rich = torch.from_numpy(
            np.stack([row[f"rich_k{main_support}"] for row in rows])
        )
        teammate_profile = torch.from_numpy(
            np.stack([row["teammate_profile"] for row in rows])
        )
        opponent_profile = torch.from_numpy(
            np.stack([row["opponent_profile"] for row in rows])
        )
        shuffled_rich = torch.from_numpy(
            np.stack([row[f"shuffled_rich_k{main_support}"] for row in rows])
        )
        feature_views = {
            "A_current_geometry": current,
            "B_geometry_role": torch.cat([current, role], dim=1),
            "D_geometry_role_static_identity": torch.cat(
                [current, role, identity],
                dim=1,
            ),
            "E_geometry_role_rolling_stats": torch.cat(
                [current, role, rolling],
                dim=1,
            ),
            "F_geometry_role_history_profile": torch.cat(
                [current, role, rich],
                dim=1,
            ),
            "G_profile_teammate_lineup": torch.cat(
                [current, role, rich, teammate_profile],
                dim=1,
            ),
            "H_profile_lineups_opponent": torch.cat(
                [current, role, rich, teammate_profile, opponent_profile],
                dim=1,
            ),
            "same_role_shuffled_history": torch.cat(
                [current, role, shuffled_rich],
                dim=1,
            ),
        }
        fitted = {
            name: fit_logistic_model(features, labels, split_indices, **common)
            for name, features in feature_views.items()
        }
        output["tasks"][target] = {
            name: _jsonable_metrics(result) for name, result in fitted.items()
        }
        output["tasks"][target]["incremental_gains"] = {}
        for split in ("validation", "development_test"):
            rolling_metrics = fitted["E_geometry_role_rolling_stats"]["metrics"][split]
            profile_metrics = fitted["F_geometry_role_history_profile"]["metrics"][split]
            shuffle_metrics = fitted["same_role_shuffled_history"]["metrics"][split]
            output["tasks"][target]["incremental_gains"][split] = {
                "profile_minus_rolling_average_precision": float(
                    profile_metrics["average_precision"]
                    - rolling_metrics["average_precision"]
                ),
                "profile_minus_rolling_brier_improvement": float(
                    rolling_metrics["brier"] - profile_metrics["brier"]
                ),
                "profile_minus_rolling_log_loss_improvement": float(
                    rolling_metrics["log_loss"] - profile_metrics["log_loss"]
                ),
                "profile_minus_same_role_shuffle_average_precision": float(
                    profile_metrics["average_precision"]
                    - shuffle_metrics["average_precision"]
                ),
                "profile_minus_same_role_shuffle_brier_improvement": float(
                    shuffle_metrics["brier"] - profile_metrics["brier"]
                ),
            }

        support_curve: dict[str, Any] = {}
        for support_size in config["profiles"]["support_sizes"]:
            rich_k = torch.from_numpy(
                np.stack([row[f"rich_k{int(support_size)}"] for row in rows])
            )
            result = fit_logistic_model(
                torch.cat([current, role, rich_k], dim=1),
                labels,
                split_indices,
                **common,
            )
            support_curve[str(support_size)] = {
                split: result["metrics"][split]
                for split in ("validation", "development_test")
            }
        output["support_size_curves"][target] = support_curve
    output["result_payload_sha256"] = _stable_hash(output)
    return output
