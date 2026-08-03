"""Past-only telemetry windows and ten-touch outcome labels for RLCS V2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from footballq.data.rlcs_player_profiles import PROFILE_DIMENSION
from footballq.data.rlcs_replay import IdentityObservation, normalize_handle, repair_score_columns
from footballq.data.rlcs_touch_windows import (
    FIELD_SCALE,
    N_ENTITIES,
    N_FEATURES,
    N_PLAYERS,
    STATE_MASK_SIZE,
    STATE_SIZE,
    TIME_STEPS,
    Touch,
    TouchWindowError,
    build_state_tensor,
    extract_touches,
    relative_player_order,
    select_past_context,
)

NO_GOAL = 0
SCORE = 1
CONCEDE = 2
OUTCOME_NAMES = ("no_goal", "score", "concede")

PAIR_GEOMETRY_DIMENSION = 9
TEAM_FORM_DIMENSION = 6
SCALAR_CONTEXT_DIMENSION = 4

Termination = Literal["ten_touches", "goal", "kickoff", "censored"]


@dataclass(frozen=True)
class BoundaryEvent:
    frame_idx: int
    game_time_s: float
    kind: Literal["goal", "kickoff"]
    team: str | None


@dataclass(frozen=True)
class TenTouchOutcome:
    label: int | None
    touches_observed: int
    horizon_end_frame: int | None
    terminated_by: Termination

    @property
    def eligible(self) -> bool:
        return self.label is not None


def extract_boundary_events(events: pd.DataFrame) -> list[BoundaryEvent]:
    """Return de-duplicated chronological goal and kickoff boundaries."""

    ordered = events.sort_values(["observed_frame_number", "event_number"], kind="stable")
    records = ordered.to_dict(orient="records")
    goals: list[BoundaryEvent] = []
    last_goal_frame: int | None = None
    last_goal_team: str | None = None
    for row in records:
        kind = normalize_handle(row.get("event_type"))
        if kind != "goal":
            continue
        frame = int(row["observed_frame_number"])
        team = normalize_handle(row.get("event_team") or row.get("event_player_1_team"))
        if team not in {"blue", "orange"}:
            raise TouchWindowError("Goal boundary lacks a valid scoring team.")
        if (
            last_goal_frame is not None
            and last_goal_team == team
            and frame - last_goal_frame <= 45
        ):
            last_goal_frame = frame
            continue
        goals.append(
            BoundaryEvent(
                frame_idx=frame,
                game_time_s=float(row["game_time_s_precise"]),
                kind="goal",
                team=team,
            )
        )
        last_goal_frame = frame
        last_goal_team = team

    kickoffs = [
        BoundaryEvent(
            frame_idx=int(row["observed_frame_number"]),
            game_time_s=float(row["game_time_s_precise"]),
            kind="kickoff",
            team=None,
        )
        for row in records
        if normalize_handle(row.get("event_type")) == "kickoff"
    ]
    return sorted([*goals, *kickoffs], key=lambda item: (item.frame_idx, item.kind))


def label_ten_touch_outcome(
    touches: Sequence[Touch],
    current_index: int,
    boundaries: Sequence[BoundaryEvent],
    *,
    horizon_touches: int = 10,
) -> TenTouchOutcome:
    """Label the first goal before ten subsequent touches without crossing a reset."""

    if current_index < 0 or current_index >= len(touches):
        raise IndexError("Current touch index is outside the touch sequence.")
    if horizon_touches <= 0:
        raise ValueError("The touch horizon must be positive.")
    current = touches[current_index]
    future = list(touches[current_index + 1 : current_index + 1 + int(horizon_touches)])
    provisional_end = future[-1].frame_idx if future else current.frame_idx
    future_boundaries = [
        item
        for item in boundaries
        if item.frame_idx > current.frame_idx
        and (len(future) < horizon_touches or item.frame_idx <= provisional_end)
    ]
    boundary = (
        min(future_boundaries, key=lambda item: item.frame_idx)
        if future_boundaries
        else None
    )
    if boundary is not None:
        touches_before_boundary = sum(touch.frame_idx <= boundary.frame_idx for touch in future)
        if boundary.kind == "goal":
            label = SCORE if boundary.team == current.team else CONCEDE
            return TenTouchOutcome(
                label=label,
                touches_observed=touches_before_boundary,
                horizon_end_frame=boundary.frame_idx,
                terminated_by="goal",
            )
        return TenTouchOutcome(
            label=NO_GOAL,
            touches_observed=touches_before_boundary,
            horizon_end_frame=boundary.frame_idx,
            terminated_by="kickoff",
        )
    if len(future) == horizon_touches:
        return TenTouchOutcome(
            label=NO_GOAL,
            touches_observed=horizon_touches,
            horizon_end_frame=future[-1].frame_idx,
            terminated_by="ten_touches",
        )
    return TenTouchOutcome(
        label=None,
        touches_observed=len(future),
        horizon_end_frame=future[-1].frame_idx if future else None,
        terminated_by="censored",
    )


def near_reset_boundary(
    touch: Touch, boundaries: Sequence[BoundaryEvent], *, exclusion_seconds: float
) -> bool:
    return any(
        item.game_time_s <= touch.game_time_s
        and touch.game_time_s - item.game_time_s <= float(exclusion_seconds)
        for item in boundaries
    )


def _optional_score(value: Any) -> int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(numeric) else int(numeric)


def _finite_vector(row: Mapping[str, Any], prefix: str, stem: str) -> np.ndarray:
    values = np.asarray([row.get(f"{prefix}_{stem}_{axis}") for axis in "xyz"], dtype=float)
    if not np.isfinite(values).all():
        raise TouchWindowError(f"Missing {stem} vector for {prefix}.")
    return values


def pair_geometry(
    frame: Mapping[str, Any],
    *,
    actor_prefix: str,
    other_prefix: str,
    actor_team: str,
) -> np.ndarray:
    """Build the frozen relative-geometry/intercept vector for one player pair."""

    actor_position = _finite_vector(frame, actor_prefix, "pos")
    actor_velocity = _finite_vector(frame, actor_prefix, "vel")
    other_position = _finite_vector(frame, other_prefix, "pos")
    other_velocity = _finite_vector(frame, other_prefix, "vel")
    ball = np.asarray([frame.get(f"ball_pos_{axis}") for axis in "xyz"], dtype=float)
    if not np.isfinite(ball).all():
        raise TouchWindowError("Current ball position is missing.")
    side = np.ones(3, dtype=np.float64)
    if str(actor_team).casefold() == "orange":
        side[1] = -1.0
    relative_position = side * (other_position - actor_position) / FIELD_SCALE
    relative_velocity = side * (other_velocity - actor_velocity) / FIELD_SCALE
    other_ball_distance = float(np.linalg.norm(other_position - ball) / 12000.0)
    own_goal_y = -5120.0 if str(actor_team).casefold() == "blue" else 5120.0
    other_goal_distance = float(
        np.linalg.norm(other_position - np.asarray([0.0, own_goal_y, 0.0])) / 12000.0
    )

    def intercept(position: np.ndarray, velocity: np.ndarray) -> float:
        distance = float(np.linalg.norm(position - ball))
        speed = max(float(np.linalg.norm(velocity)), 500.0)
        return distance / speed

    intercept_difference = float(
        np.clip(
            intercept(actor_position, actor_velocity)
            - intercept(other_position, other_velocity),
            -5.0,
            5.0,
        )
        / 5.0
    )
    output = np.concatenate(
        [
            relative_position,
            relative_velocity,
            np.asarray(
                [other_ball_distance, other_goal_distance, intercept_difference],
                dtype=np.float64,
            ),
        ]
    ).astype(np.float32)
    if output.shape != (PAIR_GEOMETRY_DIMENSION,) or not np.isfinite(output).all():
        raise TouchWindowError("Pair geometry is not a finite nine-vector.")
    return output


def _profile_arrays(
    order: Sequence[str],
    roster_ids: Mapping[str, str],
    snapshots: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    profiles = []
    uncertainty = []
    effective = []
    form = []
    for prefix in order:
        player_id = str(roster_ids[prefix])
        if player_id not in snapshots:
            raise TouchWindowError(f"Missing chronology-safe profile for {player_id}.")
        snapshot = snapshots[player_id]
        profile = np.asarray(snapshot["profile"], dtype=np.float32)
        error = np.asarray(snapshot["uncertainty"], dtype=np.float32)
        if profile.shape != (PROFILE_DIMENSION,) or error.shape != (PROFILE_DIMENSION,):
            raise TouchWindowError("Profile snapshot has the wrong frozen width.")
        profiles.append(profile)
        uncertainty.append(error)
        effective.append(float(snapshot["effective_sample_size"]))
        form.append(
            [
                float(snapshot.get("prior_win_rate", 0.5)),
                float(snapshot.get("prior_goal_diff", 0.0)),
                math.log1p(float(snapshot.get("n_prior_games", 0))),
            ]
        )
    form_array = np.asarray(form, dtype=np.float32)
    team_form = np.asarray(
        [
            form_array[:3, 0].mean(),
            form_array[3:, 0].mean(),
            form_array[:3, 1].mean(),
            form_array[3:, 1].mean(),
            form_array[:3, 2].mean(),
            form_array[3:, 2].mean(),
        ],
        dtype=np.float32,
    )
    return (
        np.stack(profiles),
        np.stack(uncertainty),
        np.asarray(effective, dtype=np.float32),
        team_form,
    )


def build_replay_value_rows(
    frames: pd.DataFrame,
    events: pd.DataFrame,
    *,
    replay_id: str,
    inventory: Mapping[str, Any],
    stage: str,
    observations: Sequence[IdentityObservation],
    roster_ids: Mapping[str, str],
    snapshots: Mapping[str, Mapping[str, Any]],
    fps: float = 10.0,
    context_seconds: float = 2.0,
    horizon_touches: int = 10,
    exclude_goal_reset_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    """Construct V2 state/profile rows for one accepted replay."""

    if stage == "test":
        raise PermissionError("Ordinary V2 dataset construction may not open sealed test rows.")
    if len(roster_ids) != N_PLAYERS:
        raise TouchWindowError("V2 value construction requires six resolved identities.")
    repaired = repair_score_columns(
        events,
        expected_blue_score=_optional_score(inventory.get("blue_score")),
        expected_orange_score=_optional_score(inventory.get("orange_score")),
    )
    touches = extract_touches(repaired, observations, roster_ids, scores_repaired=True)
    boundaries = extract_boundary_events(repaired)
    frame_ids = frames["observed_frame_number"].to_numpy(dtype=np.int64)
    if np.any(np.diff(frame_ids) <= 0):
        raise TouchWindowError("Value frames must be strictly ordered.")
    rows: list[dict[str, Any]] = []
    for current_index, current in enumerate(touches):
        outcome = label_ten_touch_outcome(
            touches, current_index, boundaries, horizon_touches=horizon_touches
        )
        if not outcome.eligible or near_reset_boundary(
            current, boundaries, exclusion_seconds=exclude_goal_reset_seconds
        ):
            continue
        current_row_index = int(np.searchsorted(frame_ids, current.frame_idx, side="right") - 1)
        if current_row_index < 0:
            continue
        current_row = frames.iloc[current_row_index]
        try:
            selection = select_past_context(
                frames,
                touch_frame_idx=current.frame_idx,
                touch_time_s=current.game_time_s,
                fps=fps,
                context_seconds=context_seconds,
                max_frame_lag_seconds=0.15,
            )
            order = relative_player_order(
                current_row.to_dict(), actor_prefix=current.player_prefix, observations=observations
            )
            if "stint_number" in frames:
                stints = pd.to_numeric(
                    frames.iloc[list(selection.row_indices)]["stint_number"], errors="coerce"
                ).dropna()
                if stints.nunique() > 1:
                    continue
            state, state_mask = build_state_tensor(
                frames, selection, car_order=order, actor_team=current.team
            )
            profiles, uncertainty, effective, team_form = _profile_arrays(
                order, roster_ids, snapshots
            )
            pair = np.stack(
                [
                    pair_geometry(
                        current_row.to_dict(),
                        actor_prefix=order[0],
                        other_prefix=prefix,
                        actor_team=current.team,
                    )
                    for prefix in order[3:6]
                ]
            )
            teammate = np.stack(
                [
                    pair_geometry(
                        current_row.to_dict(),
                        actor_prefix=order[0],
                        other_prefix=prefix,
                        actor_team=current.team,
                    )
                    for prefix in order[1:3]
                ]
            )
        except TouchWindowError:
            continue
        score_diff = (
            current.blue_score - current.orange_score
            if current.team == "blue"
            else current.orange_score - current.blue_score
        )
        raw_stint = current_row.get("stint_number", 0)
        stint = int(raw_stint) if pd.notna(raw_stint) else 0
        player_ids = [str(roster_ids[prefix]) for prefix in order]
        rows.append(
            {
                "sample_id": f"{replay_id}:stint_{stint}:touch_{current.frame_idx}:v2",
                "replay_id": str(replay_id),
                "series_id": str(inventory.get("series_id") or ""),
                "group_path": str(inventory.get("group_path") or ""),
                "region": str(inventory.get("region") or "").upper(),
                "event_time_utc": inventory.get("event_time_utc"),
                "v2_stage": str(stage),
                "frame_idx": int(current.frame_idx),
                "game_time_s": np.float32(current.game_time_s),
                "seconds_remaining": np.float32(max(300.0 - current.game_time_s, 0.0)),
                "score_diff_actor": int(np.clip(score_diff, -127, 127)),
                "overtime": bool(current.game_time_s > 300.0),
                "actor_player_id": current.player_id,
                "actor_team": current.team,
                "player_ids": player_ids,
                "state_flat": state.reshape(-1).tolist(),
                "state_mask": state_mask.reshape(-1).tolist(),
                "profile_flat": profiles.reshape(-1).tolist(),
                "profile_uncertainty_flat": uncertainty.reshape(-1).tolist(),
                "profile_effective_sample_size": effective.tolist(),
                "team_form": team_form.tolist(),
                "pair_geometry_flat": pair.reshape(-1).tolist(),
                "teammate_geometry_flat": teammate.reshape(-1).tolist(),
                "outcome_label": int(outcome.label),
                "outcome_name": OUTCOME_NAMES[int(outcome.label)],
                "horizon_touch_count": int(outcome.touches_observed),
                "horizon_end_frame": int(outcome.horizon_end_frame or current.frame_idx),
                "terminated_by": outcome.terminated_by,
            }
        )
    return rows


def value_arrow_schema() -> Any:
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RLCS value dataset writing requires pyarrow.") from exc
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("replay_id", pa.string(), nullable=False),
            pa.field("series_id", pa.string(), nullable=False),
            pa.field("group_path", pa.string(), nullable=False),
            pa.field("region", pa.string(), nullable=False),
            pa.field("event_time_utc", pa.timestamp("ms", tz="UTC"), nullable=True),
            pa.field("v2_stage", pa.string(), nullable=False),
            pa.field("frame_idx", pa.int32(), nullable=False),
            pa.field("game_time_s", pa.float32(), nullable=False),
            pa.field("seconds_remaining", pa.float32(), nullable=False),
            pa.field("score_diff_actor", pa.int8(), nullable=False),
            pa.field("overtime", pa.bool_(), nullable=False),
            pa.field("actor_player_id", pa.string(), nullable=False),
            pa.field("actor_team", pa.string(), nullable=False),
            pa.field("player_ids", pa.list_(pa.string(), N_PLAYERS), nullable=False),
            pa.field("state_flat", pa.list_(pa.float32(), STATE_SIZE), nullable=False),
            pa.field("state_mask", pa.list_(pa.bool_(), STATE_MASK_SIZE), nullable=False),
            pa.field(
                "profile_flat",
                pa.list_(pa.float32(), N_PLAYERS * PROFILE_DIMENSION),
                nullable=False,
            ),
            pa.field(
                "profile_uncertainty_flat",
                pa.list_(pa.float32(), N_PLAYERS * PROFILE_DIMENSION),
                nullable=False,
            ),
            pa.field(
                "profile_effective_sample_size",
                pa.list_(pa.float32(), N_PLAYERS),
                nullable=False,
            ),
            pa.field("team_form", pa.list_(pa.float32(), TEAM_FORM_DIMENSION), nullable=False),
            pa.field(
                "pair_geometry_flat",
                pa.list_(pa.float32(), 3 * PAIR_GEOMETRY_DIMENSION),
                nullable=False,
            ),
            pa.field(
                "teammate_geometry_flat",
                pa.list_(pa.float32(), 2 * PAIR_GEOMETRY_DIMENSION),
                nullable=False,
            ),
            pa.field("outcome_label", pa.int8(), nullable=False),
            pa.field("outcome_name", pa.string(), nullable=False),
            pa.field("horizon_touch_count", pa.int8(), nullable=False),
            pa.field("horizon_end_frame", pa.int32(), nullable=False),
            pa.field("terminated_by", pa.string(), nullable=False),
        ]
    )


def write_value_parquet(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RLCS value dataset writing requires pyarrow.") from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    for source in rows:
        row = dict(source)
        timestamp = row.get("event_time_utc")
        missing_timestamp = timestamp is None or pd.isna(timestamp)
        row["event_time_utc"] = (
            None if missing_timestamp else pd.Timestamp(timestamp).to_pydatetime()
        )
        normalized.append(row)
    table = pa.Table.from_pylist(normalized, schema=value_arrow_schema())
    pq.write_table(table, destination, compression="zstd")
    return destination


def validate_value_shapes() -> dict[str, int]:
    """Expose frozen tensor widths for manifests and tests."""

    return {
        "time_steps": TIME_STEPS,
        "entities": N_ENTITIES,
        "features": N_FEATURES,
        "state_size": STATE_SIZE,
        "profile_dimension": PROFILE_DIMENSION,
        "players": N_PLAYERS,
        "pair_geometry_dimension": PAIR_GEOMETRY_DIMENSION,
        "team_form_dimension": TEAM_FORM_DIMENSION,
        "scalar_context_dimension": SCALAR_CONTEXT_DIMENSION,
    }
