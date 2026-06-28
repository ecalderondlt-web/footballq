"""Build fixed 23-entity Torch windows for Phase 1 baselines."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from footballq.constants import AGENT_BALL, AGENT_PLAYER
from footballq.data.normalize import normalize_velocity_from_mps, normalize_xy_from_meters
from footballq.repro.identity import make_sample_id, sample_ids_from_components

N_ENTITIES = 23
BALL_INDEX = 0
HOME_START = 1
AWAY_START = 12
N_PLAYERS_PER_TEAM = 11

ENTITY_BALL = 0
ENTITY_PLAYER = 1
TEAM_NEUTRAL = 0
TEAM_HOME = 1
TEAM_AWAY = 2
TEAM_UNKNOWN = 3

FEATURE_NAMES = [
    "x_norm",
    "y_norm",
    "vx_norm",
    "vy_norm",
    "is_ball",
    "is_home",
    "is_away",
    "is_possession_team",
    "has_possession",
    "visible_mask",
]


@dataclass
class TrackingWindowTensorData:
    """Tensor-backed window dataset payload saved by ``prepare_tracking_data``."""

    past: torch.Tensor
    future_xy: torch.Tensor
    past_mask: torch.Tensor
    future_mask: torch.Tensor
    entity_type: torch.Tensor
    team_id: torch.Tensor
    match_id: list[str]
    period: list[int]
    start_frame: list[int]
    feature_names: list[str]
    fps: float
    context_seconds: float
    horizon_seconds: float
    stride_seconds: float
    coordinate_mode: str = "centered_normalized"
    phase: list[str] | None = None
    event_type: list[str] | None = None
    possession_team_id: list[str] | None = None
    possession_available: list[bool] | None = None
    label_frame: list[int] | None = None
    sample_id: list[str] | None = None

    def __post_init__(self) -> None:
        n = len(self.match_id)
        if len(self.period) != n:
            raise ValueError("period must have the same length as match_id.")
        if self.phase is None:
            self.phase = ["unknown"] * n
        if self.event_type is None:
            self.event_type = ["unknown"] * n
        if self.possession_team_id is None:
            self.possession_team_id = ["unknown"] * n
        if self.possession_available is None:
            self.possession_available = [False] * n
        if self.label_frame is None:
            self.label_frame = list(self.start_frame)
        if self.sample_id is None:
            self.sample_id = sample_ids_from_components(
                self.match_id, self.period, self.start_frame
            )

    @property
    def history_steps(self) -> int:
        return int(self.past.shape[1])

    @property
    def horizon_steps(self) -> int:
        return int(self.future_xy.shape[1])

    @property
    def n_entities(self) -> int:
        return int(self.past.shape[2])

    @property
    def n_features(self) -> int:
        return int(self.past.shape[3])

    def window_records(self) -> list[dict[str, Any]]:
        """Return per-window dicts for lightweight inspection scripts."""

        return [
            {
                "past": self.past[idx],
                "future_xy": self.future_xy[idx],
                "past_mask": self.past_mask[idx],
                "future_mask": self.future_mask[idx],
                "entity_type": self.entity_type[idx],
                "team_id": self.team_id[idx],
                "match_id": self.match_id[idx],
                "period": self.period[idx],
                "start_frame": self.start_frame[idx],
                "sample_id": self.sample_id[idx],
                "label_frame": self.label_frame[idx],
                "phase": self.phase[idx],
                "event_type": self.event_type[idx],
                "possession_team_id": self.possession_team_id[idx],
                "possession_available": self.possession_available[idx],
            }
            for idx in range(len(self.match_id))
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "windows": self.window_records(),
            "past": self.past,
            "future_xy": self.future_xy,
            "past_mask": self.past_mask,
            "future_mask": self.future_mask,
            "entity_type": self.entity_type,
            "team_id": self.team_id,
            "match_id": self.match_id,
            "period": self.period,
            "start_frame": self.start_frame,
            "sample_id": self.sample_id,
            "label_frame": self.label_frame,
            "phase": self.phase,
            "event_type": self.event_type,
            "possession_team_id": self.possession_team_id,
            "possession_available": self.possession_available,
            "feature_names": self.feature_names,
            "fps": self.fps,
            "context_seconds": self.context_seconds,
            "horizon_seconds": self.horizon_seconds,
            "stride_seconds": self.stride_seconds,
            "coordinate_mode": self.coordinate_mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrackingWindowTensorData:
        return cls(
            past=payload["past"].float(),
            future_xy=payload["future_xy"].float(),
            past_mask=payload["past_mask"].bool(),
            future_mask=payload["future_mask"].bool(),
            entity_type=payload["entity_type"].long(),
            team_id=payload["team_id"].long(),
            match_id=[str(value) for value in payload["match_id"]],
            period=[
                int(value)
                for value in payload.get(
                    "period",
                    payload.get("periods", [1 for _ in payload["match_id"]]),
                )
            ],
            start_frame=[int(value) for value in payload["start_frame"]],
            sample_id=(
                [str(value) for value in payload["sample_id"]] if "sample_id" in payload else None
            ),
            label_frame=[
                int(value) for value in payload.get("label_frame", payload["start_frame"])
            ],
            phase=(
                [_clean_metadata_value(value) for value in payload["phase"]]
                if "phase" in payload
                else None
            ),
            event_type=(
                [_clean_metadata_value(value) for value in payload["event_type"]]
                if "event_type" in payload
                else None
            ),
            possession_team_id=(
                [_normalize_team_id(value) for value in payload["possession_team_id"]]
                if "possession_team_id" in payload
                else None
            ),
            possession_available=(
                [bool(value) for value in payload["possession_available"]]
                if "possession_available" in payload
                else None
            ),
            feature_names=[str(value) for value in payload["feature_names"]],
            fps=float(payload["fps"]),
            context_seconds=float(payload["context_seconds"]),
            horizon_seconds=float(payload["horizon_seconds"]),
            stride_seconds=float(payload["stride_seconds"]),
            coordinate_mode=str(payload.get("coordinate_mode", "centered_normalized")),
        )


class TrackingWindowDataset(Dataset):
    """PyTorch dataset for fixed soccer tracking windows."""

    def __init__(self, data: TrackingWindowTensorData, indices: list[int] | None = None) -> None:
        self.data = data
        self.indices = list(range(len(data.match_id))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data_index = self.indices[index]
        return {
            "past": self.data.past[data_index],
            "future_xy": self.data.future_xy[data_index],
            "past_mask": self.data.past_mask[data_index],
            "future_mask": self.data.future_mask[data_index],
            "entity_type": self.data.entity_type[data_index],
            "team_id": self.data.team_id[data_index],
            "match_id": self.data.match_id[data_index],
            "period": self.data.period[data_index],
            "sample_id": self.data.sample_id[data_index],
            "start_frame": self.data.start_frame[data_index],
        }

    def subset(self, indices: list[int]) -> TrackingWindowDataset:
        return TrackingWindowDataset(self.data, indices=indices)


def save_windows_pt(data: TrackingWindowTensorData, out: str | Path) -> Path:
    """Save window tensors and metadata to a portable ``.pt`` payload."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data.to_dict(), out_path)
    return out_path


def load_windows_pt(path: str | Path) -> TrackingWindowTensorData:
    """Load windows saved by :func:`save_windows_pt`."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, list):
        return _from_window_records(payload)
    if isinstance(payload, dict) and "train" in payload and "past" not in payload:
        return _from_window_records(payload["train"])
    return TrackingWindowTensorData.from_dict(payload)


def _from_window_records(records: list[dict[str, Any]]) -> TrackingWindowTensorData:
    if not records:
        raise ValueError("Window record list is empty.")
    return TrackingWindowTensorData(
        past=torch.stack([record["past"].float() for record in records]),
        future_xy=torch.stack([record["future_xy"].float() for record in records]),
        past_mask=torch.stack([record["past_mask"].bool() for record in records]),
        future_mask=torch.stack([record["future_mask"].bool() for record in records]),
        entity_type=torch.stack([record["entity_type"].long() for record in records]),
        team_id=torch.stack([record["team_id"].long() for record in records]),
        match_id=[str(record.get("match_id", "")) for record in records],
        period=[int(record.get("period", 1)) for record in records],
        start_frame=[int(record.get("start_frame", 0)) for record in records],
        sample_id=[
            str(record["sample_id"])
            if "sample_id" in record
            else make_sample_id(
                record.get("match_id", ""), record.get("period", 1), record.get("start_frame", 0)
            )
            for record in records
        ],
        label_frame=[
            int(record.get("label_frame", record.get("start_frame", 0))) for record in records
        ],
        phase=[_clean_metadata_value(record.get("phase")) for record in records],
        event_type=[_clean_metadata_value(record.get("event_type")) for record in records],
        possession_team_id=[
            _normalize_team_id(record.get("possession_team_id")) for record in records
        ],
        possession_available=[
            bool(record.get("possession_available", False)) for record in records
        ],
        feature_names=list(FEATURE_NAMES),
        fps=10.0,
        context_seconds=float(records[0]["past"].shape[0]) / 10.0,
        horizon_seconds=float(records[0]["future_xy"].shape[0]) / 10.0,
        stride_seconds=0.2,
    )


def _natural_key(value: object) -> tuple[str, int]:
    text = str(value)
    match = re.search(r"(\d+)$", text)
    return re.sub(r"\d+$", "", text), int(match.group(1)) if match else -1


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "entity_id" in out.columns and "agent_id" not in out.columns:
        out["agent_id"] = out["entity_id"]
    if "entity_type" in out.columns and "agent_type" not in out.columns:
        out["agent_type"] = out["entity_type"]
    if "visible" in out.columns and "is_visible" not in out.columns:
        out["is_visible"] = out["visible"]
    if "agent_id" in out.columns and "entity_id" not in out.columns:
        out["entity_id"] = out["agent_id"]
    if "agent_type" in out.columns and "entity_type" not in out.columns:
        out["entity_type"] = out["agent_type"]
    if "is_visible" in out.columns and "visible" not in out.columns:
        out["visible"] = out["is_visible"]
    if "period" not in out.columns:
        out["period"] = 1
    if "frame_id" not in out.columns:
        out["frame_id"] = out.groupby(["match_id", "period"], dropna=False).cumcount()
    if "fps" not in out.columns:
        out["fps"] = pd.NA
    if "has_possession" not in out.columns:
        out["has_possession"] = False
    if "possession_team_id" not in out.columns:
        out["possession_team_id"] = pd.NA
    out["x_m"] = pd.to_numeric(out["x_m"], errors="coerce")
    out["y_m"] = pd.to_numeric(out["y_m"], errors="coerce")
    out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce")
    out["frame_id"] = pd.to_numeric(out["frame_id"], errors="coerce")
    out["visible"] = (
        out["visible"].fillna(True).astype(bool) & out["x_m"].notna() & out["y_m"].notna()
    )
    out["is_visible"] = out["visible"]
    return out.sort_values(
        ["match_id", "period", "time_s", "frame_id", "agent_id"],
        kind="mergesort",
    )


def _normalize_team_id(value: object) -> str:
    if isinstance(value, dict):
        value = (
            value.get("team")
            or value.get("team_id")
            or value.get("group")
            or value.get("side")
            or value.get("name")
        )
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"home", "h", "home_team", "team_home", "1", "team_1"}:
        return "home"
    if text in {"away", "a", "away_team", "team_away", "2", "team_2"}:
        return "away"
    if "home_team" in text or text.endswith("_home") or text.startswith("home_"):
        return "home"
    if "away_team" in text or text.endswith("_away") or text.startswith("away_"):
        return "away"
    if text in {"ball", "neutral", "none", "nan", "<na>", ""}:
        return "neutral"
    return text


def _clean_metadata_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>", "none", "null"}:
        return "unknown"
    return text


def _first_non_missing(values: pd.Series | None) -> object:
    if values is None:
        return pd.NA
    for value in values:
        if pd.notna(value):
            return value
    return pd.NA


def _metadata_for_time(period_df: pd.DataFrame, time_s: float) -> dict[str, object]:
    rows = period_df[period_df["time_s"].astype(float) == float(time_s)]
    possession = _normalize_team_id(_first_non_missing(rows.get("possession_team_id")))
    return {
        "phase": _clean_metadata_value(_first_non_missing(rows.get("phase"))),
        "event_type": _clean_metadata_value(_first_non_missing(rows.get("event_type"))),
        "possession_team_id": possession,
        "possession_available": possession in {"home", "away"},
    }


def _team_code(value: object) -> int:
    team = _normalize_team_id(value)
    if team == "home":
        return TEAM_HOME
    if team == "away":
        return TEAM_AWAY
    if team == "neutral":
        return TEAM_NEUTRAL
    return TEAM_UNKNOWN


def _with_causal_velocity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    needs_vx = "vx_mps" not in out.columns or out["vx_mps"].isna().all()
    needs_vy = "vy_mps" not in out.columns or out["vy_mps"].isna().all()
    if not needs_vx and not needs_vy:
        out["vx_mps"] = pd.to_numeric(out["vx_mps"], errors="coerce").fillna(0.0)
        out["vy_mps"] = pd.to_numeric(out["vy_mps"], errors="coerce").fillna(0.0)
        return out

    out["vx_mps"] = pd.to_numeric(out.get("vx_mps", np.nan), errors="coerce")
    out["vy_mps"] = pd.to_numeric(out.get("vy_mps", np.nan), errors="coerce")
    group_cols = ["match_id", "period", "agent_id"]
    for _, group in out.groupby(group_cols, dropna=False, sort=False):
        idx = group.index
        times = group["time_s"].to_numpy(dtype=float)
        x = group["x_m"].to_numpy(dtype=float)
        y = group["y_m"].to_numpy(dtype=float)
        dt = np.diff(times)
        vx = np.zeros(len(group), dtype=np.float32)
        vy = np.zeros(len(group), dtype=np.float32)
        finite = np.isfinite(dt) & (dt > 0)
        if finite.any():
            dx = np.diff(x)
            dy = np.diff(y)
            vx[1:] = np.divide(dx, dt, out=np.zeros_like(dx, dtype=np.float64), where=finite)
            vy[1:] = np.divide(dy, dt, out=np.zeros_like(dy, dtype=np.float64), where=finite)
            vx[0] = vx[1] if len(vx) > 1 else 0.0
            vy[0] = vy[1] if len(vy) > 1 else 0.0
        out.loc[idx, "vx_mps"] = out.loc[idx, "vx_mps"].fillna(pd.Series(vx, index=idx))
        out.loc[idx, "vy_mps"] = out.loc[idx, "vy_mps"].fillna(pd.Series(vy, index=idx))
    out["vx_mps"] = out["vx_mps"].fillna(0.0)
    out["vy_mps"] = out["vy_mps"].fillna(0.0)
    return out


def _selected_times(times: np.ndarray, fps_out: float) -> np.ndarray:
    times = np.asarray(sorted(np.unique(times[np.isfinite(times)])), dtype=float)
    if len(times) < 2:
        return times
    target_dt = 1.0 / fps_out
    source_dt = np.median(np.diff(times))
    if math.isclose(source_dt, target_dt, rel_tol=0.05, abs_tol=1e-5):
        return times
    target = np.arange(times[0], times[-1] + target_dt / 2.0, target_dt)
    selected: list[float] = []
    for value in target:
        idx = int(np.argmin(np.abs(times - value)))
        selected.append(float(times[idx]))
    return np.asarray(sorted(set(selected)), dtype=float)


def _agents_for_period(period_df: pd.DataFrame) -> list[str]:
    ball_ids = sorted(
        period_df[period_df["agent_type"] == AGENT_BALL]["agent_id"].dropna().astype(str).unique(),
        key=_natural_key,
    )
    players = period_df[period_df["agent_type"] == AGENT_PLAYER].copy()
    players["team_norm"] = players["team_id"].map(_normalize_team_id)

    home = sorted(
        players[players["team_norm"] == "home"]["agent_id"].dropna().astype(str).unique(),
        key=_natural_key,
    )
    away = sorted(
        players[players["team_norm"] == "away"]["agent_id"].dropna().astype(str).unique(),
        key=_natural_key,
    )
    unknown = sorted(
        players[~players["team_norm"].isin(["home", "away"])]["agent_id"]
        .dropna()
        .astype(str)
        .unique(),
        key=_natural_key,
    )
    if len(home) < N_PLAYERS_PER_TEAM and unknown:
        take = unknown[: N_PLAYERS_PER_TEAM - len(home)]
        home.extend(take)
        unknown = unknown[len(take) :]
    if len(away) < N_PLAYERS_PER_TEAM and unknown:
        take = unknown[: N_PLAYERS_PER_TEAM - len(away)]
        away.extend(take)

    return (
        [(ball_ids[0] if ball_ids else "ball")]
        + home[:N_PLAYERS_PER_TEAM]
        + [""] * max(0, N_PLAYERS_PER_TEAM - len(home))
        + away[:N_PLAYERS_PER_TEAM]
        + [""] * max(0, N_PLAYERS_PER_TEAM - len(away))
    )


def _static_entity_arrays(
    agent_ids: list[str],
    period_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    entity_type = np.zeros(N_ENTITIES, dtype=np.int64)
    team_id = np.zeros(N_ENTITIES, dtype=np.int64)
    lookup = (
        period_df.drop_duplicates("agent_id")
        .set_index("agent_id")[["agent_type", "team_id"]]
        .to_dict(orient="index")
    )
    for idx, agent_id in enumerate(agent_ids):
        if idx == BALL_INDEX:
            entity_type[idx] = ENTITY_BALL
            team_id[idx] = TEAM_NEUTRAL
            continue
        entity_type[idx] = ENTITY_PLAYER
        if HOME_START <= idx < AWAY_START:
            team_id[idx] = TEAM_HOME
        elif idx >= AWAY_START:
            team_id[idx] = TEAM_AWAY
        elif agent_id in lookup:
            team_id[idx] = _team_code(lookup[agent_id]["team_id"])
    return entity_type, team_id


def _row_features(row: Any, entity_type: int, team_id: int) -> np.ndarray:
    xy_norm = normalize_xy_from_meters(np.array([float(row.x_m), float(row.y_m)], dtype=np.float32))
    v_norm = normalize_velocity_from_mps(
        np.array([float(row.vx_mps), float(row.vy_mps)], dtype=np.float32)
    )
    possession_team = _team_code(getattr(row, "possession_team_id", pd.NA))
    has_possession = bool(getattr(row, "has_possession", False))
    is_possession_team = float(possession_team == team_id and team_id in {TEAM_HOME, TEAM_AWAY})
    return np.array(
        [
            xy_norm[0],
            xy_norm[1],
            v_norm[0],
            v_norm[1],
            float(entity_type == ENTITY_BALL),
            float(team_id == TEAM_HOME),
            float(team_id == TEAM_AWAY),
            is_possession_team,
            float(has_possession),
            float(bool(getattr(row, "visible", True))),
        ],
        dtype=np.float32,
    )


def build_tracking_windows(
    tracking_df: pd.DataFrame,
    fps_out: float = 10.0,
    context_seconds: float = 2.0,
    horizon_seconds: float = 2.0,
    stride_seconds: float = 0.2,
) -> TrackingWindowTensorData:
    """Convert canonical tracking rows to fixed-shape Torch tensors.

    Output order is fixed per window: ball at index 0, home players at 1-11,
    and away players at 12-22. Missing or invisible entities keep tensor shape
    and are represented through masks.
    """

    source = _with_causal_velocity(_standardize_columns(tracking_df))
    history_steps = int(round(context_seconds * fps_out))
    horizon_steps = int(round(horizon_seconds * fps_out))
    stride_steps = max(1, int(round(stride_seconds * fps_out)))
    total_steps = history_steps + horizon_steps

    past_rows: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    past_masks: list[np.ndarray] = []
    future_masks: list[np.ndarray] = []
    entity_types: list[np.ndarray] = []
    team_ids: list[np.ndarray] = []
    match_ids: list[str] = []
    periods: list[int] = []
    start_frames: list[int] = []
    label_frames: list[int] = []
    phases: list[str] = []
    event_types: list[str] = []
    possession_team_ids: list[str] = []
    possession_available_values: list[bool] = []

    for (match_id, period), period_df in source.groupby(
        ["match_id", "period"],
        dropna=False,
        sort=False,
    ):
        period_df = period_df.copy()
        times = _selected_times(period_df["time_s"].to_numpy(dtype=float), fps_out=fps_out)
        if len(times) < total_steps:
            continue
        allowed_times = set(float(value) for value in times)
        period_df = period_df[period_df["time_s"].astype(float).isin(allowed_times)]
        agent_ids = _agents_for_period(period_df)
        entity_type_arr, team_id_arr = _static_entity_arrays(agent_ids, period_df)
        indexed = {
            (float(row.time_s), str(row.agent_id)): row for row in period_df.itertuples(index=False)
        }
        frame_by_time = (
            period_df.drop_duplicates("time_s").set_index("time_s")["frame_id"].to_dict()
        )

        for start in range(0, len(times) - total_steps + 1, stride_steps):
            history_times = times[start : start + history_steps]
            future_times = times[start + history_steps : start + total_steps]
            past = np.zeros((history_steps, N_ENTITIES, len(FEATURE_NAMES)), dtype=np.float32)
            future = np.zeros((horizon_steps, N_ENTITIES, 2), dtype=np.float32)
            past_mask = np.zeros((history_steps, N_ENTITIES), dtype=bool)
            future_mask = np.zeros((horizon_steps, N_ENTITIES), dtype=bool)

            for t_idx, time_s in enumerate(history_times):
                for entity_idx, agent_id in enumerate(agent_ids):
                    if not agent_id:
                        continue
                    row = indexed.get((float(time_s), str(agent_id)))
                    if row is None or not bool(getattr(row, "visible", True)):
                        continue
                    past[t_idx, entity_idx, :] = _row_features(
                        row,
                        int(entity_type_arr[entity_idx]),
                        int(team_id_arr[entity_idx]),
                    )
                    past_mask[t_idx, entity_idx] = True

            for t_idx, time_s in enumerate(future_times):
                for entity_idx, agent_id in enumerate(agent_ids):
                    if not agent_id:
                        continue
                    row = indexed.get((float(time_s), str(agent_id)))
                    if row is None or not bool(getattr(row, "visible", True)):
                        continue
                    future[t_idx, entity_idx, :] = normalize_xy_from_meters(
                        np.array([float(row.x_m), float(row.y_m)], dtype=np.float32)
                    )
                    future_mask[t_idx, entity_idx] = True

            past_rows.append(past)
            future_rows.append(future)
            past_masks.append(past_mask)
            future_masks.append(future_mask)
            entity_types.append(entity_type_arr)
            team_ids.append(team_id_arr)
            match_ids.append(str(match_id))
            periods.append(int(period))
            start_frames.append(int(frame_by_time.get(float(history_times[0]), start)))
            label_time = float(history_times[-1])
            label_meta = _metadata_for_time(period_df, label_time)
            label_frames.append(int(frame_by_time.get(label_time, start + history_steps - 1)))
            phases.append(str(label_meta["phase"]))
            event_types.append(str(label_meta["event_type"]))
            possession_team_ids.append(str(label_meta["possession_team_id"]))
            possession_available_values.append(bool(label_meta["possession_available"]))

    if not past_rows:
        past = torch.empty((0, history_steps, N_ENTITIES, len(FEATURE_NAMES)), dtype=torch.float32)
        future = torch.empty((0, horizon_steps, N_ENTITIES, 2), dtype=torch.float32)
        past_mask = torch.empty((0, history_steps, N_ENTITIES), dtype=torch.bool)
        future_mask = torch.empty((0, horizon_steps, N_ENTITIES), dtype=torch.bool)
        entity_type = torch.empty((0, N_ENTITIES), dtype=torch.long)
        team_id = torch.empty((0, N_ENTITIES), dtype=torch.long)
    else:
        past = torch.from_numpy(np.stack(past_rows)).float()
        future = torch.from_numpy(np.stack(future_rows)).float()
        past_mask = torch.from_numpy(np.stack(past_masks)).bool()
        future_mask = torch.from_numpy(np.stack(future_masks)).bool()
        entity_type = torch.from_numpy(np.stack(entity_types)).long()
        team_id = torch.from_numpy(np.stack(team_ids)).long()

    return TrackingWindowTensorData(
        past=past,
        future_xy=future,
        past_mask=past_mask,
        future_mask=future_mask,
        entity_type=entity_type,
        team_id=team_id,
        match_id=match_ids,
        period=periods,
        start_frame=start_frames,
        label_frame=label_frames,
        phase=phases,
        event_type=event_types,
        possession_team_id=possession_team_ids,
        possession_available=possession_available_values,
        feature_names=list(FEATURE_NAMES),
        fps=float(fps_out),
        context_seconds=float(context_seconds),
        horizon_seconds=float(horizon_seconds),
        stride_seconds=float(stride_seconds),
    )
