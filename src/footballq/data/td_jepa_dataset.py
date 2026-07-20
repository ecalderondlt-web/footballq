"""TD-JEPA shifted-state dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from footballq.data.normalize import normalize_xy_from_meters
from footballq.data.windows import (
    ENTITY_BALL,
    FEATURE_NAMES,
    N_ENTITIES,
    TEAM_AWAY,
    TEAM_HOME,
    _agents_for_period,
    _row_features,
    _selected_times,
    _standardize_columns,
    _static_entity_arrays,
    _with_causal_velocity,
)
from footballq.repro.feature_views import (
    FULL_STATE_LEGACY,
    POSITION_ONLY,
    apply_feature_view,
    feature_view_names,
)
from footballq.repro.identity import sample_ids_from_components
from footballq.repro.splits import split_manifest_metadata

LEGACY_SHIFTED_OVERLAP = "legacy_shifted_overlap"
FUTURE_NONOVERLAP_CONTEXT_ONLY = "future_nonoverlap_context_only"
OBJECTIVE_MODES = {LEGACY_SHIFTED_OVERLAP, FUTURE_NONOVERLAP_CONTEXT_ONLY}


@dataclass
class TDJEPAData:
    """Tensor payload for temporal-difference JEPA examples."""

    state_t: torch.Tensor
    state_t_plus_delta: torch.Tensor
    delta_state: torch.Tensor
    mask_t: torch.Tensor
    mask_t_plus_delta: torch.Tensor
    delta_mask: torch.Tensor
    entity_type: torch.Tensor
    team_id: torch.Tensor
    match_id: list[str]
    period: list[int]
    frame_t: list[int]
    delta_frames: int
    feature_names: list[str]
    fps: float
    context_seconds: float
    delta_seconds: float
    stride_seconds: float
    sample_id: list[str] | None = None
    objective_mode: str = LEGACY_SHIFTED_OVERLAP
    prediction_gap_frames: int = 0
    feature_view: str = FULL_STATE_LEGACY
    context_frame_indices: torch.Tensor | None = None
    target_frame_indices: torch.Tensor | None = None
    delta_frame_indices: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.period) != len(self.match_id):
            raise ValueError("period must have the same length as match_id.")
        if self.sample_id is None:
            self.sample_id = sample_ids_from_components(self.match_id, self.period, self.frame_t)
        if self.context_frame_indices is None:
            self.context_frame_indices = torch.empty(
                (len(self.match_id), self.context_steps), dtype=torch.long
            )
        if self.target_frame_indices is None:
            self.target_frame_indices = torch.empty(
                (len(self.match_id), self.context_steps), dtype=torch.long
            )
        if self.delta_frame_indices is None:
            self.delta_frame_indices = torch.empty(
                (len(self.match_id), self.delta_steps), dtype=torch.long
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_t": self.state_t,
            "state_t_plus_delta": self.state_t_plus_delta,
            "delta_state": self.delta_state,
            "mask_t": self.mask_t,
            "mask_t_plus_delta": self.mask_t_plus_delta,
            "delta_mask": self.delta_mask,
            "entity_type": self.entity_type,
            "team_id": self.team_id,
            "match_id": self.match_id,
            "period": self.period,
            "frame_t": self.frame_t,
            "sample_id": self.sample_id,
            "delta_frames": self.delta_frames,
            "feature_names": self.feature_names,
            "fps": self.fps,
            "context_seconds": self.context_seconds,
            "delta_seconds": self.delta_seconds,
            "stride_seconds": self.stride_seconds,
            "objective_mode": self.objective_mode,
            "prediction_gap_frames": self.prediction_gap_frames,
            "feature_view": self.feature_view,
            "context_frame_indices": self.context_frame_indices,
            "target_frame_indices": self.target_frame_indices,
            "delta_frame_indices": self.delta_frame_indices,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TDJEPAData:
        return cls(
            state_t=payload["state_t"].float(),
            state_t_plus_delta=payload["state_t_plus_delta"].float(),
            delta_state=payload["delta_state"].float(),
            mask_t=payload["mask_t"].bool(),
            mask_t_plus_delta=payload["mask_t_plus_delta"].bool(),
            delta_mask=payload["delta_mask"].bool(),
            entity_type=payload["entity_type"].long(),
            team_id=payload["team_id"].long(),
            match_id=[str(value) for value in payload["match_id"]],
            period=[
                int(value) for value in payload.get("period", [1 for _ in payload["match_id"]])
            ],
            frame_t=[int(value) for value in payload["frame_t"]],
            sample_id=(
                [str(value) for value in payload["sample_id"]] if "sample_id" in payload else None
            ),
            delta_frames=int(payload["delta_frames"]),
            feature_names=[str(value) for value in payload["feature_names"]],
            fps=float(payload["fps"]),
            context_seconds=float(payload["context_seconds"]),
            delta_seconds=float(payload["delta_seconds"]),
            stride_seconds=float(payload["stride_seconds"]),
            objective_mode=str(payload.get("objective_mode", LEGACY_SHIFTED_OVERLAP)),
            prediction_gap_frames=int(payload.get("prediction_gap_frames", 0)),
            feature_view=str(payload.get("feature_view", FULL_STATE_LEGACY)),
            context_frame_indices=payload.get("context_frame_indices"),
            target_frame_indices=payload.get("target_frame_indices"),
            delta_frame_indices=payload.get("delta_frame_indices"),
            metadata=dict(payload.get("metadata", {})),
        )

    @property
    def n_features(self) -> int:
        return int(self.state_t.shape[-1])

    @property
    def context_steps(self) -> int:
        return int(self.state_t.shape[1])

    @property
    def delta_steps(self) -> int:
        return int(self.delta_state.shape[1])


class TDJEPADataset(Dataset):
    """PyTorch dataset for TD-JEPA examples."""

    def __init__(self, data: TDJEPAData, indices: list[int] | None = None) -> None:
        self.data = data
        self.indices = list(range(len(data.match_id))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        idx = self.indices[index]
        return {
            "state_t": self.data.state_t[idx],
            "state_t_plus_delta": self.data.state_t_plus_delta[idx],
            "delta_state": self.data.delta_state[idx],
            "mask_t": self.data.mask_t[idx],
            "mask_t_plus_delta": self.data.mask_t_plus_delta[idx],
            "delta_mask": self.data.delta_mask[idx],
            "entity_type": self.data.entity_type[idx],
            "team_id": self.data.team_id[idx],
            "match_id": self.data.match_id[idx],
            "period": self.data.period[idx],
            "sample_id": self.data.sample_id[idx],
            "frame_t": self.data.frame_t[idx],
            "delta_frames": self.data.delta_frames,
            "objective_mode": self.data.objective_mode,
            "context_frame_indices": self.data.context_frame_indices[idx],
            "target_frame_indices": self.data.target_frame_indices[idx],
            "delta_frame_indices": self.data.delta_frame_indices[idx],
        }


def save_td_jepa_data(data: TDJEPAData, out: str | Path) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data.to_dict(), out_path)
    return out_path


def load_td_jepa_data(path: str | Path) -> TDJEPAData:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return TDJEPAData.from_dict(payload)


def _state_at_times(
    times: np.ndarray,
    indexed: dict[tuple[float, str], Any],
    agent_ids: list[str],
    entity_type: np.ndarray,
    team_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((len(times), N_ENTITIES, len(FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros((len(times), N_ENTITIES), dtype=bool)
    for t_idx, time_s in enumerate(times):
        for entity_idx, agent_id in enumerate(agent_ids):
            if not agent_id:
                continue
            row = indexed.get((float(time_s), str(agent_id)))
            if row is None or not bool(getattr(row, "visible", True)):
                continue
            state[t_idx, entity_idx] = _row_features(
                row,
                int(entity_type[entity_idx]),
                int(team_id[entity_idx]),
            )
            mask[t_idx, entity_idx] = True
    return state, mask


def _position_state_at_times(
    times: np.ndarray,
    period_df: pd.DataFrame,
    agent_ids: list[str],
    entity_type: np.ndarray,
    team_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the full-layout state without materializing unused dynamic channels."""

    state = np.zeros((len(times), N_ENTITIES, len(FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros((len(times), N_ENTITIES), dtype=bool)
    rows = period_df.drop_duplicates(["time_s", "agent_id"], keep="last")
    time_lookup = {float(value): index for index, value in enumerate(times)}
    agent_lookup = {str(value): index for index, value in enumerate(agent_ids) if value}
    time_index = rows["time_s"].astype(float).map(time_lookup)
    entity_index = rows["agent_id"].astype(str).map(agent_lookup)
    visible = rows["visible"].fillna(False).astype(bool)
    selected = time_index.notna() & entity_index.notna() & visible
    if not selected.any():
        return state, mask

    selected_rows = rows.loc[selected]
    selected_time = time_index.loc[selected].to_numpy(dtype=int)
    selected_entity = entity_index.loc[selected].to_numpy(dtype=int)
    xy = normalize_xy_from_meters(
        selected_rows[["x_m", "y_m"]].to_numpy(dtype=np.float32)
    )
    state[selected_time, selected_entity, :2] = xy
    state[selected_time, selected_entity, 4] = (
        entity_type[selected_entity] == ENTITY_BALL
    ).astype(np.float32)
    state[selected_time, selected_entity, 5] = (
        team_id[selected_entity] == TEAM_HOME
    ).astype(np.float32)
    state[selected_time, selected_entity, 6] = (
        team_id[selected_entity] == TEAM_AWAY
    ).astype(np.float32)
    state[selected_time, selected_entity, 9] = 1.0
    mask[selected_time, selected_entity] = True
    return state, mask


def build_td_jepa_examples(
    tracking_df: pd.DataFrame,
    fps_out: float = 10.0,
    context_seconds: float = 1.0,
    delta_seconds: float = 0.2,
    stride_seconds: float = 0.2,
    objective_mode: str = LEGACY_SHIFTED_OVERLAP,
    prediction_gap_seconds: float = 0.0,
    feature_view: str = FULL_STATE_LEGACY,
    split_manifest_path: str | Path | None = None,
    scientific_mode: bool = False,
) -> TDJEPAData:
    """Build shifted temporal examples from canonical tracking rows."""

    if objective_mode not in OBJECTIVE_MODES:
        raise ValueError(f"Unknown TD-JEPA objective_mode {objective_mode!r}.")
    source = _with_causal_velocity(_standardize_columns(tracking_df))
    context_steps = max(1, int(round(context_seconds * fps_out)))
    delta_frames = max(1, int(round(delta_seconds * fps_out)))
    stride_steps = max(1, int(round(stride_seconds * fps_out)))
    prediction_gap_frames = max(0, int(round(prediction_gap_seconds * fps_out)))
    if objective_mode == LEGACY_SHIFTED_OVERLAP:
        total_steps = context_steps + delta_frames
    else:
        total_steps = context_steps + prediction_gap_frames + context_steps

    states_t: list[np.ndarray] = []
    states_target: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    masks_t: list[np.ndarray] = []
    masks_target: list[np.ndarray] = []
    masks_delta: list[np.ndarray] = []
    entity_types: list[np.ndarray] = []
    team_ids: list[np.ndarray] = []
    match_ids: list[str] = []
    periods: list[int] = []
    frame_ts: list[int] = []
    context_frame_rows: list[list[int]] = []
    target_frame_rows: list[list[int]] = []
    delta_frame_rows: list[list[int]] = []
    selected_feature_names = feature_view_names(list(FEATURE_NAMES), feature_view)

    for (match_id, period), period_df in source.groupby(
        ["match_id", "period"],
        dropna=False,
        sort=False,
    ):
        period_df = period_df.copy()
        times = _selected_times(period_df["time_s"].to_numpy(dtype=float), fps_out=fps_out)
        if len(times) < total_steps:
            continue
        period_df = period_df[period_df["time_s"].astype(float).isin({float(t) for t in times})]
        agent_ids = _agents_for_period(period_df)
        entity_type_arr, team_id_arr = _static_entity_arrays(agent_ids, period_df)
        if feature_view == POSITION_ONLY:
            period_state, period_mask = _position_state_at_times(
                times,
                period_df,
                agent_ids,
                entity_type_arr,
                team_id_arr,
            )
        else:
            indexed = {
                (float(row.time_s), str(row.agent_id)): row
                for row in period_df.itertuples(index=False)
            }
            period_state, period_mask = _state_at_times(
                times,
                indexed,
                agent_ids,
                entity_type_arr,
                team_id_arr,
            )
        frame_by_time = (
            period_df.drop_duplicates("time_s").set_index("time_s")["frame_id"].to_dict()
        )
        segmented = "temporal_segment_id" in period_df.columns
        if segmented:
            segment_by_time = (
                period_df.drop_duplicates("time_s")
                .set_index("time_s")["temporal_segment_id"]
                .to_dict()
            )
            stride_origin = int(period_df["temporal_stride_origin_frame_id"].iloc[0])

        for start in range(0, len(times) - total_steps + 1, stride_steps):
            context_times = times[start : start + context_steps]
            if segmented:
                frame_start = int(frame_by_time[float(context_times[0])])
                if (frame_start - stride_origin) % stride_steps != 0:
                    continue
                window_times = times[start : start + total_steps]
                window_segments = {
                    int(segment_by_time[float(value)]) for value in window_times
                }
                if len(window_segments) != 1:
                    continue
            if objective_mode == LEGACY_SHIFTED_OVERLAP:
                target_start = start + delta_frames
                target_times = times[start + delta_frames : start + delta_frames + context_steps]
                delta_times = times[start + context_steps : start + context_steps + delta_frames]
            else:
                target_start = start + context_steps + prediction_gap_frames
                target_times = times[target_start : target_start + context_steps]
                delta_times = times[start + context_steps : start + context_steps + delta_frames]
            if len(target_times) != context_steps or len(delta_times) != delta_frames:
                continue
            state_t = period_state[start : start + context_steps]
            mask_t = period_mask[start : start + context_steps]
            state_target = period_state[target_start : target_start + context_steps]
            mask_target = period_mask[target_start : target_start + context_steps]
            delta_start = start + context_steps
            delta_state = period_state[delta_start : delta_start + delta_frames]
            delta_mask = period_mask[delta_start : delta_start + delta_frames]
            if objective_mode == FUTURE_NONOVERLAP_CONTEXT_ONLY:
                delta_state = np.zeros_like(delta_state)
                delta_mask = np.zeros_like(delta_mask)
            if not mask_t.any() or not mask_target.any() or not delta_mask.any():
                if objective_mode != FUTURE_NONOVERLAP_CONTEXT_ONLY:
                    continue
            if objective_mode == FUTURE_NONOVERLAP_CONTEXT_ONLY and (
                not mask_t.any() or not mask_target.any()
            ):
                continue
            states_t.append(state_t)
            states_target.append(state_target)
            deltas.append(delta_state)
            masks_t.append(mask_t)
            masks_target.append(mask_target)
            masks_delta.append(delta_mask)
            entity_types.append(entity_type_arr)
            team_ids.append(team_id_arr)
            match_ids.append(str(match_id))
            periods.append(int(period))
            frame_ts.append(int(frame_by_time.get(float(context_times[0]), start)))
            context_frame_rows.append(
                [
                    int(frame_by_time.get(float(value), start + offset))
                    for offset, value in enumerate(context_times)
                ]
            )
            target_frame_rows.append(
                [
                    int(frame_by_time.get(float(value), start + offset))
                    for offset, value in enumerate(target_times)
                ]
            )
            if objective_mode == FUTURE_NONOVERLAP_CONTEXT_ONLY:
                delta_frame_rows.append([-1 for _ in range(delta_frames)])
            else:
                delta_frame_rows.append(
                    [
                        int(frame_by_time.get(float(value), start + context_steps + offset))
                        for offset, value in enumerate(delta_times)
                    ]
                )

    if not states_t:
        state_t = torch.empty((0, context_steps, N_ENTITIES, len(selected_feature_names)))
        target = torch.empty((0, context_steps, N_ENTITIES, len(selected_feature_names)))
        delta = torch.empty((0, delta_frames, N_ENTITIES, len(selected_feature_names)))
        mask_t = torch.empty((0, context_steps, N_ENTITIES), dtype=torch.bool)
        mask_target = torch.empty((0, context_steps, N_ENTITIES), dtype=torch.bool)
        delta_mask = torch.empty((0, delta_frames, N_ENTITIES), dtype=torch.bool)
        entity_type = torch.empty((0, N_ENTITIES), dtype=torch.long)
        team_id = torch.empty((0, N_ENTITIES), dtype=torch.long)
        context_frame_indices = torch.empty((0, context_steps), dtype=torch.long)
        target_frame_indices = torch.empty((0, context_steps), dtype=torch.long)
        delta_frame_indices = torch.empty((0, delta_frames), dtype=torch.long)
    else:
        state_t = torch.from_numpy(np.stack(states_t)).float()
        target = torch.from_numpy(np.stack(states_target)).float()
        delta = torch.from_numpy(np.stack(deltas)).float()
        state_t, _ = apply_feature_view(state_t, list(FEATURE_NAMES), feature_view)
        target, _ = apply_feature_view(target, list(FEATURE_NAMES), feature_view)
        delta, _ = apply_feature_view(delta, list(FEATURE_NAMES), feature_view)
        mask_t = torch.from_numpy(np.stack(masks_t)).bool()
        mask_target = torch.from_numpy(np.stack(masks_target)).bool()
        delta_mask = torch.from_numpy(np.stack(masks_delta)).bool()
        entity_type = torch.from_numpy(np.stack(entity_types)).long()
        team_id = torch.from_numpy(np.stack(team_ids)).long()
        context_frame_indices = torch.tensor(context_frame_rows, dtype=torch.long)
        target_frame_indices = torch.tensor(target_frame_rows, dtype=torch.long)
        delta_frame_indices = torch.tensor(delta_frame_rows, dtype=torch.long)

    return TDJEPAData(
        state_t=state_t,
        state_t_plus_delta=target,
        delta_state=delta,
        mask_t=mask_t,
        mask_t_plus_delta=mask_target,
        delta_mask=delta_mask,
        entity_type=entity_type,
        team_id=team_id,
        match_id=match_ids,
        period=periods,
        frame_t=frame_ts,
        delta_frames=delta_frames,
        feature_names=list(selected_feature_names),
        fps=float(fps_out),
        context_seconds=float(context_seconds),
        delta_seconds=float(delta_seconds),
        stride_seconds=float(stride_seconds),
        objective_mode=objective_mode,
        prediction_gap_frames=prediction_gap_frames,
        feature_view=feature_view,
        context_frame_indices=context_frame_indices,
        target_frame_indices=target_frame_indices,
        delta_frame_indices=delta_frame_indices,
        metadata={
            **split_manifest_metadata(split_manifest_path, scientific_mode=scientific_mode),
            "objective_mode": objective_mode,
            "feature_view": feature_view,
            "prediction_gap_frames": prediction_gap_frames,
            "legacy_alignment_allowed": False,
        },
    )
