"""TD-JEPA shifted-state dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from footballq.data.windows import (
    FEATURE_NAMES,
    N_ENTITIES,
    _agents_for_period,
    _row_features,
    _selected_times,
    _standardize_columns,
    _static_entity_arrays,
    _with_causal_velocity,
)


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
    frame_t: list[int]
    delta_frames: int
    feature_names: list[str]
    fps: float
    context_seconds: float
    delta_seconds: float
    stride_seconds: float

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
            "frame_t": self.frame_t,
            "delta_frames": self.delta_frames,
            "feature_names": self.feature_names,
            "fps": self.fps,
            "context_seconds": self.context_seconds,
            "delta_seconds": self.delta_seconds,
            "stride_seconds": self.stride_seconds,
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
            frame_t=[int(value) for value in payload["frame_t"]],
            delta_frames=int(payload["delta_frames"]),
            feature_names=[str(value) for value in payload["feature_names"]],
            fps=float(payload["fps"]),
            context_seconds=float(payload["context_seconds"]),
            delta_seconds=float(payload["delta_seconds"]),
            stride_seconds=float(payload["stride_seconds"]),
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
            "frame_t": self.data.frame_t[idx],
            "delta_frames": self.data.delta_frames,
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


def build_td_jepa_examples(
    tracking_df: pd.DataFrame,
    fps_out: float = 10.0,
    context_seconds: float = 1.0,
    delta_seconds: float = 0.2,
    stride_seconds: float = 0.2,
) -> TDJEPAData:
    """Build shifted temporal examples from canonical tracking rows."""

    source = _with_causal_velocity(_standardize_columns(tracking_df))
    context_steps = max(1, int(round(context_seconds * fps_out)))
    delta_frames = max(1, int(round(delta_seconds * fps_out)))
    stride_steps = max(1, int(round(stride_seconds * fps_out)))
    total_steps = context_steps + delta_frames

    states_t: list[np.ndarray] = []
    states_target: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    masks_t: list[np.ndarray] = []
    masks_target: list[np.ndarray] = []
    masks_delta: list[np.ndarray] = []
    entity_types: list[np.ndarray] = []
    team_ids: list[np.ndarray] = []
    match_ids: list[str] = []
    frame_ts: list[int] = []

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
        indexed = {
            (float(row.time_s), str(row.agent_id)): row
            for row in period_df.itertuples(index=False)
        }
        frame_by_time = (
            period_df.drop_duplicates("time_s").set_index("time_s")["frame_id"].to_dict()
        )

        for start in range(0, len(times) - total_steps + 1, stride_steps):
            context_times = times[start : start + context_steps]
            target_times = times[start + delta_frames : start + delta_frames + context_steps]
            delta_times = times[start + context_steps : start + context_steps + delta_frames]
            if len(target_times) != context_steps or len(delta_times) != delta_frames:
                continue
            state_t, mask_t = _state_at_times(
                context_times,
                indexed,
                agent_ids,
                entity_type_arr,
                team_id_arr,
            )
            state_target, mask_target = _state_at_times(
                target_times,
                indexed,
                agent_ids,
                entity_type_arr,
                team_id_arr,
            )
            delta_state, delta_mask = _state_at_times(
                delta_times,
                indexed,
                agent_ids,
                entity_type_arr,
                team_id_arr,
            )
            if not mask_t.any() or not mask_target.any() or not delta_mask.any():
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
            frame_ts.append(int(frame_by_time.get(float(context_times[0]), start)))

    if not states_t:
        state_t = torch.empty((0, context_steps, N_ENTITIES, len(FEATURE_NAMES)))
        target = torch.empty((0, context_steps, N_ENTITIES, len(FEATURE_NAMES)))
        delta = torch.empty((0, delta_frames, N_ENTITIES, len(FEATURE_NAMES)))
        mask_t = torch.empty((0, context_steps, N_ENTITIES), dtype=torch.bool)
        mask_target = torch.empty((0, context_steps, N_ENTITIES), dtype=torch.bool)
        delta_mask = torch.empty((0, delta_frames, N_ENTITIES), dtype=torch.bool)
        entity_type = torch.empty((0, N_ENTITIES), dtype=torch.long)
        team_id = torch.empty((0, N_ENTITIES), dtype=torch.long)
    else:
        state_t = torch.from_numpy(np.stack(states_t)).float()
        target = torch.from_numpy(np.stack(states_target)).float()
        delta = torch.from_numpy(np.stack(deltas)).float()
        mask_t = torch.from_numpy(np.stack(masks_t)).bool()
        mask_target = torch.from_numpy(np.stack(masks_target)).bool()
        delta_mask = torch.from_numpy(np.stack(masks_delta)).bool()
        entity_type = torch.from_numpy(np.stack(entity_types)).long()
        team_id = torch.from_numpy(np.stack(team_ids)).long()

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
        frame_t=frame_ts,
        delta_frames=delta_frames,
        feature_names=list(FEATURE_NAMES),
        fps=float(fps_out),
        context_seconds=float(context_seconds),
        delta_seconds=float(delta_seconds),
        stride_seconds=float(stride_seconds),
    )
