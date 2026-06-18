import pandas as pd

from footballq.data.windows import build_tracking_windows
from footballq.models.constant_velocity import ConstantVelocityBaseline
from footballq.schema import canonical_tracking_frame
from footballq.training.metrics import compute_metrics


def _linear_tracking(num_frames: int = 8, fps: float = 2.0) -> pd.DataFrame:
    rows = []
    agents = [("ball", "ball", "ball")]
    agents.extend((f"home_{idx:02d}", "player", "home") for idx in range(1, 12))
    agents.extend((f"away_{idx:02d}", "player", "away") for idx in range(1, 12))
    for frame in range(num_frames):
        time_s = frame / fps
        for entity_idx, (agent_id, agent_type, team_id) in enumerate(agents):
            rows.append(
                {
                    "match_id": "linear",
                    "dataset": "synthetic",
                    "period": 1,
                    "frame_id": frame,
                    "time_s": time_s,
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "team_id": team_id,
                    "player_id": agent_id if agent_type == "player" else pd.NA,
                    "jersey_number": entity_idx if agent_type == "player" else pd.NA,
                    "role": agent_type,
                    "x_m": 10.0 + entity_idx + time_s,
                    "y_m": 5.0 + entity_idx * 0.5,
                    "is_visible": True,
                }
            )
    return canonical_tracking_frame(pd.DataFrame(rows))


def test_constant_velocity_baseline_low_error_on_linear_motion():
    windows = build_tracking_windows(
        _linear_tracking(),
        fps_out=2.0,
        context_seconds=1.0,
        horizon_seconds=1.0,
        stride_seconds=0.5,
    )
    model = ConstantVelocityBaseline(
        horizon_steps=windows.horizon_steps,
        fps=windows.fps,
        feature_names=windows.feature_names,
    )
    batch = {
        "past": windows.past,
        "past_mask": windows.past_mask,
    }
    pred = model(batch)
    metrics = compute_metrics(
        pred,
        windows.future_xy,
        windows.future_mask,
        windows.entity_type,
        windows.team_id,
    )
    assert metrics["all_entity_ADE_m"] < 1e-4
