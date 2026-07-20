"""Deterministic trajectory-prediction baselines."""

from footballq.models.constant_velocity import ConstantVelocityBaseline, predict_constant_velocity
from footballq.models.event_context_residual import (
    EVENT_CONTEXT_FAMILIES,
    FrozenTrackingEventResidual,
    event_context_residual_loss,
)
from footballq.models.mlp_baseline import MLPBaseline
from footballq.models.soccer_state_encoder import SoccerStateEncoder
from footballq.models.st_transformer import SpatioTemporalTransformerBaseline
from footballq.models.statsbomb_event_encoder import (
    FreezeFrameSetEncoder,
    StatsBombEventEncoder,
    statsbomb_event_loss,
)
from footballq.models.td_jepa import MotionEncoder, SoccerTDJEPA

__all__ = [
    "ConstantVelocityBaseline",
    "EVENT_CONTEXT_FAMILIES",
    "FreezeFrameSetEncoder",
    "FrozenTrackingEventResidual",
    "MLPBaseline",
    "MotionEncoder",
    "SoccerStateEncoder",
    "SoccerTDJEPA",
    "SpatioTemporalTransformerBaseline",
    "StatsBombEventEncoder",
    "event_context_residual_loss",
    "predict_constant_velocity",
    "statsbomb_event_loss",
]
