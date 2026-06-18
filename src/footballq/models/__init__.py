"""Deterministic trajectory-prediction baselines."""

from footballq.models.constant_velocity import ConstantVelocityBaseline, predict_constant_velocity
from footballq.models.mlp_baseline import MLPBaseline
from footballq.models.soccer_state_encoder import SoccerStateEncoder
from footballq.models.st_transformer import SpatioTemporalTransformerBaseline
from footballq.models.td_jepa import MotionEncoder, SoccerTDJEPA

__all__ = [
    "ConstantVelocityBaseline",
    "MLPBaseline",
    "MotionEncoder",
    "SoccerStateEncoder",
    "SoccerTDJEPA",
    "SpatioTemporalTransformerBaseline",
    "predict_constant_velocity",
]
