"""Phase 1 data utilities for canonical tracking windows."""

from footballq.data.normalize import (
    denormalize_velocity_to_mps,
    denormalize_xy_to_meters,
    normalize_velocity_from_mps,
    normalize_xy_from_meters,
)
from footballq.data.td_jepa_dataset import (
    TDJEPAData,
    TDJEPADataset,
    build_td_jepa_examples,
    load_td_jepa_data,
    save_td_jepa_data,
)
from footballq.data.windows import (
    FEATURE_NAMES,
    N_ENTITIES,
    TrackingWindowDataset,
    TrackingWindowTensorData,
    build_tracking_windows,
    load_windows_pt,
    save_windows_pt,
)

__all__ = [
    "FEATURE_NAMES",
    "N_ENTITIES",
    "TrackingWindowDataset",
    "TrackingWindowTensorData",
    "TDJEPAData",
    "TDJEPADataset",
    "build_tracking_windows",
    "build_td_jepa_examples",
    "denormalize_velocity_to_mps",
    "denormalize_xy_to_meters",
    "load_windows_pt",
    "load_td_jepa_data",
    "normalize_velocity_from_mps",
    "normalize_xy_from_meters",
    "save_windows_pt",
    "save_td_jepa_data",
]
