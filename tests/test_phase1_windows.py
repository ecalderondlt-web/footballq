import torch

from footballq.data.normalize import denormalize_xy_to_meters, normalize_xy_from_meters
from footballq.data.windows import N_ENTITIES, build_tracking_windows
from footballq.synthetic.generate import generate_synthetic_tracking


def test_synthetic_tracking_has_23_entities_per_frame():
    tracking = generate_synthetic_tracking(duration_s=2.0, fps=10.0)
    counts = tracking.groupby(["match_id", "period", "frame_id"])["agent_id"].nunique()
    assert counts.min() == N_ENTITIES
    assert counts.max() == N_ENTITIES


def test_phase1_window_tensors_have_fixed_shapes():
    tracking = generate_synthetic_tracking(duration_s=5.0, fps=10.0)
    windows = build_tracking_windows(
        tracking,
        fps_out=10.0,
        context_seconds=2.0,
        horizon_seconds=2.0,
        stride_seconds=0.2,
    )
    assert windows.past.shape[1:] == (20, 23, len(windows.feature_names))
    assert windows.future_xy.shape[1:] == (20, 23, 2)
    assert windows.past_mask.shape[1:] == (20, 23)
    assert windows.future_mask.shape[1:] == (20, 23)
    assert windows.past_mask.all()


def test_centered_normalization_round_trips():
    xy_m = torch.tensor([[0.0, 0.0], [52.5, 34.0], [105.0, 68.0]])
    xy_norm = normalize_xy_from_meters(xy_m)
    assert torch.allclose(denormalize_xy_to_meters(xy_norm), xy_m)
