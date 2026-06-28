from footballq.data.windows import build_tracking_windows
from footballq.probes.labels import derive_probe_targets
from footballq.synthetic.generate import generate_synthetic_tracking


def test_global_x_displacement_uses_geometric_names():
    tracking = generate_synthetic_tracking(duration_s=4.0, fps=10.0)
    windows = build_tracking_windows(tracking, context_seconds=1.0, horizon_seconds=1.0)
    derived = derive_probe_targets(
        windows,
        ["future_ball_dx_global_m", "future_ball_global_x_bucket"],
    )
    assert "future_ball_dx_global_m" in derived.targets
    assert "future_ball_global_x_bucket" in derived.targets


def test_attack_relative_progression_unavailable_without_direction():
    tracking = generate_synthetic_tracking(duration_s=4.0, fps=10.0)
    windows = build_tracking_windows(tracking, context_seconds=1.0, horizon_seconds=1.0)
    derived = derive_probe_targets(
        windows,
        ["future_ball_progression_attacking_m", "future_ball_progression_attacking_bucket"],
    )
    assert derived.targets == {}
    assert any("attacking-direction metadata" in warning for warning in derived.warnings)
