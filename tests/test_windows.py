from footballq.processing.features import compute_features
from footballq.processing.windows import build_windows
from footballq.synthetic.generate import generate_synthetic_tracking


def test_window_shape_correctness():
    tracking = generate_synthetic_tracking(duration_s=3.0, fps=2.0)
    features = compute_features(tracking)
    batch = build_windows(
        tracking,
        features,
        history_s=1.0,
        future_s=1.0,
        fps=2.0,
        max_agents=23,
    )
    assert batch.X_history.shape[1:] == (2, 23, len(batch.feature_names))
    assert batch.Y_future.shape[1:] == (2, 23, len(batch.target_names))
    assert batch.agent_mask_history.shape[1:] == (2, 23)
    assert batch.agent_ids.shape[1] == 23

