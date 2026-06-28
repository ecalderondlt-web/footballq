from footballq.data.td_jepa_dataset import build_td_jepa_examples
from footballq.synthetic.generate import generate_synthetic_tracking


def test_nonoverlap_has_zero_shared_time_indices():
    tracking = generate_synthetic_tracking(duration_s=4.0, fps=10.0)
    data = build_td_jepa_examples(
        tracking,
        fps_out=10.0,
        context_seconds=0.5,
        delta_seconds=0.2,
        objective_mode="future_nonoverlap_context_only",
        prediction_gap_seconds=0.2,
        feature_view="geometry_only",
    )
    assert data.objective_mode == "future_nonoverlap_context_only"
    assert set(data.feature_names).isdisjoint({"is_possession_team", "has_possession"})
    assert not data.delta_mask.any()
    for context, target in zip(
        data.context_frame_indices.tolist(),
        data.target_frame_indices.tolist(),
        strict=True,
    ):
        assert set(context).isdisjoint(target)
