import pandas as pd

from footballq.data.td_jepa_dataset import build_td_jepa_examples
from footballq.synthetic.generate import generate_synthetic_tracking


def test_td_jepa_dataset_shapes_and_23_entities():
    tracking = generate_synthetic_tracking(duration_s=4.0, fps=10.0)
    data = build_td_jepa_examples(
        tracking,
        fps_out=10.0,
        context_seconds=1.0,
        delta_seconds=0.2,
        stride_seconds=0.2,
    )
    assert data.state_t.shape[1:] == (10, 23, len(data.feature_names))
    assert data.state_t_plus_delta.shape[1:] == (10, 23, len(data.feature_names))
    assert data.delta_state.shape[1:] == (2, 23, len(data.feature_names))
    assert data.mask_t.shape[1:] == (10, 23)
    assert data.delta_frames == 2


def test_td_jepa_shift_uses_exact_delta_frames():
    tracking = generate_synthetic_tracking(duration_s=4.0, fps=10.0)
    data = build_td_jepa_examples(
        tracking,
        fps_out=10.0,
        context_seconds=1.0,
        delta_seconds=0.2,
        stride_seconds=0.2,
    )
    x_idx = data.feature_names.index("x_norm")
    expected = data.state_t[0, data.delta_frames :, :, x_idx]
    actual = data.state_t_plus_delta[0, : -data.delta_frames, :, x_idx]
    assert expected.shape == actual.shape
    assert (expected - actual).abs().max() < 1e-6


def test_td_jepa_features_do_not_include_labels_or_targets():
    tracking = generate_synthetic_tracking(duration_s=2.0, fps=10.0)
    tracking["event_type"] = "shot"
    tracking["phase"] = "attack"
    data = build_td_jepa_examples(tracking, fps_out=10.0, context_seconds=0.5, delta_seconds=0.2)
    forbidden = {"event_type", "phase", "future_xy", "target", "label"}
    assert forbidden.isdisjoint(set(data.feature_names))


def test_td_jepa_dataset_split_has_multiple_match_ids():
    frames = [
        generate_synthetic_tracking(match_id=f"m{idx}", duration_s=2.0, fps=10.0, seed=idx)
        for idx in range(3)
    ]
    data = build_td_jepa_examples(pd.concat(frames, ignore_index=True), fps_out=10.0)
    assert sorted(set(data.match_id)) == ["m0", "m1", "m2"]
