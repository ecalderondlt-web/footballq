import torch

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.probes.dataset import build_probe_dataset
from footballq.synthetic.generate import generate_synthetic_tracking


def test_leaked_probe_is_classified_as_sanity_check(tmp_path):
    tracking = generate_synthetic_tracking(match_id="probe_validity", duration_s=4.0, fps=10.0)
    tracking["possession_team_id"] = "home"
    windows = build_tracking_windows(tracking, context_seconds=1.0, horizon_seconds=1.0)
    windows_path = save_windows_pt(windows, tmp_path / "windows.pt")
    embeddings = tmp_path / "embeddings.pt"
    torch.save(
        {
            "z": torch.randn(len(windows.match_id), 4),
            "match_id": windows.match_id,
            "period": windows.period,
            "frame_t": windows.start_frame,
            "sample_id": windows.sample_id,
            "feature_view": "full_state_legacy",
        },
        embeddings,
    )
    data = build_probe_dataset(embeddings, windows_path, ["possession_team"])
    assert data.metadata["target_validity_classes"]["possession_team"] == "leakage_sanity_check"
