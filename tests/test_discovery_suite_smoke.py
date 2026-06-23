import json

import pandas as pd
import torch

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.discovery.report import run_discovery_suite
from footballq.synthetic.generate import generate_synthetic_tracking


def test_discovery_suite_smoke_runs_on_tiny_synthetic_data(tmp_path):
    frames = [
        generate_synthetic_tracking(match_id=f"suite_m{idx}", duration_s=5.0, fps=5.0, seed=idx)
        for idx in range(3)
    ]
    windows = build_tracking_windows(
        pd.concat(frames, ignore_index=True),
        fps_out=5.0,
        context_seconds=1.0,
        horizon_seconds=1.0,
        stride_seconds=0.2,
    )
    windows_path = save_windows_pt(windows, tmp_path / "windows.pt")
    torch.save(
        {
            "z": torch.randn(len(windows.match_id), 5, generator=torch.Generator().manual_seed(1)),
            "match_id": windows.match_id,
            "frame_t": windows.start_frame,
            "source_split": ["train" for _ in windows.match_id],
        },
        tmp_path / "embeddings.pt",
    )
    result = run_discovery_suite(
        tmp_path / "embeddings.pt",
        windows_path,
        tmp_path / "suite",
        delta_steps=[1],
        k_values=[2],
        fps=5.0,
        max_iter=3,
        fit_sample_size=100,
    )
    assert result["transition_dataset"]["num_matches"] == 3
    assert (tmp_path / "suite" / "summary.json").exists()
    assert (tmp_path / "suite" / "report.md").exists()
    delta_dir = tmp_path / "suite" / "delta_0p2s"
    assert (delta_dir / "clusters_k2.csv").exists()
    assert (delta_dir / "enrichment_k2.csv").exists()
    assert (delta_dir / "exemplars_k2.csv").exists()
    assert (delta_dir / "surprise_examples.csv").exists()
    summary = json.loads((tmp_path / "suite" / "summary.json").read_text())
    assert summary["recommended_k"] == 2
