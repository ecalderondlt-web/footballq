import pandas as pd

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.synthetic.generate import generate_synthetic_tracking
from footballq.training.train import train_from_config


def test_one_batch_training_smoke(tmp_path):
    frames = []
    for idx in range(2):
        frames.append(
            generate_synthetic_tracking(
                match_id=f"smoke_{idx}",
                duration_s=3.0,
                fps=5.0,
                seed=idx,
            )
        )
    tracking = pd.concat(frames, ignore_index=True)
    windows = build_tracking_windows(
        tracking,
        fps_out=5.0,
        context_seconds=1.0,
        horizon_seconds=1.0,
        stride_seconds=0.2,
    )
    windows_path = save_windows_pt(windows, tmp_path / "windows.pt")
    result = train_from_config(
        {
            "model": {"name": "mlp", "hidden_sizes": [32], "dropout": 0.0},
            "data": {"windows": str(windows_path), "batch_size": 4, "num_workers": 0},
            "split": {"val_fraction": 0.5, "test_fraction": 0.5},
            "training": {
                "seed": 7,
                "epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "device": "cpu",
                "run_root": str(tmp_path / "runs"),
                "max_train_batches": 1,
            },
        }
    )
    assert result["latest_checkpoint"].exists()
    assert (result["run_dir"] / "eval_test.json").exists()
