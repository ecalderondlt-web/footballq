from footballq.cli import run_synthetic_demo


def test_synthetic_demo_outputs(tmp_path):
    outputs = run_synthetic_demo(tmp_path / "synthetic_demo", duration_s=4.0, fps=5.0)
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0

