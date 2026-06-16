from footballq.synthetic.generate import generate_synthetic_tracking
from footballq.viz.render import render_tracking_clip


def test_renderer_smoke(tmp_path):
    tracking = generate_synthetic_tracking(duration_s=2.0, fps=5.0)
    out = render_tracking_clip(
        tracking,
        tmp_path / "clip.gif",
        start_time_s=0.0,
        duration_s=1.0,
        fps=5.0,
        format="gif",
    )
    assert out.exists()
    assert out.stat().st_size > 0

