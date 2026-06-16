import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.io.metrica import MetricaAdapter  # noqa: E402
from footballq.processing.features import compute_features  # noqa: E402
from footballq.viz.render import render_tracking_clip  # noqa: E402

if __name__ == "__main__":
    raw_dir = ROOT / "data" / "raw" / "metrica"
    out_dir = ROOT / "data" / "processed" / "metrica"
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = MetricaAdapter(raw_dir=raw_dir, match_id="sample_game_1")
    tracking = adapter.load_tracking()
    events = adapter.load_events()
    features = compute_features(tracking)

    tracking_path = out_dir / "tracking.parquet"
    features_path = out_dir / "features.parquet"
    events_path = out_dir / "events.parquet"
    tracking.to_parquet(tracking_path, index=False)
    features.to_parquet(features_path, index=False)
    if not events.empty:
        events.to_parquet(events_path, index=False)

    clip_path = render_tracking_clip(
        tracking,
        ROOT / "artifacts" / "metrica_clip.gif",
        start_time_s=float(tracking["time_s"].min()),
        duration_s=10.0,
        fps=10.0,
        format="gif",
    )
    print(f"tracking: {tracking_path}")
    print(f"features: {features_path}")
    if not events.empty:
        print(f"events: {events_path}")
    print(f"clip: {clip_path}")

