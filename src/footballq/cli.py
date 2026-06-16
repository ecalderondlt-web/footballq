"""Command-line interface for footballq."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from footballq.io.metrica import MetricaAdapter
from footballq.processing.features import compute_features
from footballq.processing.quality import check_quality
from footballq.processing.windows import build_windows, export_windows_npz
from footballq.synthetic.generate import generate_synthetic_tracking
from footballq.viz.render import render_tracking_clip

app = typer.Typer(no_args_is_help=True)
console = Console()


def run_synthetic_demo(
    out_dir: Path = Path("artifacts/synthetic_demo"),
    duration_s: float = 12.0,
    fps: float = 10.0,
) -> dict[str, Path]:
    """Run the full synthetic Phase 1 pipeline."""

    out_dir.mkdir(parents=True, exist_ok=True)
    tracking = generate_synthetic_tracking(duration_s=duration_s, fps=fps)
    features = compute_features(tracking)
    windows = build_windows(
        tracking,
        features,
        history_s=2.0,
        future_s=2.0,
        fps=fps,
        max_agents=23,
    )

    tracking_path = out_dir / "tracking.parquet"
    features_path = out_dir / "features.parquet"
    tracking.to_parquet(tracking_path, index=False)
    features.to_parquet(features_path, index=False)
    windows_npz, window_meta = export_windows_npz(windows, out_dir / "windows")
    clip_path = render_tracking_clip(
        tracking,
        out_dir / "synthetic_clip.gif",
        start_time_s=0.0,
        duration_s=min(6.0, duration_s),
        fps=fps,
        trail_s=1.0,
        format="gif",
    )
    return {
        "tracking": tracking_path,
        "features": features_path,
        "windows": windows_npz,
        "window_meta": window_meta,
        "clip": clip_path,
    }


@app.command("synthetic-demo")
def synthetic_demo(
    out_dir: Path = typer.Option(Path("artifacts/synthetic_demo"), "--out-dir"),
    duration_s: float = typer.Option(12.0, "--duration-s"),
    fps: float = typer.Option(10.0, "--fps"),
) -> None:
    """Generate synthetic data, features, windows, and a rendered clip."""

    paths = run_synthetic_demo(out_dir=out_dir, duration_s=duration_s, fps=fps)
    for name, path in paths.items():
        console.print(f"[green]{name}[/green]: {path}")


@app.command("ingest-metrica")
def ingest_metrica(
    raw_dir: Path = typer.Option(Path("data/raw/metrica"), "--raw-dir"),
    out_dir: Path = typer.Option(Path("data/processed/metrica"), "--out-dir"),
    match_id: str = typer.Option("sample_game_1", "--match-id"),
) -> None:
    """Load local Metrica sample CSVs and write canonical parquet outputs."""

    adapter = MetricaAdapter(raw_dir=raw_dir, match_id=match_id)
    try:
        paths = adapter.write_outputs(out_dir)
    except FileNotFoundError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        console.print("Download Metrica sample data from https://github.com/metrica-sports/sample-data")
        raise typer.Exit(code=1) from exc
    for name, path in paths.items():
        console.print(f"[green]{name}[/green]: {path}")


@app.command("features")
def features_command(
    tracking: Path = typer.Option(..., "--tracking"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    """Compute movement and tactical features from canonical tracking parquet."""

    tracking_df = pd.read_parquet(tracking)
    features_df = compute_features(tracking_df)
    out.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(out, index=False)
    console.print(f"[green]features[/green]: {out}")


@app.command("quality")
def quality_command(
    tracking: Path = typer.Option(..., "--tracking"),
) -> None:
    """Run structured quality checks on canonical tracking parquet."""

    tracking_df = pd.read_parquet(tracking)
    report = check_quality(tracking_df)
    console.print(json.dumps(report, indent=2, default=str))


@app.command("render")
def render_command(
    tracking: Path = typer.Option(..., "--tracking"),
    out: Path = typer.Option(..., "--out"),
    start_time_s: float = typer.Option(..., "--start-time-s"),
    duration_s: float = typer.Option(..., "--duration-s"),
    fps: float = typer.Option(10.0, "--fps"),
) -> None:
    """Render a tracer/minimap clip."""

    tracking_df = pd.read_parquet(tracking)
    actual_path = render_tracking_clip(
        tracking_df,
        out,
        start_time_s=start_time_s,
        duration_s=duration_s,
        fps=fps,
        format=out.suffix.lstrip(".") or "mp4",
    )
    console.print(f"[green]clip[/green]: {actual_path}")


@app.command("export-windows")
def export_windows_command(
    tracking: Path = typer.Option(..., "--tracking"),
    features: Path | None = typer.Option(None, "--features"),
    out: Path = typer.Option(..., "--out"),
    history_s: float = typer.Option(5.0, "--history-s"),
    future_s: float = typer.Option(5.0, "--future-s"),
    fps: float = typer.Option(10.0, "--fps"),
    max_agents: int = typer.Option(23, "--max-agents"),
) -> None:
    """Export fixed-length model-prep windows."""

    tracking_df = pd.read_parquet(tracking)
    features_df = pd.read_parquet(features) if features else None
    batch = build_windows(
        tracking_df,
        features_df,
        history_s=history_s,
        future_s=future_s,
        fps=fps,
        max_agents=max_agents,
    )
    npz_path, meta_path = export_windows_npz(batch, out)
    table = Table("output", "path")
    table.add_row("windows", str(npz_path))
    table.add_row("metadata", str(meta_path))
    console.print(table)


if __name__ == "__main__":
    app()

