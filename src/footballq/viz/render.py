"""Render canonical tracking data as minimap clips."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import animation

from footballq.constants import AGENT_BALL
from footballq.viz.pitch import draw_pitch

TEAM_COLORS = {
    "home": "#1f77b4",
    "away": "#d62728",
    "ball": "#111111",
}


def _select_writer(
    format_name: str,
    out_path: Path,
    fps: float,
) -> tuple[animation.AbstractMovieWriter, Path]:
    suffix = out_path.suffix.lower().lstrip(".")
    requested = suffix or format_name.lower()
    if requested == "mp4" and animation.writers.is_available("ffmpeg"):
        return animation.FFMpegWriter(fps=fps, bitrate=1800), out_path.with_suffix(".mp4")
    return animation.PillowWriter(fps=fps), out_path.with_suffix(".gif")


def render_tracking_clip(
    tracking_df: pd.DataFrame,
    out_path: str | Path,
    start_time_s: float,
    duration_s: float,
    fps: float = 10.0,
    trail_s: float = 1.0,
    format: str = "mp4",
) -> Path:
    """Render a tracer/minimap clip from canonical tracking rows."""

    end_time_s = start_time_s + duration_s
    df = tracking_df[
        (tracking_df["time_s"] >= start_time_s) & (tracking_df["time_s"] <= end_time_s)
    ].copy()
    if df.empty:
        raise ValueError("No tracking rows fall inside the requested clip window.")

    frames = (
        df[["period", "frame_id", "time_s"]]
        .drop_duplicates()
        .sort_values(["time_s", "frame_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if frames.empty:
        raise ValueError("No frames available to render.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer, actual_path = _select_writer(format, out_path, fps)

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=100)
    draw_pitch(ax=ax)
    def update(frame_idx: int) -> list[object]:
        ax.clear()
        draw_pitch(ax=ax)
        frame = frames.iloc[frame_idx]
        frame_rows = df[
            (df["period"] == frame["period"])
            & (df["frame_id"] == frame["frame_id"])
            & (df["time_s"] == frame["time_s"])
        ]
        recent = df[
            (df["time_s"] >= float(frame["time_s"]) - trail_s)
            & (df["time_s"] <= float(frame["time_s"]))
        ]
        for agent_id, trail in recent.groupby("agent_id", sort=False):
            if agent_id == "ball":
                continue
            if len(trail) > 1:
                color = TEAM_COLORS.get(str(trail["team_id"].iloc[-1]), "#666666")
                ax.plot(trail["x_m"], trail["y_m"], color=color, lw=0.8, alpha=0.22)

        players = frame_rows[frame_rows["agent_type"] != AGENT_BALL]
        for team_id, team_rows in players.groupby("team_id", dropna=False):
            ax.scatter(
                team_rows["x_m"],
                team_rows["y_m"],
                s=46,
                color=TEAM_COLORS.get(str(team_id), "#6f6f6f"),
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )

        ball = frame_rows[frame_rows["agent_type"] == AGENT_BALL]
        if not ball.empty:
            ax.scatter(ball["x_m"], ball["y_m"], s=24, color=TEAM_COLORS["ball"], zorder=6)

        match_id = frame_rows["match_id"].iloc[0] if not frame_rows.empty else ""
        title = ax.text(
            1.5,
            2.5,
            f"{match_id} | period {frame['period']} | "
            f"t={float(frame['time_s']):.1f}s | frame {int(frame['frame_id'])}",
            fontsize=9,
            color="#222222",
            ha="left",
            va="top",
        )
        return [title]

    clip = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000.0 / fps,
        blit=False,
    )
    try:
        clip.save(actual_path, writer=writer)
    finally:
        plt.close(fig)
    return actual_path
