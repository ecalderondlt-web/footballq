"""Create blinded diagnostic annotation CSV scaffolds."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import torch
from matplotlib import animation

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.normalize import denormalize_xy_to_meters  # noqa: E402
from footballq.data.windows import (  # noqa: E402
    BALL_INDEX,
    TEAM_AWAY,
    TEAM_HOME,
    TrackingWindowTensorData,
    load_windows_pt,
)
from footballq.viz.pitch import draw_pitch  # noqa: E402

TEAM_COLORS = {
    TEAM_HOME: "#1f77b4",
    TEAM_AWAY: "#d62728",
    "other": "#666666",
    "ball": "#111111",
}


def write_blinded_annotation_files(
    rows: list[dict[str, object]],
    annotator_csv: str | Path,
    key_csv: str | Path,
) -> tuple[Path, Path]:
    """Write blinded annotator rows and a separate key file."""

    annotator_path = Path(annotator_csv)
    key_path = Path(key_csv)
    annotator_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    annotator_fields = ["blind_id", "match_id", "period", "frame_t", "clip_path", "annotation"]
    key_fields = [
        "blind_id",
        "cluster_id",
        "latent_residual_score",
        "positive_control",
        "rank_source",
        "control_group",
        "control_match_reason",
    ]
    with annotator_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=annotator_fields)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(
                {
                    "blind_id": f"blind_{idx:05d}",
                    "match_id": row.get("match_id", ""),
                    "period": row.get("period", ""),
                    "frame_t": row.get("frame_t", ""),
                    "clip_path": row.get("clip_path", ""),
                    "annotation": "",
                }
            )
    with key_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=key_fields)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(
                {
                    "blind_id": f"blind_{idx:05d}",
                    "cluster_id": row.get("cluster_id", ""),
                    "latent_residual_score": row.get("latent_residual_score", ""),
                    "positive_control": row.get("positive_control", ""),
                    "rank_source": row.get("rank_source", ""),
                    "control_group": row.get("control_group", ""),
                    "control_match_reason": row.get("control_match_reason", ""),
                }
            )
    return annotator_path, key_path


def _is_low_residual_control(row: dict[str, object]) -> bool:
    return str(row.get("rank_source", "")).startswith("low_")


def _same_values(
    left: dict[str, object],
    right: dict[str, object],
    keys: tuple[str, ...],
) -> bool:
    return all(str(left.get(key, "")).strip() == str(right.get(key, "")).strip() for key in keys)


def _choose_control_row(
    positive: dict[str, object],
    controls: list[dict[str, object]],
    used: set[int],
) -> tuple[int | None, str]:
    strategies: list[tuple[str, tuple[str, ...]]] = [
        ("same_cluster_match_period", ("cluster_id", "match_id", "period")),
        ("same_cluster", ("cluster_id",)),
        ("same_match_period", ("match_id", "period")),
        ("lowest_available", ()),
    ]
    for reason, keys in strategies:
        for idx, control in enumerate(controls):
            if idx in used:
                continue
            if keys and not _same_values(positive, control, keys):
                continue
            return idx, reason
    return None, "none_available"


def select_rows_with_low_residual_controls(
    rows: list[dict[str, object]],
    *,
    positive_rows: int,
    controls_per_positive: int,
    shuffle_seed: int | None = 123,
) -> list[dict[str, object]]:
    """Select high-residual rows plus hidden low-residual controls for annotation."""

    positives = [row for row in rows if not _is_low_residual_control(row)]
    controls = [row for row in rows if _is_low_residual_control(row)]
    selected: list[dict[str, object]] = []
    used_controls: set[int] = set()
    for idx, positive in enumerate(positives[: int(positive_rows)]):
        group = f"group_{idx:05d}"
        positive_out = dict(positive)
        positive_out["positive_control"] = True
        positive_out["control_group"] = group
        positive_out["control_match_reason"] = "positive"
        selected.append(positive_out)
        for _ in range(max(0, int(controls_per_positive))):
            control_idx, reason = _choose_control_row(positive, controls, used_controls)
            if control_idx is None:
                break
            used_controls.add(control_idx)
            control_out = dict(controls[control_idx])
            control_out["positive_control"] = False
            control_out["control_group"] = group
            control_out["control_match_reason"] = reason
            selected.append(control_out)
    if shuffle_seed is not None:
        random.Random(int(shuffle_seed)).shuffle(selected)
    return selected


def _parse_int(value: object, *, field: str) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Missing {field}.")
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Could not parse {field}={value!r} as an integer.") from exc


def _window_identity(row: dict[str, object]) -> tuple[str, int, int]:
    return (
        str(row.get("match_id", "")).strip(),
        _parse_int(row.get("period", ""), field="period"),
        _parse_int(row.get("frame_t", row.get("start_frame", "")), field="frame_t"),
    )


def _window_index_by_identity(windows: TrackingWindowTensorData) -> dict[tuple[str, int, int], int]:
    lookup: dict[tuple[str, int, int], int] = {}
    for idx, (match_id, period, start_frame) in enumerate(
        zip(windows.match_id, windows.period, windows.start_frame, strict=True)
    ):
        lookup.setdefault((str(match_id), int(period), int(start_frame)), idx)
    return lookup


def _window_xy_m_and_mask(
    windows: TrackingWindowTensorData,
    window_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_to_idx = {name: idx for idx, name in enumerate(windows.feature_names)}
    try:
        xy_idx = [feature_to_idx["x_norm"], feature_to_idx["y_norm"]]
    except KeyError as exc:
        raise ValueError("Window features must include x_norm and y_norm.") from exc
    past_xy = windows.past[window_idx, :, :, xy_idx].float()
    future_xy = windows.future_xy[window_idx].float()
    xy_norm = torch.cat([past_xy, future_xy], dim=0)
    mask = torch.cat(
        [windows.past_mask[window_idx].bool(), windows.future_mask[window_idx].bool()],
        dim=0,
    )
    return denormalize_xy_to_meters(xy_norm).cpu(), mask.cpu()


def _plot_trails(
    ax: plt.Axes,
    xy_m: torch.Tensor,
    mask: torch.Tensor,
    team_id: torch.Tensor,
    frame_idx: int,
    history_steps: int,
    trail_steps: int,
) -> None:
    start = max(0, int(frame_idx) - int(trail_steps))
    for entity_idx in range(int(xy_m.shape[1])):
        if entity_idx == BALL_INDEX:
            continue
        valid = mask[start : frame_idx + 1, entity_idx].bool()
        if int(valid.sum().item()) < 2:
            continue
        path = xy_m[start : frame_idx + 1, entity_idx][valid]
        color = TEAM_COLORS.get(int(team_id[entity_idx].item()), TEAM_COLORS["other"])
        linestyle = "-" if frame_idx < history_steps else "--"
        ax.plot(
            path[:, 0].numpy(),
            path[:, 1].numpy(),
            color=color,
            lw=0.8,
            alpha=0.22,
            linestyle=linestyle,
        )


def _scatter_team(
    ax: plt.Axes,
    xy_frame: torch.Tensor,
    visible: torch.Tensor,
    team_id: torch.Tensor,
    team_value: int | None,
    color: str,
) -> None:
    entity_indices = torch.arange(int(xy_frame.shape[0]))
    selection = visible & (entity_indices != BALL_INDEX)
    if team_value is None:
        selection = selection & (team_id != TEAM_HOME) & (team_id != TEAM_AWAY)
    else:
        selection = selection & (team_id == int(team_value))
    if not bool(selection.any()):
        return
    xy = xy_frame[selection]
    ax.scatter(
        xy[:, 0].numpy(),
        xy[:, 1].numpy(),
        s=48,
        color=color,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )


def render_window_gif(
    windows: TrackingWindowTensorData,
    window_idx: int,
    out_path: str | Path,
    *,
    blind_id: str,
    fps: float = 5.0,
    trail_seconds: float = 0.8,
) -> Path:
    """Render a blinded diagnostic GIF from a processed tracking window."""

    out_path = Path(out_path).with_suffix(".gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xy_m, mask = _window_xy_m_and_mask(windows, int(window_idx))
    team_id = windows.team_id[int(window_idx)].cpu().long()
    history_steps = int(windows.history_steps)
    total_steps = int(xy_m.shape[0])
    frame_fps = float(windows.fps) if float(windows.fps) > 0 else 10.0
    render_fps = max(float(fps), 1.0)
    trail_steps = max(1, int(round(float(trail_seconds) * frame_fps)))
    start_frame = int(windows.start_frame[int(window_idx)])
    match_id = str(windows.match_id[int(window_idx)])
    period = int(windows.period[int(window_idx)])

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=90)

    def update(frame_idx: int) -> list[object]:
        ax.clear()
        draw_pitch(ax=ax)
        _plot_trails(ax, xy_m, mask, team_id, frame_idx, history_steps, trail_steps)
        xy_frame = xy_m[frame_idx]
        visible = mask[frame_idx].bool()
        _scatter_team(ax, xy_frame, visible, team_id, TEAM_HOME, TEAM_COLORS[TEAM_HOME])
        _scatter_team(ax, xy_frame, visible, team_id, TEAM_AWAY, TEAM_COLORS[TEAM_AWAY])
        _scatter_team(ax, xy_frame, visible, team_id, None, TEAM_COLORS["other"])
        if int(xy_frame.shape[0]) > BALL_INDEX and bool(visible[BALL_INDEX]):
            ball = xy_frame[BALL_INDEX]
            ax.scatter(
                [float(ball[0].item())],
                [float(ball[1].item())],
                s=30,
                color=TEAM_COLORS["ball"],
                zorder=6,
            )
        phase = "context" if frame_idx < history_steps else "horizon"
        frame_no = start_frame + int(frame_idx)
        title = ax.text(
            1.5,
            2.5,
            f"{blind_id} | {match_id} | period {period} | frame {frame_no} | {phase}",
            fontsize=9,
            color="#222222",
            ha="left",
            va="top",
        )
        return [title]

    clip = animation.FuncAnimation(
        fig,
        update,
        frames=total_steps,
        interval=1000.0 / render_fps,
        blit=False,
    )
    try:
        clip.save(out_path, writer=animation.PillowWriter(fps=render_fps))
    finally:
        plt.close(fig)
    return out_path


def attach_window_clip_paths(
    rows: list[dict[str, object]],
    windows: TrackingWindowTensorData,
    media_dir: str | Path,
    *,
    fps: float = 5.0,
    reuse_existing: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Render matched processed windows and attach annotator-facing clip paths."""

    lookup = _window_index_by_identity(windows)
    media_path = Path(media_dir)
    rendered_rows: list[dict[str, object]] = []
    missing: list[tuple[str, int, int]] = []
    rendered = 0
    reused = 0
    for idx, row in enumerate(rows):
        blind_id = f"blind_{idx:05d}"
        row_out = dict(row)
        identity = _window_identity(row_out)
        window_idx = lookup.get(identity)
        if window_idx is None:
            missing.append(identity)
            rendered_rows.append(row_out)
            continue
        target_path = media_path / f"{blind_id}.gif"
        if reuse_existing and target_path.exists():
            clip_path = target_path
            reused += 1
        else:
            clip_path = render_window_gif(
                windows,
                window_idx,
                target_path,
                blind_id=blind_id,
                fps=fps,
            )
            rendered += 1
        row_out["clip_path"] = str(clip_path)
        rendered_rows.append(row_out)
    stats: dict[str, object] = {
        "rendered_clips": rendered,
        "reused_clips": reused,
        "missing_windows": len(missing),
        "missing_window_identities": [
            {"match_id": match_id, "period": period, "frame_t": frame_t}
            for match_id, period, frame_t in missing
        ],
        "media_dir": str(media_path),
    }
    return rendered_rows, stats


def write_render_manifest(
    manifest_json: str | Path,
    *,
    examples_csv: str | Path | None,
    windows_path: str | Path | None,
    annotator_csv: str | Path,
    key_csv: str | Path,
    rows: list[dict[str, object]],
    stats: dict[str, object],
) -> Path:
    """Write a machine-readable provenance summary for a blinded media render."""

    manifest_path = Path(manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_by": "scripts/render_diagnostic_clips.py",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_status": "diagnostic_only",
        "examples_csv": str(examples_csv) if examples_csv is not None else None,
        "windows_path": str(windows_path) if windows_path is not None else None,
        "annotator_csv": str(annotator_csv),
        "key_csv": str(key_csv),
        "annotator_fields": [
            "blind_id",
            "match_id",
            "period",
            "frame_t",
            "clip_path",
            "annotation",
        ],
        "private_key_fields": [
            "blind_id",
            "cluster_id",
            "latent_residual_score",
            "positive_control",
            "rank_source",
            "control_group",
            "control_match_reason",
        ],
        "rows": len(rows),
        "rows_with_clip_path": sum(bool(row.get("clip_path", "")) for row in rows),
        "rows_without_clip_path": sum(not bool(row.get("clip_path", "")) for row in rows),
        "positive_control_counts": {
            str(value): sum(str(row.get("positive_control", "")) == str(value) for row in rows)
            for value in sorted({str(row.get("positive_control", "")) for row in rows})
        },
        "render_stats": stats,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=40)
    parser.add_argument("--blinded", action="store_true")
    parser.add_argument("--annotator-csv", type=Path, default=None)
    parser.add_argument("--key-csv", type=Path, default=None)
    parser.add_argument("--windows", type=Path, default=None)
    parser.add_argument("--media-dir", type=Path, default=None)
    parser.add_argument("--clip-fps", type=float, default=5.0)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--reuse-existing-media", action="store_true")
    parser.add_argument("--positive-rows", type=int, default=None)
    parser.add_argument("--controls-per-positive", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=123)
    return parser.parse_args()


def _read_examples(path: Path, max_rows: int | None = None) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[: int(max_rows)] if max_rows is not None else rows


def main() -> None:
    args = parse_args()
    if args.examples is not None:
        if args.out is None:
            raise ValueError("--examples requires --out.")
        needs_control_pool = args.positive_rows is not None or int(args.controls_per_positive) > 0
        rows = _read_examples(args.examples, None if needs_control_pool else args.max_rows)
        if needs_control_pool:
            rows = select_rows_with_low_residual_controls(
                rows,
                positive_rows=args.positive_rows
                if args.positive_rows is not None
                else args.max_rows,
                controls_per_positive=args.controls_per_positive,
                shuffle_seed=args.shuffle_seed,
            )
        else:
            rows = rows[: int(args.max_rows)]
        annotator_csv = args.out / "annotator" / "annotations.csv"
        key_csv = args.out / "private" / "annotation_key.csv"
    else:
        if args.annotator_csv is None or args.key_csv is None:
            raise ValueError("Set --examples/--out or --annotator-csv/--key-csv.")
        rows = []
        annotator_csv = args.annotator_csv
        key_csv = args.key_csv
    render_stats = None
    if args.windows is not None:
        windows = load_windows_pt(args.windows)
        if args.media_dir is not None:
            media_dir = args.media_dir
        elif args.out is not None:
            media_dir = args.out / "media"
        else:
            media_dir = annotator_csv.parent / "media"
        rows, render_stats = attach_window_clip_paths(
            rows,
            windows,
            media_dir,
            fps=args.clip_fps,
            reuse_existing=args.reuse_existing_media,
        )
    write_blinded_annotation_files(rows, annotator_csv, key_csv)
    manifest_json = None
    if render_stats is not None:
        if args.manifest_json is not None:
            manifest_json = args.manifest_json
        elif args.out is not None:
            manifest_json = args.out / "render_manifest.json"
        else:
            manifest_json = annotator_csv.parent / "render_manifest.json"
        write_render_manifest(
            manifest_json,
            examples_csv=args.examples,
            windows_path=args.windows,
            annotator_csv=annotator_csv,
            key_csv=key_csv,
            rows=rows,
            stats=render_stats,
        )
    print(f"annotator_csv: {annotator_csv}")
    print(f"key_csv: {key_csv}")
    print(f"rows: {len(rows)}")
    if render_stats is not None:
        print(f"media_dir: {render_stats['media_dir']}")
        print(f"rendered_clips: {render_stats['rendered_clips']}")
        print(f"reused_clips: {render_stats['reused_clips']}")
        print(f"missing_windows: {render_stats['missing_windows']}")
        print(f"manifest_json: {manifest_json}")


if __name__ == "__main__":
    main()
