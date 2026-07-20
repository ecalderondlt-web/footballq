"""Prepare several horizon-specific tracking window files from one raw load."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.windows import (  # noqa: E402
    TrackingWindowTensorData,
    build_tracking_windows,
    load_windows_pt,
    save_windows_pt,
)
from footballq.io.gfootball import GFootballAdapter  # noqa: E402
from footballq.io.idsse import IDSSEAdapter  # noqa: E402
from footballq.io.metrica import MetricaAdapter  # noqa: E402
from footballq.io.pff import PFFAdapter  # noqa: E402
from footballq.io.skillcorner import SkillCornerAdapter  # noqa: E402
from footballq.io.skillcorner_report import (  # noqa: E402
    SkillCornerRawMatch,
    discover_skillcorner_raw_matches,
    horizon_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["skillcorner", "gfootball", "pff", "idsse", "metrica"],
        required=True,
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="skillcorner_windows")
    parser.add_argument("--match-id", default="match_1")
    parser.add_argument("--fps-out", type=float, default=10.0)
    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--horizon-seconds", nargs="+", type=float, default=[2.0, 4.0, 6.0])
    parser.add_argument("--stride-seconds", type=float, default=0.2)
    parser.add_argument("--combined-load", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--match-ids", nargs="*", default=None)
    parser.add_argument("--skip-combine", action="store_true")
    return parser.parse_args()


def _load_source(source: str, raw: Path, match_id: str) -> pd.DataFrame:
    if source == "skillcorner":
        return SkillCornerAdapter(raw_dir=raw, match_id=match_id).load_tracking()
    if source == "gfootball":
        return GFootballAdapter(raw_dir=raw, match_id=match_id).load_tracking()
    if source == "pff":
        return PFFAdapter(raw_dir=raw, match_id=match_id).load_tracking()
    if source == "idsse":
        return IDSSEAdapter(raw_dir=raw, match_id=match_id).load_tracking()
    if source == "metrica":
        return MetricaAdapter(raw_dir=raw, match_id=match_id).load_tracking()
    raise ValueError(f"Unknown source: {source}")


def _concat_windows(parts: list[TrackingWindowTensorData]) -> TrackingWindowTensorData:
    if not parts:
        raise ValueError("Cannot concatenate an empty list of windows.")
    first = parts[0]
    return TrackingWindowTensorData(
        past=torch.cat([part.past for part in parts], dim=0),
        future_xy=torch.cat([part.future_xy for part in parts], dim=0),
        past_mask=torch.cat([part.past_mask for part in parts], dim=0),
        future_mask=torch.cat([part.future_mask for part in parts], dim=0),
        entity_type=torch.cat([part.entity_type for part in parts], dim=0),
        team_id=torch.cat([part.team_id for part in parts], dim=0),
        match_id=[value for part in parts for value in part.match_id],
        period=[value for part in parts for value in part.period],
        start_frame=[value for part in parts for value in part.start_frame],
        sample_id=[value for part in parts for value in part.sample_id],
        label_frame=[value for part in parts for value in part.label_frame],
        phase=[value for part in parts for value in part.phase],
        event_type=[value for part in parts for value in part.event_type],
        possession_team_id=[value for part in parts for value in part.possession_team_id],
        possession_available=[value for part in parts for value in part.possession_available],
        feature_names=list(first.feature_names),
        fps=first.fps,
        context_seconds=first.context_seconds,
        horizon_seconds=first.horizon_seconds,
        stride_seconds=first.stride_seconds,
        coordinate_mode=first.coordinate_mode,
    )


def _write_horizon(windows: TrackingWindowTensorData, out: Path) -> None:
    save_windows_pt(windows, out)
    counts = pd.Series(windows.match_id, dtype="string").value_counts().sort_index()
    print(
        f"wrote {len(windows.match_id):,} windows to {out} "
        f"with past={tuple(windows.past.shape)} future={tuple(windows.future_xy.shape)}"
    )
    print("windows_per_match:")
    for match_id, count in counts.items():
        print(f"- {match_id}: {int(count)}")


def _filter_raw_matches(
    raw_matches: list[SkillCornerRawMatch],
    match_ids: list[str] | None,
) -> list[SkillCornerRawMatch]:
    if not match_ids:
        return raw_matches
    requested = [str(value) for value in match_ids]
    by_match_id = {match.match_id: match for match in raw_matches}
    missing = [match_id for match_id in requested if match_id not in by_match_id]
    if missing:
        raise ValueError(
            "Requested SkillCorner match IDs were not found under the raw directory: "
            + ", ".join(missing)
        )
    return [by_match_id[match_id] for match_id in requested]


def _cached_window_periods(path: Path) -> list[int]:
    windows = load_windows_pt(path)
    return sorted(set(int(value) for value in windows.period))


def _cache_covers_raw_periods(cache_path: Path, raw_match: SkillCornerRawMatch) -> bool:
    raw_periods = set(raw_match.raw_periods)
    if not raw_periods:
        return True
    try:
        cached_periods = set(_cached_window_periods(cache_path))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"warning: could not read cache {cache_path}: {exc}; regenerating")
        return False
    missing_periods = sorted(raw_periods - cached_periods)
    if not missing_periods:
        return True
    print(
        "stale_cache_missing_periods: "
        f"{cache_path} raw_periods={','.join(str(value) for value in sorted(raw_periods))} "
        f"cached_periods={','.join(str(value) for value in sorted(cached_periods)) or 'none'} "
        f"missing={','.join(str(value) for value in missing_periods)}; regenerating"
    )
    return False


def _prepare_combined(args: argparse.Namespace) -> list[str]:
    tracking = _load_source(args.source, args.raw, args.match_id)
    match_ids = sorted(str(value) for value in tracking["match_id"].dropna().unique())
    print(f"source_matches: {len(match_ids)}")
    print(f"match_ids: {', '.join(match_ids)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for horizon_seconds in args.horizon_seconds:
        windows = build_tracking_windows(
            tracking,
            fps_out=args.fps_out,
            context_seconds=args.context_seconds,
            horizon_seconds=horizon_seconds,
            stride_seconds=args.stride_seconds,
        )
        if len(windows.match_id) == 0:
            raise RuntimeError(
                f"No windows were produced for horizon_seconds={horizon_seconds}. "
                "Check raw data duration and visibility coverage."
            )
        out = args.out_dir / f"{args.prefix}_{horizon_label(horizon_seconds)}.pt"
        _write_horizon(windows, out)
    return match_ids


def _prepare_skillcorner_per_match(args: argparse.Namespace) -> list[str]:
    raw_matches = _filter_raw_matches(
        discover_skillcorner_raw_matches(args.raw),
        args.match_ids,
    )
    if not raw_matches:
        raise FileNotFoundError(f"No SkillCorner tracking files found under {args.raw}.")
    match_ids = [match.match_id for match in raw_matches]
    print(f"source_matches: {len(match_ids)}")
    print(f"match_ids: {', '.join(match_ids)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.out_dir / ".skillcorner_window_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_by_horizon: dict[float, list[Path]] = {
        float(value): [] for value in args.horizon_seconds
    }
    for raw_match in raw_matches:
        print(f"preparing_match: {raw_match.match_id}")
        tracking: pd.DataFrame | None = None
        for horizon_seconds in args.horizon_seconds:
            label = horizon_label(horizon_seconds)
            cache_path = cache_dir / f"{raw_match.match_id}_{args.prefix}_{label}.pt"
            if args.resume and cache_path.exists():
                if _cache_covers_raw_periods(cache_path, raw_match):
                    print(f"using_cached: {cache_path}")
                    cached_by_horizon[float(horizon_seconds)].append(cache_path)
                    continue
            if tracking is None:
                tracking = SkillCornerAdapter(
                    raw_dir=Path(raw_match.match_dir),
                    match_id=raw_match.match_id,
                ).load_tracking()
            windows = build_tracking_windows(
                tracking,
                fps_out=args.fps_out,
                context_seconds=args.context_seconds,
                horizon_seconds=horizon_seconds,
                stride_seconds=args.stride_seconds,
            )
            if len(windows.match_id) == 0:
                print(
                    f"warning: no windows for match={raw_match.match_id} "
                    f"horizon_seconds={horizon_seconds}"
                )
                continue
            save_windows_pt(windows, cache_path)
            print(f"cached_match_horizon: {cache_path} windows={len(windows.match_id)}")
            cached_by_horizon[float(horizon_seconds)].append(cache_path)
    for horizon_seconds, paths in cached_by_horizon.items():
        if not paths:
            raise RuntimeError(f"No windows were produced for horizon_seconds={horizon_seconds}.")
        if args.skip_combine:
            print(
                "skipped_combined_horizon: "
                f"{horizon_label(horizon_seconds)} cache_files={len(paths)}"
            )
            continue
        combined = _concat_windows([load_windows_pt(path) for path in paths])
        out = args.out_dir / f"{args.prefix}_{horizon_label(horizon_seconds)}.pt"
        _write_horizon(combined, out)
    return match_ids


def main() -> None:
    args = parse_args()
    if args.source == "skillcorner" and not args.combined_load:
        match_ids = _prepare_skillcorner_per_match(args)
    else:
        match_ids = _prepare_combined(args)
    if args.source == "skillcorner" and len(match_ids) < 3:
        print(
            "warning: fewer than 3 SkillCorner matches were found; downstream real split "
            "evaluation should be treated as smoke-only unless more matches are added."
        )


if __name__ == "__main__":
    main()
