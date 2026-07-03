"""Availability reports for local SkillCorner Open Data artifacts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from footballq.data.windows import load_windows_pt
from footballq.decoding.dataset import load_decoder_dataset
from footballq.io.skillcorner import SkillCornerAdapter


@dataclass(frozen=True)
class SkillCornerRawMatch:
    match_id: str
    match_dir: str
    tracking_files: list[str]
    metadata_files: list[str]
    event_files: list[str]
    raw_frame_count_by_period: dict[str, int] = field(default_factory=dict)

    @property
    def has_tracking(self) -> bool:
        return bool(self.tracking_files)

    @property
    def has_metadata(self) -> bool:
        return bool(self.metadata_files)

    @property
    def has_events(self) -> bool:
        return bool(self.event_files)

    @property
    def raw_periods(self) -> list[int]:
        return _integer_period_keys(self.raw_frame_count_by_period)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "match_dir": self.match_dir,
            "tracking_files": self.tracking_files,
            "metadata_files": self.metadata_files,
            "event_files": self.event_files,
            "has_tracking": self.has_tracking,
            "has_metadata": self.has_metadata,
            "has_events": self.has_events,
            "raw_periods": self.raw_periods,
            "raw_frame_count_by_period": self.raw_frame_count_by_period,
        }


def horizon_label(seconds: float) -> str:
    value = int(seconds) if float(seconds).is_integer() else seconds
    return f"h{value}s".replace(".", "p")


def _period_key(value: object) -> str:
    if value is None:
        return "missing"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return "missing"
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def _period_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _sorted_period_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter, key=_period_sort_key)}


def _integer_period_keys(counts: dict[str, int]) -> list[int]:
    periods: set[int] = set()
    for value in counts:
        try:
            periods.add(int(value))
        except ValueError:
            continue
    return sorted(periods)


def _raw_frame_period_counts(files: list[Path]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in files:
        try:
            for frame in SkillCornerAdapter._iter_json_records(path):
                if not isinstance(frame, dict):
                    counts["unsupported_record"] += 1
                    continue
                period = SkillCornerAdapter._first(frame, ["period", "period_id"])
                counts[_period_key(period)] += 1
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            counts["unreadable"] += 1
    return _sorted_period_counts(counts)


def discover_skillcorner_raw_matches(raw_dir: str | Path) -> list[SkillCornerRawMatch]:
    """Return one report row per local match folder or tracking file."""

    root = Path(raw_dir)
    if not root.exists():
        return []
    tracking_files = sorted(
        path
        for path in [*root.rglob("*.jsonl"), *root.rglob("*.json")]
        if "tracking" in path.name.lower()
    )
    by_dir: dict[Path, list[Path]] = {}
    for path in tracking_files:
        by_dir.setdefault(path.parent, []).append(path)
    matches: list[SkillCornerRawMatch] = []
    for match_dir, files in sorted(by_dir.items(), key=lambda item: item[0].name):
        match_id = match_dir.name if match_dir != root else files[0].stem
        metadata = sorted(
            path
            for path in match_dir.glob("*.json")
            if "match" in path.name.lower() and "tracking" not in path.name.lower()
        )
        events = sorted(
            path
            for path in [*match_dir.glob("*event*.json"), *match_dir.glob("*event*.jsonl")]
            if "tracking" not in path.name.lower()
        )
        matches.append(
            SkillCornerRawMatch(
                match_id=str(match_id),
                match_dir=str(match_dir),
                tracking_files=[str(path) for path in files],
                metadata_files=[str(path) for path in metadata],
                event_files=[str(path) for path in events],
                raw_frame_count_by_period=_raw_frame_period_counts(files),
            )
        )
    return matches


def _counter(values: list[str]) -> dict[str, int]:
    return _sorted_period_counts(Counter(str(value) for value in values))


def _match_period_counts(match_ids: list[str], periods: list[int]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for match_id, period in zip(match_ids, periods, strict=True):
        counts.setdefault(str(match_id), Counter())[str(int(period))] += 1
    return {match_id: dict(sorted(counter.items())) for match_id, counter in sorted(counts.items())}


def _match_period_start_frame_ranges(
    match_ids: list[str],
    periods: list[int],
    start_frames: list[int],
) -> dict[str, dict[str, dict[str, int]]]:
    ranges: dict[str, dict[str, dict[str, int]]] = {}
    for match_id, period, start_frame in zip(match_ids, periods, start_frames, strict=True):
        match_key = str(match_id)
        period_key = str(int(period))
        period_ranges = ranges.setdefault(match_key, {})
        item = period_ranges.setdefault(
            period_key,
            {
                "count": 0,
                "min_start_frame": int(start_frame),
                "max_start_frame": int(start_frame),
            },
        )
        item["count"] += 1
        item["min_start_frame"] = min(item["min_start_frame"], int(start_frame))
        item["max_start_frame"] = max(item["max_start_frame"], int(start_frame))
    return {match_id: dict(sorted(periods.items())) for match_id, periods in sorted(ranges.items())}


def _int_keys(values: dict[str, Any]) -> set[int]:
    keys: set[int] = set()
    for value in values:
        try:
            keys.add(int(value))
        except ValueError:
            continue
    return keys


def _load_embedding_keys(path: Path | None) -> tuple[set[tuple[str, int]], set[str]]:
    if path is None or not path.exists():
        return set(), set()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    match_ids = [str(value) for value in payload.get("match_id", [])]
    frame_ts = [int(value) for value in payload.get("frame_t", [])]
    return set(zip(match_ids, frame_ts, strict=True)), set(match_ids)


def _processed_horizon_report(
    *,
    processed_dir: Path,
    prefix: str,
    decoder_prefix: str,
    horizon_seconds: float,
    embedding_keys: set[tuple[str, int]],
    embedding_matches: set[str],
) -> dict[str, Any]:
    label = horizon_label(horizon_seconds)
    windows_path = processed_dir / f"{prefix}_{label}.pt"
    decoder_path = processed_dir / f"{decoder_prefix}_{label}.pt"
    report: dict[str, Any] = {
        "horizon_seconds": float(horizon_seconds),
        "horizon_label": label,
        "windows_path": str(windows_path),
        "windows_exists": windows_path.exists(),
        "decoder_dataset_path": str(decoder_path),
        "decoder_dataset_exists": decoder_path.exists(),
        "window_count": 0,
        "window_count_by_match": {},
        "window_periods": [],
        "window_count_by_period": {},
        "window_count_by_match_period": {},
        "window_start_frame_range_by_match_period": {},
        "missing_processed_periods": [],
        "missing_processed_periods_by_match": {},
        "decoder_example_count": 0,
        "decoder_example_count_by_match": {},
        "embedding_alignment": {
            "checked": bool(embedding_keys),
            "missing_window_matches_in_embeddings": [],
            "matching_window_keys": 0,
            "window_keys": 0,
        },
    }
    if windows_path.exists():
        windows = load_windows_pt(windows_path)
        match_ids = [str(value) for value in windows.match_id]
        periods = [int(value) for value in windows.period]
        start_frames = [int(value) for value in windows.start_frame]
        report["window_count"] = len(windows.match_id)
        report["window_count_by_match"] = _counter(match_ids)
        report["window_periods"] = sorted(set(periods))
        report["window_count_by_period"] = _counter([str(value) for value in periods])
        report["window_count_by_match_period"] = _match_period_counts(match_ids, periods)
        report["window_start_frame_range_by_match_period"] = _match_period_start_frame_ranges(
            match_ids,
            periods,
            start_frames,
        )
        if embedding_keys:
            window_keys = set(
                zip(
                    match_ids,
                    start_frames,
                    strict=True,
                )
            )
            report["embedding_alignment"] = {
                "checked": True,
                "missing_window_matches_in_embeddings": sorted(
                    set(str(value) for value in windows.match_id) - embedding_matches
                ),
                "matching_window_keys": len(window_keys & embedding_keys),
                "window_keys": len(window_keys),
            }
    if decoder_path.exists():
        data = load_decoder_dataset(decoder_path)
        report["decoder_example_count"] = data.num_examples
        report["decoder_example_count_by_match"] = _counter(
            [str(value) for value in data.examples["match_id"]]
        )
    return report


def _add_raw_processed_period_gaps(
    horizon_reports: list[dict[str, Any]],
    raw_matches: list[SkillCornerRawMatch],
) -> None:
    raw_periods_by_match = {
        match.match_id: set(match.raw_periods)
        for match in raw_matches
        if match.raw_periods
    }
    for horizon in horizon_reports:
        missing_by_match: dict[str, list[int]] = {}
        processed_counts = horizon.get("window_count_by_match_period", {})
        for match_id, raw_periods in raw_periods_by_match.items():
            processed_periods = _int_keys(processed_counts.get(match_id, {}))
            missing = sorted(raw_periods - processed_periods)
            if missing:
                missing_by_match[match_id] = missing
        missing_all = sorted(
            {period for periods in missing_by_match.values() for period in periods}
        )
        horizon["missing_processed_periods"] = missing_all
        horizon["missing_processed_periods_by_match"] = missing_by_match


def build_skillcorner_availability_report(
    raw_dir: str | Path,
    processed_dir: str | Path = "data/processed",
    *,
    embeddings: str | Path | None = None,
    horizons: list[float] | None = None,
    windows_prefix: str = "skillcorner_windows",
    decoder_prefix: str = "skillcorner_decoder_dataset",
) -> dict[str, Any]:
    """Summarize raw matches, prepared windows, decoder datasets, and embedding coverage."""

    raw_matches = discover_skillcorner_raw_matches(raw_dir)
    embedding_path = Path(embeddings) if embeddings is not None else None
    embedding_keys, embedding_matches = _load_embedding_keys(embedding_path)
    horizon_reports = [
        _processed_horizon_report(
            processed_dir=Path(processed_dir),
            prefix=windows_prefix,
            decoder_prefix=decoder_prefix,
            horizon_seconds=horizon,
            embedding_keys=embedding_keys,
            embedding_matches=embedding_matches,
        )
        for horizon in (horizons or [2.0, 4.0, 6.0])
    ]
    _add_raw_processed_period_gaps(horizon_reports, raw_matches)
    raw_periods = sorted({period for match in raw_matches for period in match.raw_periods})
    return {
        "raw_dir": str(raw_dir),
        "raw_match_count": len(raw_matches),
        "raw_match_ids": [match.match_id for match in raw_matches],
        "raw_periods": raw_periods,
        "raw_frame_count_by_match_period": {
            match.match_id: match.raw_frame_count_by_period for match in raw_matches
        },
        "raw_matches": [match.to_dict() for match in raw_matches],
        "embeddings_path": str(embedding_path) if embedding_path else "",
        "embedding_match_count": len(embedding_matches),
        "embedding_match_ids": sorted(embedding_matches),
        "horizons": horizon_reports,
    }
