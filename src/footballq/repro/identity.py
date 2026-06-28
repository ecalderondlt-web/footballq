"""Period-aware sample identity helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any


def make_sample_id(match_id: object, period: object, frame_t: object) -> str:
    """Return the canonical period-aware sample identifier."""

    return f"{str(match_id)}:{int(period)}:{int(frame_t)}"


def sample_ids_from_components(
    match_ids: list[Any],
    periods: list[Any],
    frame_ts: list[Any],
) -> list[str]:
    """Build sample IDs from aligned component lists."""

    if not (len(match_ids) == len(periods) == len(frame_ts)):
        raise ValueError(
            "match_ids, periods, and frame_ts must have equal lengths to build sample IDs."
        )
    return [
        make_sample_id(match_id, period, frame_t)
        for match_id, period, frame_t in zip(match_ids, periods, frame_ts, strict=True)
    ]


def ensure_unique_sample_ids(sample_ids: list[str], context: str = "samples") -> None:
    """Fail when duplicate period-aware IDs are present."""

    counts = Counter(str(value) for value in sample_ids)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        suffix = "" if len(duplicates) <= 5 else f", ... (+{len(duplicates) - 5} more)"
        raise ValueError(f"Duplicate sample_id values in {context}: {preview}{suffix}")


def payload_periods(payload: dict[str, Any], n: int, default: int | None = 1) -> list[int]:
    """Return period values from a payload, defaulting only for legacy artifacts."""

    if "period" in payload:
        values = payload["period"]
    elif "periods" in payload:
        values = payload["periods"]
    elif default is not None:
        return [int(default)] * n
    else:
        raise ValueError("Payload is missing period values required for scientific alignment.")
    if len(values) != n:
        raise ValueError(f"Expected {n} period values, got {len(values)}.")
    return [int(value) for value in values]


def payload_sample_ids(
    payload: dict[str, Any],
    match_ids: list[str],
    periods: list[int],
    frame_ts: list[int],
) -> list[str]:
    """Return payload sample IDs, deriving them for old artifacts when needed."""

    if "sample_id" in payload:
        values = [str(value) for value in payload["sample_id"]]
        if len(values) != len(match_ids):
            raise ValueError(f"Expected {len(match_ids)} sample_id values, got {len(values)}.")
        return values
    return sample_ids_from_components(match_ids, periods, frame_ts)
