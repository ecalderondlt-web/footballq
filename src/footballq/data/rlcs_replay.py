"""Replay parsing, quality control, and chronology-safe RLCS identity resolution."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PARSER_PACKAGE = "analyzerl-parser"
PARSER_VERSION = "1.0.5"
PLAYER_PREFIX_PATTERN = re.compile(r"^(blue|orange)_player_(\d+)_id$")
APPROVED_ALIAS_STATUSES = frozenset({"approved", "frozen"})
GOAL_DUPLICATE_MAX_FRAME_GAP = 45


class ReplayParseError(RuntimeError):
    """Raised when a replay cannot enter the scientific dataset."""


class IdentityResolutionError(ValueError):
    """Raised when a replay contains an unresolved identity collision."""


@dataclass(frozen=True)
class PlayerSlot:
    """One parser roster slot."""

    prefix: str
    team: str
    team_index: int


@dataclass(frozen=True)
class IdentityObservation:
    """One player identifier observed before identity fitting."""

    replay_id: str
    split: str
    event_time_utc: str | None
    group_id: str | None
    prefix: str
    team: str
    handle: str
    normalized_handle: str
    platform: str
    platform_id: str

    @property
    def platform_key(self) -> str:
        return f"{self.platform}:{self.platform_id}"


@dataclass(frozen=True)
class AliasEntry:
    """One reviewed row in the frozen identity alias registry."""

    canonical_player_id: str
    canonical_handle: str
    observed_handle: str
    platform: str
    platform_id: str
    valid_from: str | None
    valid_to: str | None
    evidence_group_id: str | None
    resolution_method: str
    review_status: str


@dataclass(frozen=True)
class IdentityResolution:
    """Resolution result without access to targets or future matches."""

    canonical_player_id: str | None
    method: str
    known: bool


@dataclass(frozen=True)
class ReplayQC:
    """Machine-readable replay-level quality decision."""

    accepted: bool
    reasons: tuple[str, ...]
    team_size: int | None
    player_slots: int
    duration_seconds: float | None
    frame_count: int
    event_count: int


@dataclass
class ParsedReplay:
    """Parser tables plus a deterministic QC report."""

    replay_id: str
    frames: pd.DataFrame
    events: pd.DataFrame
    qc: ReplayQC


def split_for_inventory_row(row: Mapping[str, Any]) -> str:
    """Apply the frozen chronological Split 1/Split 2 regional rule."""

    split_number = int(row.get("split_number") or 0)
    regional_number = int(row.get("regional_number") or 0)
    region = str(row.get("region") or "").upper()
    if region not in {"EU", "NA"}:
        raise ValueError(f"Replay {row.get('replay_id')} has unsupported region {region!r}.")
    if split_number == 1:
        return "train"
    if split_number == 2 and regional_number == 1:
        return "val"
    if split_number == 2 and regional_number in {2, 3}:
        return "test"
    raise ValueError(
        f"Replay {row.get('replay_id')} is outside the frozen split selectors: "
        f"split={split_number}, regional={regional_number}."
    )


def freeze_chronological_split_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    template_path: str | Path,
    output_path: str | Path | None = None,
    inventory_sha256: str | None = None,
) -> Path:
    """Replace selector placeholders with deduplicated replay IDs atomically."""

    template = Path(template_path)
    payload = json.loads(template.read_text(encoding="utf-8"))
    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    seen: set[str] = set()
    for row in sorted(
        records,
        key=lambda value: (
            str(value.get("event_time_utc") or ""),
            str(value.get("replay_id") or ""),
        ),
    ):
        replay_id = str(row.get("replay_id") or "")
        if not replay_id or replay_id in seen:
            continue
        seen.add(replay_id)
        by_split[split_for_inventory_row(row)].append(replay_id)
    if any(not values for values in by_split.values()):
        raise ValueError("Frozen RLCS split requires non-empty train, val, and test replay sets.")
    payload.update(
        {
            "status": "frozen",
            "split_unit": "replay_id",
            "protocol": "chronological_train_split1_val_s2r1_test_s2r2r3",
            "train_match_ids": by_split["train"],
            "val_match_ids": by_split["val"],
            "test_match_ids": by_split["test"],
            "all_match_ids": [*by_split["train"], *by_split["val"], *by_split["test"]],
            "expected_count": sum(len(values) for values in by_split.values()),
            "inventory_sha256": inventory_sha256,
            "frozen_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
    )
    payload.pop("freeze_rule", None)
    destination = Path(output_path or template)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_frozen_rlcs_split(path: str | Path) -> dict[str, Any]:
    """Reject selector templates at every scientific training boundary."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("dataset") != "rlcs":
        raise ValueError("Expected an RLCS split manifest.")
    if payload.get("status") != "frozen" or payload.get("split_unit") != "replay_id":
        raise ValueError("RLCS split is not frozen; selector IDs cannot be used for training.")
    groups = [
        [str(value) for value in payload[f"{split}_match_ids"]]
        for split in ("train", "val", "test")
    ]
    flattened = [value for group in groups for value in group]
    if any(value.startswith("selector:") for value in flattened):
        raise ValueError("Frozen RLCS split still contains selector placeholders.")
    if len(flattened) != len(set(flattened)):
        raise ValueError("Frozen RLCS split contains duplicate replay IDs.")
    if set(flattened) != set(str(value) for value in payload.get("all_match_ids", [])):
        raise ValueError("Frozen RLCS all_match_ids does not match train/val/test union.")
    return payload


def normalize_handle(value: Any) -> str:
    """Normalize a handle exactly; fuzzy matching is deliberately forbidden."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return " ".join(normalized.split())


def normalize_platform(value: Any) -> str:
    """Map parser platform labels to stable short names."""

    raw = normalize_handle(value)
    raw = raw.removeprefix("onlineplatform_")
    aliases = {
        "ps4": "playstation",
        "psn": "playstation",
        "playstationnetwork": "playstation",
        "xboxone": "xbox",
        "xboxlive": "xbox",
    }
    return aliases.get(raw, raw)


def player_slots(columns: Iterable[str]) -> list[PlayerSlot]:
    """Discover parser player columns in a deterministic team/slot order."""

    slots: list[PlayerSlot] = []
    for column in columns:
        match = PLAYER_PREFIX_PATTERN.match(str(column))
        if match:
            slots.append(
                PlayerSlot(
                    prefix=str(column)[: -len("_id")],
                    team=match.group(1),
                    team_index=int(match.group(2)),
                )
            )
    return sorted(slots, key=lambda slot: (slot.team != "blue", slot.team_index))


def _first_present(frame: pd.DataFrame, column: str) -> Any:
    if column not in frame:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    for value in values:
        if str(value).strip():
            return value
    return None


def _platform_parts(frame: pd.DataFrame, slot: PlayerSlot) -> tuple[str, str]:
    network_id = str(_first_present(frame, f"{slot.prefix}_network_id") or "").strip()
    platform = normalize_platform(_first_present(frame, f"{slot.prefix}_platform"))
    platform_id = str(_first_present(frame, f"{slot.prefix}_id") or "").strip()
    if ":" in network_id:
        network_platform, network_value = network_id.split(":", 1)
        platform = normalize_platform(network_platform) or platform
        platform_id = network_value.strip() or platform_id
    return platform, platform_id


def roster_observations(
    frames: pd.DataFrame,
    *,
    replay_id: str,
    split: str,
    event_time_utc: str | None,
    group_id: str | None,
) -> list[IdentityObservation]:
    """Extract roster identifiers without reading any outcome column."""

    observations: list[IdentityObservation] = []
    for slot in player_slots(frames.columns):
        handle = str(_first_present(frames, f"{slot.prefix}_name") or "").strip()
        platform, platform_id = _platform_parts(frames, slot)
        observations.append(
            IdentityObservation(
                replay_id=str(replay_id),
                split=str(split),
                event_time_utc=event_time_utc,
                group_id=group_id,
                prefix=slot.prefix,
                team=slot.team,
                handle=handle,
                normalized_handle=normalize_handle(handle),
                platform=platform,
                platform_id=platform_id,
            )
        )
    return observations


def load_alias_registry(path: str | Path) -> list[AliasEntry]:
    """Load reviewed aliases; unreviewed rows never affect resolution."""

    entries: list[AliasEntry] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            status = normalize_handle(row.get("review_status"))
            if status not in APPROVED_ALIAS_STATUSES:
                continue
            entries.append(
                AliasEntry(
                    canonical_player_id=str(row.get("canonical_player_id") or "").strip(),
                    canonical_handle=normalize_handle(row.get("canonical_handle")),
                    observed_handle=normalize_handle(row.get("observed_handle")),
                    platform=normalize_platform(row.get("platform")),
                    platform_id=str(row.get("platform_id") or "").strip(),
                    valid_from=str(row.get("valid_from") or "").strip() or None,
                    valid_to=str(row.get("valid_to") or "").strip() or None,
                    evidence_group_id=(
                        str(row.get("evidence_group_id") or "").strip() or None
                    ),
                    resolution_method=str(row.get("resolution_method") or "").strip(),
                    review_status=status,
                )
            )
    return entries


def _date_part(value: str | None) -> str | None:
    if not value:
        return None
    return str(value)[:10]


def _date_applies(entry: AliasEntry, event_time_utc: str | None) -> bool:
    date = _date_part(event_time_utc)
    if date is None:
        return not entry.valid_from and not entry.valid_to
    return (entry.valid_from is None or date >= entry.valid_from[:10]) and (
        entry.valid_to is None or date <= entry.valid_to[:10]
    )


class IdentityResolver:
    """Frozen resolver fitted exclusively on training-roster observations."""

    def __init__(
        self,
        *,
        stable_platform_keys: set[str],
        colliding_platform_keys: set[str],
        ambiguous_handles: set[str],
        aliases: Sequence[AliasEntry],
    ) -> None:
        self.stable_platform_keys = frozenset(stable_platform_keys)
        self.colliding_platform_keys = frozenset(colliding_platform_keys)
        self.ambiguous_handles = frozenset(ambiguous_handles)
        self.aliases = tuple(aliases)

    @classmethod
    def fit_training(
        cls,
        observations: Sequence[IdentityObservation],
        aliases: Sequence[AliasEntry] = (),
    ) -> IdentityResolver:
        """Fit maps only from records explicitly labelled ``train``."""

        non_train = sorted({item.split for item in observations if item.split != "train"})
        if non_train:
            raise ValueError(
                "Identity fitting accepts training observations only; received: "
                + ", ".join(non_train)
            )
        handles_by_key: dict[str, set[str]] = defaultdict(set)
        keys_by_handle: dict[str, set[str]] = defaultdict(set)
        for item in observations:
            if not item.platform or not item.platform_id or not item.normalized_handle:
                continue
            handles_by_key[item.platform_key].add(item.normalized_handle)
            keys_by_handle[item.normalized_handle].add(item.platform_key)
        stable = {key for key, handles in handles_by_key.items() if len(handles) == 1}
        colliding = {key for key, handles in handles_by_key.items() if len(handles) > 1}
        ambiguous_handles = {
            handle for handle, keys in keys_by_handle.items() if len(keys) > 1
        }
        return cls(
            stable_platform_keys=stable,
            colliding_platform_keys=colliding,
            ambiguous_handles=ambiguous_handles,
            aliases=aliases,
        )

    def resolve(self, observation: IdentityObservation) -> IdentityResolution:
        """Resolve from frozen identifiers only; outcomes are not accepted as inputs."""

        matches = [
            entry
            for entry in self.aliases
            if entry.observed_handle == observation.normalized_handle
            and (not entry.platform or entry.platform == observation.platform)
            and (not entry.platform_id or entry.platform_id == observation.platform_id)
            and _date_applies(entry, observation.event_time_utc)
        ]
        canonical = {entry.canonical_player_id for entry in matches if entry.canonical_player_id}
        if len(canonical) == 1:
            return IdentityResolution(next(iter(canonical)), "frozen_alias", True)
        if len(canonical) > 1:
            return IdentityResolution(None, "ambiguous_frozen_alias", False)
        key = observation.platform_key
        if (
            observation.platform
            and observation.platform_id
            and key not in self.colliding_platform_keys
            and observation.normalized_handle not in self.ambiguous_handles
        ):
            method = "stable_platform_id" if key in self.stable_platform_keys else "new_platform_id"
            return IdentityResolution(key, method, True)
        if key in self.colliding_platform_keys:
            return IdentityResolution(None, "generic_platform_id_requires_alias", False)
        if observation.normalized_handle in self.ambiguous_handles:
            return IdentityResolution(None, "many_to_one_handle_collision", False)
        return IdentityResolution(None, "unknown_or_incomplete_identity", False)

    def with_collision_audit(
        self, observations: Sequence[IdentityObservation]
    ) -> IdentityResolver:
        """Block cross-corpus identifier collisions without using events or targets to map them."""

        handles_by_key: dict[str, set[str]] = defaultdict(set)
        keys_by_handle: dict[str, set[str]] = defaultdict(set)
        for item in observations:
            if not item.platform or not item.platform_id or not item.normalized_handle:
                continue
            handles_by_key[item.platform_key].add(item.normalized_handle)
            keys_by_handle[item.normalized_handle].add(item.platform_key)
        colliding = {
            key for key, handles in handles_by_key.items() if len(handles) > 1
        }
        ambiguous = {handle for handle, keys in keys_by_handle.items() if len(keys) > 1}
        return IdentityResolver(
            stable_platform_keys=set(self.stable_platform_keys),
            colliding_platform_keys=set(self.colliding_platform_keys) | colliding,
            ambiguous_handles=set(self.ambiguous_handles) | ambiguous,
            aliases=self.aliases,
        )

    def resolve_roster(
        self, observations: Sequence[IdentityObservation]
    ) -> dict[str, str]:
        """Resolve all six players or exclude the replay as a unit."""

        resolved: dict[str, str] = {}
        failures: list[str] = []
        for item in observations:
            result = self.resolve(item)
            if not result.known or result.canonical_player_id is None:
                failures.append(f"{item.prefix}:{item.handle}:{result.method}")
            else:
                resolved[item.prefix] = result.canonical_player_id
        if failures:
            raise IdentityResolutionError("Unresolved roster identities: " + "; ".join(failures))
        if len(set(resolved.values())) != len(resolved):
            raise IdentityResolutionError("Canonical identity collision within replay roster.")
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_platform_keys": sorted(self.stable_platform_keys),
            "colliding_platform_keys": sorted(self.colliding_platform_keys),
            "ambiguous_handles": sorted(self.ambiguous_handles),
            "aliases": [asdict(value) for value in self.aliases],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IdentityResolver:
        return cls(
            stable_platform_keys=set(payload.get("stable_platform_keys", [])),
            colliding_platform_keys=set(payload.get("colliding_platform_keys", [])),
            ambiguous_handles=set(payload.get("ambiguous_handles", [])),
            aliases=[AliasEntry(**row) for row in payload.get("aliases", [])],
        )


def add_precise_frame_time(frames: pd.DataFrame) -> pd.DataFrame:
    """Interpolate integer parser clock buckets onto exact observed frame indices."""

    required = {"observed_frame_number", "seconds_elapsed"}
    if not required.issubset(frames.columns):
        raise ReplayParseError("Frame table lacks observed_frame_number or seconds_elapsed.")
    ordered = frames.sort_values("observed_frame_number", kind="stable").reset_index(drop=True)
    frame_number = ordered["observed_frame_number"].to_numpy(dtype=np.float64)
    elapsed = ordered["seconds_elapsed"].to_numpy(dtype=np.float64)
    finite = np.isfinite(frame_number) & np.isfinite(elapsed)
    if finite.sum() < 2 or np.any(np.diff(frame_number[finite]) < 0):
        raise ReplayParseError("Replay frame clock is not coherent.")
    first_by_second = (
        ordered.loc[finite, ["observed_frame_number", "seconds_elapsed"]]
        .groupby("seconds_elapsed", sort=True, as_index=False)
        .first()
    )
    anchors_x = first_by_second["observed_frame_number"].to_numpy(dtype=np.float64)
    anchors_y = first_by_second["seconds_elapsed"].to_numpy(dtype=np.float64)
    if len(anchors_x) < 2 or np.any(np.diff(anchors_x) <= 0) or np.any(np.diff(anchors_y) < 0):
        raise ReplayParseError("Replay clock cannot be interpolated monotonically.")
    precise = np.interp(frame_number, anchors_x, anchors_y)
    final_rate = np.median(np.diff(anchors_x)[-min(30, len(anchors_x) - 1) :])
    after = frame_number > anchors_x[-1]
    if np.isfinite(final_rate) and final_rate > 0:
        precise[after] = anchors_y[-1] + (frame_number[after] - anchors_x[-1]) / final_rate
    ordered["game_time_s_precise"] = precise.astype(np.float32)
    return ordered


def align_event_times(events: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Attach precise frame-derived event times without interpolating from future events."""

    if "game_time_s_precise" not in frames:
        frames = add_precise_frame_time(frames)
    mapping = frames.drop_duplicates("observed_frame_number").set_index(
        "observed_frame_number"
    )["game_time_s_precise"]
    aligned = events.copy()
    aligned["game_time_s_precise"] = aligned["observed_frame_number"].map(mapping)
    if aligned["game_time_s_precise"].isna().any():
        lookup_x = mapping.index.to_numpy(dtype=np.float64)
        lookup_y = mapping.to_numpy(dtype=np.float64)
        missing = aligned["game_time_s_precise"].isna()
        query = aligned.loc[missing, "observed_frame_number"].to_numpy(dtype=np.float64)
        aligned.loc[missing, "game_time_s_precise"] = np.interp(query, lookup_x, lookup_y)
    return aligned


def _clustered_goal_events(events: pd.DataFrame) -> list[tuple[int, str]]:
    """Return one chronological scoring event per parser goal cluster."""

    event_types = events["event_type"].astype(str).str.casefold()
    goals = events.loc[event_types.eq("goal")]
    clusters: list[dict[str, Any]] = []
    for row in goals.to_dict(orient="records"):
        raw_frame = pd.to_numeric(row.get("observed_frame_number"), errors="coerce")
        if pd.isna(raw_frame):
            raise ReplayParseError("Goal event lacks an observed frame number.")
        frame = int(raw_frame)
        team = normalize_handle(row.get("event_team") or row.get("event_player_1_team"))
        if team not in {"blue", "orange"}:
            raise ReplayParseError("Goal event lacks a valid scoring team.")
        raw_number = pd.to_numeric(row.get("goal_number"), errors="coerce")
        number = None if pd.isna(raw_number) else int(raw_number)
        if clusters:
            previous = clusters[-1]
            same_team = team == previous["team"]
            close = frame - int(previous["last_frame"]) <= GOAL_DUPLICATE_MAX_FRAME_GAP
            compatible_number = (
                frame == int(previous["last_frame"])
                or number is None
                or not previous["numbers"]
                or number in previous["numbers"]
            )
            if same_team and close and compatible_number:
                previous["rows"].append((frame, number))
                previous["last_frame"] = frame
                if number is not None:
                    previous["numbers"].add(number)
                continue
        clusters.append(
            {
                "team": team,
                "last_frame": frame,
                "numbers": set() if number is None else {number},
                "rows": [(frame, number)],
            }
        )
    scoring_events: list[tuple[int, str]] = []
    for cluster in clusters:
        numbered = [item for item in cluster["rows"] if item[1] is not None]
        representative = max(numbered or cluster["rows"], key=lambda item: item[0])
        scoring_events.append((int(representative[0]), str(cluster["team"])))
    return scoring_events


def repair_score_columns(
    events: pd.DataFrame,
    *,
    expected_blue_score: int | None = None,
    expected_orange_score: int | None = None,
) -> pd.DataFrame:
    """Rebuild the scoreboard from deduplicated goal events.

    AnalyzerL 1.0.5 can increment exported score columns more than once when a
    classified goal and a synthesized official-goal row refer to the same
    scoring play. Goal rows are therefore clustered before cumulative scores
    are reconstructed. When Ballchasing final scores are available, a replay
    fails closed unless the reconstructed final score agrees.
    """

    required = {
        "event_type",
        "observed_frame_number",
        "blue_score",
        "orange_score",
    }
    if not required.issubset(events.columns):
        raise ReplayParseError("Event table lacks goal, score, or frame columns.")
    order_columns = ["observed_frame_number"]
    if "event_number" in events:
        order_columns.append("event_number")
    repaired = events.sort_values(order_columns, kind="stable").reset_index(drop=True)
    raw = repaired[["blue_score", "orange_score"]].apply(pd.to_numeric, errors="coerce")
    if raw.dropna().empty or bool((raw.dropna() < 0).any().any()):
        raise ReplayParseError("Event table contains invalid raw score values.")
    scoring_events = _clustered_goal_events(repaired)
    observed_blue = sum(team == "blue" for _, team in scoring_events)
    observed_orange = sum(team == "orange" for _, team in scoring_events)
    if max(observed_blue, observed_orange) > 30:
        raise ReplayParseError("Reconstructed score exceeds the plausibility bound.")
    if expected_blue_score is not None or expected_orange_score is not None:
        expected = (
            int(expected_blue_score or 0),
            int(expected_orange_score or 0),
        )
        if (observed_blue, observed_orange) != expected:
            raise ReplayParseError(
                "Reconstructed final score does not match the Ballchasing inventory."
            )
    frames = pd.to_numeric(repaired["observed_frame_number"], errors="coerce").to_numpy()
    if not np.isfinite(frames).all():
        raise ReplayParseError("Event table contains invalid observed frame numbers.")
    blue = np.zeros(len(repaired), dtype=np.int16)
    orange = np.zeros(len(repaired), dtype=np.int16)
    for frame, team in scoring_events:
        target = blue if team == "blue" else orange
        target[frames >= frame] += 1
    repaired["blue_score"] = blue
    repaired["orange_score"] = orange
    return repaired


def _score_is_coherent(
    events: pd.DataFrame,
    *,
    expected_blue_score: int | None = None,
    expected_orange_score: int | None = None,
) -> bool:
    if not {"blue_score", "orange_score"}.issubset(events.columns):
        return False
    raw = events[["blue_score", "orange_score"]].apply(pd.to_numeric, errors="coerce")
    if raw.dropna().empty or bool((raw.dropna() < 0).any().any()):
        return False
    try:
        scores = repair_score_columns(
            events,
            expected_blue_score=expected_blue_score,
            expected_orange_score=expected_orange_score,
        )
    except ReplayParseError:
        return False
    final_blue = int(scores["blue_score"].iloc[-1])
    final_orange = int(scores["orange_score"].iloc[-1])
    if max(final_blue, final_orange) > 30:
        return False
    if expected_blue_score is not None and final_blue != int(expected_blue_score):
        return False
    if expected_orange_score is not None and final_orange != int(expected_orange_score):
        return False
    return True


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").dropna()


def _standard_soccar_map(map_name: str | None) -> bool:
    if not map_name:
        return True
    excluded = ("hoops", "dropshot", "throwback", "labs", "snowday", "heatseeker")
    lowered = normalize_handle(map_name)
    return not any(value in lowered for value in excluded)


def quality_control_replay(
    frames: pd.DataFrame,
    events: pd.DataFrame,
    *,
    map_name: str | None = None,
    minimum_duration_seconds: float = 180.0,
    maximum_duration_seconds: float = 1800.0,
    require_standard_3v3: bool = True,
    expected_blue_score: int | None = None,
    expected_orange_score: int | None = None,
) -> ReplayQC:
    """Apply replay-level gates before identity or target construction."""

    reasons: list[str] = []
    slots = player_slots(frames.columns)
    team_sizes = _numeric_column(frames, "team_size")
    team_size = int(team_sizes.mode().iloc[0]) if not team_sizes.empty else None
    duration_values = _numeric_column(frames, "seconds_elapsed")
    duration = float(duration_values.max()) if not duration_values.empty else None
    if require_standard_3v3 and (team_size != 3 or len(slots) != 6):
        reasons.append("not_standard_3v3")
    if sum(slot.team == "blue" for slot in slots) != 3:
        reasons.append("blue_roster_not_three")
    if sum(slot.team == "orange" for slot in slots) != 3:
        reasons.append("orange_roster_not_three")
    if duration is None or not minimum_duration_seconds <= duration <= maximum_duration_seconds:
        reasons.append("duration_out_of_range")
    required_ball = {
        "ball_pos_x",
        "ball_pos_y",
        "ball_pos_z",
        "ball_vel_x",
        "ball_vel_y",
        "ball_vel_z",
    }
    if not required_ball.issubset(frames.columns):
        reasons.append("missing_ball_rigid_body")
    elif frames[list(required_ball)].notna().mean().min() < 0.95:
        reasons.append("truncated_ball_state")
    required_car_suffixes = ("pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z")
    for slot in slots:
        columns = [f"{slot.prefix}_{suffix}" for suffix in required_car_suffixes]
        if not set(columns).issubset(frames.columns) or frames[columns].notna().mean().min() < 0.90:
            reasons.append(f"missing_or_truncated_car_state:{slot.prefix}")
        if bool(_first_present(frames, f"{slot.prefix}_is_bot")):
            reasons.append(f"bot_player:{slot.prefix}")
    observed = _numeric_column(frames, "observed_frame_number")
    if observed.empty or not observed.is_monotonic_increasing or observed.duplicated().any():
        reasons.append("frame_sequence_incoherent")
    if not _score_is_coherent(
        events,
        expected_blue_score=expected_blue_score,
        expected_orange_score=expected_orange_score,
    ):
        reasons.append("score_incoherent")
    if not _standard_soccar_map(map_name):
        reasons.append("non_soccar_map")
    return ReplayQC(
        accepted=not reasons,
        reasons=tuple(sorted(set(reasons))),
        team_size=team_size,
        player_slots=len(slots),
        duration_seconds=duration,
        frame_count=int(len(frames)),
        event_count=int(len(events)),
    )


def parse_replay_file(
    path: str | Path,
    *,
    replay_id: str | None = None,
    workers: int = 1,
    map_name: str | None = None,
    minimum_duration_seconds: float = 180.0,
    maximum_duration_seconds: float = 1800.0,
    require_standard_3v3: bool = True,
    expected_blue_score: int | None = None,
    expected_orange_score: int | None = None,
) -> ParsedReplay:
    """Parse a native replay into frame and event tables with frozen parser semantics."""

    try:
        import analyzerl_parser
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "RLCS parsing requires analyzerl-parser==1.0.5; install footballq[rlcs]."
        ) from exc
    installed = str(getattr(analyzerl_parser, "__version__", "unknown"))
    if installed != PARSER_VERSION:
        raise RuntimeError(
            f"Expected {PARSER_PACKAGE}=={PARSER_VERSION}, found {installed}."
        )
    replay_path = Path(path)
    try:
        frames = analyzerl_parser.parse_replay(
            replay_path,
            workers=int(workers),
            return_type="pandas",
            output="frames-only",
            rotation_events=False,
        )
        events = analyzerl_parser.parse_replay(
            replay_path,
            workers=int(workers),
            return_type="pandas",
            output="pbp",
            rotation_events=False,
        )
    except Exception as exc:
        raise ReplayParseError(f"Parser failed for {replay_path.name}: {exc}") from exc
    if not isinstance(frames, pd.DataFrame) or not isinstance(events, pd.DataFrame):
        raise ReplayParseError("Parser did not return pandas DataFrames.")
    frames = add_precise_frame_time(frames)
    events = align_event_times(events, frames)
    qc = quality_control_replay(
        frames,
        events,
        map_name=map_name,
        minimum_duration_seconds=minimum_duration_seconds,
        maximum_duration_seconds=maximum_duration_seconds,
        require_standard_3v3=require_standard_3v3,
        expected_blue_score=expected_blue_score,
        expected_orange_score=expected_orange_score,
    )
    return ParsedReplay(
        replay_id=str(replay_id or replay_path.stem),
        frames=frames,
        events=events,
        qc=qc,
    )


def cache_parsed_replay(parsed: ParsedReplay, cache_dir: str | Path) -> Path:
    """Cache parser output atomically so corpus construction parses each file once."""

    destination = Path(cache_dir) / parsed.replay_id
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in (("frames", parsed.frames), ("events", parsed.events)):
        final = destination / f"{name}.parquet"
        temporary = final.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(final)
    metadata = {
        "replay_id": parsed.replay_id,
        "parser_package": PARSER_PACKAGE,
        "parser_version": PARSER_VERSION,
        "cached_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "qc": asdict(parsed.qc),
    }
    final_metadata = destination / "metadata.json"
    temporary_metadata = destination / "metadata.json.tmp"
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_metadata.replace(final_metadata)
    return destination


def load_cached_replay(cache_dir: str | Path, replay_id: str) -> ParsedReplay:
    """Load a previously validated parser cache."""

    source = Path(cache_dir) / str(replay_id)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("parser_package") != PARSER_PACKAGE or str(
        metadata.get("parser_version")
    ) != PARSER_VERSION:
        raise ReplayParseError(
            "Parser cache provenance does not match the frozen parser package and version."
        )
    qc_payload = dict(metadata["qc"])
    qc_payload["reasons"] = tuple(qc_payload.get("reasons", []))
    return ParsedReplay(
        replay_id=str(replay_id),
        frames=pd.read_parquet(source / "frames.parquet"),
        events=pd.read_parquet(source / "events.parquet"),
        qc=ReplayQC(**qc_payload),
    )
