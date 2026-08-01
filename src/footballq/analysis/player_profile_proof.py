"""Chronological frozen-player-profile proof of value on PFF World Cup tracking."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from footballq.analysis.skillcorner_tactical_transfer import binary_metrics
from footballq.io.pff import PFF_NTSC_FPS, iter_pff_records
from footballq.io.pff_shards import file_sha256
from footballq.models.soccer_state_encoder import SoccerStateEncoder

TARGET_TURNOVER = "turnover_within_5s"
TARGET_PENALTY_ENTRY = "penalty_area_entry_within_5s"
TARGETS = (TARGET_TURNOVER, TARGET_PENALTY_ENTRY)
ROLE_NAMES = ("goalkeeper", "defender", "midfielder", "forward", "unknown")
EVENT_STAT_NAMES = (
    "pass",
    "carry",
    "duel",
    "shot",
    "ball_recovery",
    "pressure",
    "miscontrol_or_dispossessed",
    "other_on_ball",
)
NON_PLAY_EVENTS = {
    "Starting XI",
    "Half Start",
    "Half End",
    "Tactical Shift",
    "Substitution",
    "Player On",
    "Player Off",
    "Injury Stoppage",
    "Referee Ball-Drop",
}


@dataclass(frozen=True)
class MatchInfo:
    """Cross-provider identity and chronology for one PFF match."""

    pff_match_id: str
    statsbomb_match_id: str
    match_datetime: datetime
    stage: str
    match_week: int
    split: str
    home_team_name: str
    away_team_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pff_match_id": self.pff_match_id,
            "statsbomb_match_id": self.statsbomb_match_id,
            "match_datetime": self.match_datetime.isoformat(),
            "stage": self.stage,
            "match_week": self.match_week,
            "split": self.split,
            "home_team_name": self.home_team_name,
            "away_team_name": self.away_team_name,
        }


@dataclass(frozen=True)
class Opportunity:
    """One possession-start prediction opportunity with future-only labels."""

    period: int
    possession: int
    time_s: float
    possession_team_id: str
    turnover_within_5s: int
    penalty_area_entry_within_5s: int
    penalty_area_entry_valid: bool


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_config(path: str | Path) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def timestamp_seconds(value: object) -> float:
    """Parse a StatsBomb period-local ``HH:MM:SS.sss`` timestamp."""

    parts = str(value or "00:00:00").split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid StatsBomb timestamp: {value!r}")
    return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])


def _split_name(stage: str, match_week: int) -> str:
    if stage == "Group Stage" and match_week == 1:
        return "support"
    if stage == "Group Stage" and match_week == 2:
        return "train"
    if stage == "Group Stage" and match_week == 3:
        return "val"
    return "test"


def _pff_match_metadata(
    source_path: Path,
    max_records: int = 5_000,
) -> tuple[dict[str, str], str | None]:
    teams: dict[str, str] = {}
    match_date: str | None = None
    for index, record in enumerate(iter_pff_records(source_path), start=1):
        event = record.get("game_event") or {}
        team_name = str(event.get("team_name") or "").strip()
        home_team = event.get("home_team")
        inserted_at = str(event.get("inserted_at") or "")
        if inserted_at and match_date is None:
            match_date = inserted_at[:10]
        if team_name and home_team in (0, 1, False, True):
            teams["home" if bool(home_team) else "away"] = team_name
        if len(teams) == 2:
            return teams, match_date
        if index >= max_records:
            break
    raise ValueError(
        f"Could not identify both PFF teams within {max_records} records: {source_path}"
    )


def build_match_catalog(
    pff_files: dict[str, Path],
    statsbomb_root: str | Path,
) -> list[MatchInfo]:
    """Map all PFF files to StatsBomb matches by the provider-explicit team pair."""

    sb_root = Path(statsbomb_root)
    metadata = _read_json(sb_root / "matches" / "43" / "106.json")
    by_pair: dict[frozenset[str], list[dict[str, Any]]] = defaultdict(list)
    for match in metadata:
        pair = frozenset(
            {
                str(match["home_team"]["home_team_name"]),
                str(match["away_team"]["away_team_name"]),
            }
        )
        by_pair[pair].append(match)

    catalog: list[MatchInfo] = []
    for pff_match_id, source_path in sorted(
        pff_files.items(), key=lambda item: int(item[0])
    ):
        teams, pff_date = _pff_match_metadata(source_path)
        candidates = by_pair[frozenset(teams.values())]
        if len(candidates) > 1 and pff_date:
            candidates = [
                match for match in candidates if str(match["match_date"]) == pff_date
            ]
        if len(candidates) != 1:
            raise ValueError(
                f"PFF match {pff_match_id} team pair is not unique in StatsBomb: "
                f"{teams}, candidates={len(candidates)}"
            )
        match = candidates[0]
        if teams["home"] != match["home_team"]["home_team_name"]:
            raise ValueError(f"Home-team mismatch while mapping PFF match {pff_match_id}.")
        if teams["away"] != match["away_team"]["away_team_name"]:
            raise ValueError(f"Away-team mismatch while mapping PFF match {pff_match_id}.")
        stage = str(match["competition_stage"]["name"])
        match_week = int(match.get("match_week") or 0)
        match_datetime = datetime.fromisoformat(
            f"{match['match_date']}T{str(match['kick_off']).split('.')[0]}"
        )
        catalog.append(
            MatchInfo(
                pff_match_id=str(pff_match_id),
                statsbomb_match_id=str(match["match_id"]),
                match_datetime=match_datetime,
                stage=stage,
                match_week=match_week,
                split=_split_name(stage, match_week),
                home_team_name=teams["home"],
                away_team_name=teams["away"],
            )
        )
    return sorted(catalog, key=lambda item: (item.match_datetime, item.pff_match_id))


def write_frozen_split_manifest(
    path: str | Path,
    catalog: list[MatchInfo],
) -> tuple[Path, str]:
    """Write the chronology manifest once, or verify that it has not drifted."""

    out = Path(path)
    payload: dict[str, Any] = {
        "version": 1,
        "name": "pff_wc2022_player_profile_chronological_v1",
        "dataset": "pff_fc_world_cup_2022",
        "protocol": (
            "group_round_1_support; group_round_2_train; group_round_3_val; "
            "knockout_test; support_match_datetime_strictly_before_query"
        ),
        "matches": [item.to_dict() for item in catalog],
        "split_match_ids": {
            split: [item.pff_match_id for item in catalog if item.split == split]
            for split in ("support", "train", "val", "test")
        },
    }
    payload["manifest_payload_sha256"] = _stable_hash(payload)
    rendered = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Frozen chronology manifest has drifted: {out}")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
    return out, str(payload["manifest_payload_sha256"])


def build_possession_opportunities(
    events: Iterable[dict[str, Any]],
    *,
    horizon_seconds: float = 5.0,
) -> list[Opportunity]:
    """Create possession-start opportunities without using their future outcomes to select them."""

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        period = int(event.get("period") or 0)
        possession = event.get("possession")
        if period not in {1, 2, 3, 4} or possession is None:
            continue
        grouped[(period, int(possession))].append(event)

    phases: list[dict[str, Any]] = []
    for (period, possession), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row.get("index") or 0))
        usable = [
            row
            for row in ordered
            if row.get("location")
            and str((row.get("type") or {}).get("name") or "") not in NON_PLAY_EVENTS
        ]
        if not usable:
            continue
        first = usable[0]
        start_s = timestamp_seconds(first.get("timestamp"))
        team_id = str((first.get("possession_team") or {}).get("id") or "")
        if not team_id:
            continue

        def in_penalty_area(row: dict[str, Any]) -> bool:
            location = row.get("location")
            return bool(
                isinstance(location, list)
                and len(location) >= 2
                and float(location[0]) >= 102.0
                and 18.0 <= float(location[1]) <= 62.0
            )

        starts_inside = any(
            in_penalty_area(row)
            and timestamp_seconds(row.get("timestamp")) <= start_s + 0.01
            for row in usable
        )
        entry = any(
            in_penalty_area(row)
            and start_s < timestamp_seconds(row.get("timestamp")) <= start_s + horizon_seconds
            for row in usable
        )
        phases.append(
            {
                "period": period,
                "possession": possession,
                "time_s": start_s,
                "team_id": team_id,
                "starts_inside": starts_inside,
                "entry": entry,
            }
        )

    phases.sort(key=lambda row: (row["period"], row["time_s"], row["possession"]))
    opportunities: list[Opportunity] = []
    for index, phase in enumerate(phases):
        next_phase = phases[index + 1] if index + 1 < len(phases) else None
        turnover = bool(
            next_phase is not None
            and next_phase["period"] == phase["period"]
            and next_phase["team_id"] != phase["team_id"]
            and next_phase["time_s"] - phase["time_s"] <= horizon_seconds
        )
        opportunities.append(
            Opportunity(
                period=int(phase["period"]),
                possession=int(phase["possession"]),
                time_s=float(phase["time_s"]),
                possession_team_id=str(phase["team_id"]),
                turnover_within_5s=int(turnover),
                penalty_area_entry_within_5s=int(phase["entry"]),
                penalty_area_entry_valid=not bool(phase["starts_inside"]),
            )
        )
    return opportunities


def broad_role(position_name: object) -> str:
    value = str(position_name or "").lower()
    if "goalkeeper" in value:
        return "goalkeeper"
    if any(token in value for token in ("back", "defender")):
        return "defender"
    if "midfield" in value:
        return "midfielder"
    if any(token in value for token in ("wing", "forward", "striker")):
        return "forward"
    return "unknown"


def lineup_lookup(
    statsbomb_root: Path,
    match: MatchInfo,
) -> tuple[dict[tuple[str, int], str], dict[str, str], dict[str, str]]:
    """Return side+shirt to stable player ID, player role, and player name."""

    lineups = _read_json(statsbomb_root / "lineups" / f"{match.statsbomb_match_id}.json")
    by_slot: dict[tuple[str, int], str] = {}
    roles: dict[str, str] = {}
    names: dict[str, str] = {}
    for team in lineups:
        team_name = str(team["team_name"])
        if team_name == match.home_team_name:
            side = "home"
        elif team_name == match.away_team_name:
            side = "away"
        else:
            raise ValueError(f"Unknown lineup team for PFF match {match.pff_match_id}: {team_name}")
        for row in team["lineup"]:
            player_id = str(row["player_id"])
            jersey = int(row["jersey_number"])
            positions = list(row.get("positions") or [])
            by_slot[(side, jersey)] = player_id
            roles[player_id] = broad_role(
                positions[0].get("position") if positions else None
            )
            names[player_id] = str(row.get("player_name") or player_id)
    return by_slot, roles, names


def event_stats(
    events: Iterable[dict[str, Any]],
    roster_player_ids: Iterable[str],
) -> dict[str, np.ndarray]:
    """Count simple, label-independent event history features per player and match."""

    stats = {
        str(player_id): np.zeros(len(EVENT_STAT_NAMES), dtype=np.float32)
        for player_id in roster_player_ids
    }
    event_index = {
        "Pass": 0,
        "Carry": 1,
        "Duel": 2,
        "Shot": 3,
        "Ball Recovery": 4,
        "Pressure": 5,
        "Miscontrol": 6,
        "Dispossessed": 6,
    }
    for event in events:
        player_id = str((event.get("player") or {}).get("id") or "")
        if not player_id or player_id not in stats:
            continue
        event_name = str((event.get("type") or {}).get("name") or "")
        index = event_index.get(event_name, 7)
        stats[player_id][index] += 1.0
    return {key: np.log1p(value) for key, value in stats.items()}


def _td_entries(
    manifest_paths: Iterable[Path],
) -> tuple[dict[str, list[tuple[Path, dict[str, Any]]]], list[dict[str, Any]]]:
    by_match: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    audits: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = _read_json(manifest_path)
        root = manifest_path.parent.parent
        audits.append(
            {
                "path": str(manifest_path),
                "file_sha256": file_sha256(manifest_path),
                "manifest_payload_sha256": manifest["manifest_payload_sha256"],
                "included_splits": manifest["included_splits"],
            }
        )
        for entry in manifest["shards"]:
            by_match[str(entry["match_id"])].append((root / entry["path"], entry))
    for entries in by_match.values():
        entries.sort(key=lambda item: (int(item[1]["period"]), str(item[1]["path"])))
    return dict(by_match), audits


def _load_match_td(
    entries: list[tuple[Path, dict[str, Any]]],
) -> dict[str, torch.Tensor]:
    state_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    period_parts: list[torch.Tensor] = []
    start_parts: list[torch.Tensor] = []
    end_parts: list[torch.Tensor] = []
    seen: set[str] = set()
    for path, entry in entries:
        if file_sha256(path) != str(entry["tensor_sha256"]):
            raise ValueError(f"TD tensor hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["feature_names"] != [
            "x_norm",
            "y_norm",
            "is_ball",
            "is_home",
            "is_away",
        ]:
            raise ValueError(f"Unexpected TD feature view: {path}")
        keep = [
            index
            for index, sample_id in enumerate(payload["sample_id"])
            if str(sample_id) not in seen
        ]
        seen.update(str(payload["sample_id"][index]) for index in keep)
        if not keep:
            continue
        indices = torch.tensor(keep, dtype=torch.long)
        state_parts.append(payload["state_t"][indices].float())
        mask_parts.append(payload["mask_t"][indices].bool())
        period_parts.append(
            torch.tensor(
                [int(payload["period"][index]) for index in keep],
                dtype=torch.long,
            )
        )
        context_frames = payload["context_frame_indices"][indices].long()
        start_parts.append(context_frames[:, 0])
        end_parts.append(context_frames[:, -1])
    if not state_parts:
        raise ValueError("No TD examples loaded for match.")
    return {
        "state": torch.cat(state_parts),
        "mask": torch.cat(mask_parts),
        "period": torch.cat(period_parts),
        "context_start_frame": torch.cat(start_parts),
        "context_end_frame": torch.cat(end_parts),
    }


def _canonical_manifest(canonical_root: Path, match_id: str) -> tuple[Path, dict[str, Any]]:
    candidates = list(canonical_root.glob(f"*/{match_id}/manifest.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one canonical manifest for PFF match {match_id}, found {len(candidates)}."
        )
    return candidates[0], _read_json(candidates[0])


def _period_frame_estimate(
    canonical_manifest: dict[str, Any],
    period: int,
    time_s: float,
) -> int:
    frame_bounds = canonical_manifest["period_frame_bounds"][str(period)]
    time_bounds = canonical_manifest["period_time_bounds_s"][str(period)]
    return int(
        round(
            int(frame_bounds[0])
            + (float(time_s) - float(time_bounds[0])) * PFF_NTSC_FPS
        )
    )


def _align_opportunities(
    td: dict[str, torch.Tensor],
    opportunities: list[Opportunity],
    canonical_manifest: dict[str, Any],
    *,
    max_gap_frames: int,
    min_visible_entities: int,
    require_ball_visible: bool,
) -> list[tuple[Opportunity, int, int]]:
    by_period: dict[int, tuple[list[int], list[int]]] = {}
    for period in sorted(set(td["period"].tolist())):
        indices = torch.nonzero(td["period"] == period, as_tuple=False).view(-1)
        pairs = sorted(
            (
                int(td["context_end_frame"][index]),
                int(index),
            )
            for index in indices.tolist()
        )
        by_period[period] = ([item[0] for item in pairs], [item[1] for item in pairs])

    rows: list[tuple[Opportunity, int, int]] = []
    used: set[int] = set()
    for opportunity in opportunities:
        candidate = by_period.get(opportunity.period)
        if candidate is None:
            continue
        phase_frame = _period_frame_estimate(
            canonical_manifest,
            opportunity.period,
            opportunity.time_s,
        )
        position = bisect.bisect_left(candidate[0], phase_frame) - 1
        if position < 0:
            continue
        source_index = candidate[1][position]
        context_end = candidate[0][position]
        gap = phase_frame - context_end
        if gap < 1 or gap > max_gap_frames or source_index in used:
            continue
        anchor_mask = td["mask"][source_index, -1]
        if int(anchor_mask.sum()) < min_visible_entities:
            continue
        if require_ball_visible and not bool(anchor_mask[0]):
            continue
        used.add(source_index)
        rows.append((opportunity, source_index, phase_frame))
    return rows


def _profile_indices(
    td: dict[str, torch.Tensor],
    *,
    stride_examples: int,
    min_visible_entities: int,
) -> list[int]:
    selected: list[int] = []
    for period in sorted(set(td["period"].tolist())):
        indices = torch.nonzero(td["period"] == period, as_tuple=False).view(-1).tolist()
        for index in indices[::stride_examples]:
            if int(td["mask"][index, -1].sum()) >= min_visible_entities:
                selected.append(index)
    return selected


def _slot_index(agent_id: object) -> int | None:
    value = str(agent_id)
    if value == "ball":
        return 0
    for side, offset in (("home_slot_", 1), ("away_slot_", 12)):
        if value.startswith(side):
            try:
                slot = int(value.rsplit("_", 1)[-1])
            except ValueError:
                return None
            return offset + slot if 0 <= slot < 11 else None
    return None


def _jersey_number(value: object) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) else None


def anchor_player_ids(
    manifest_path: Path,
    canonical_manifest: dict[str, Any],
    frames: Iterable[int],
    lineup_by_slot: dict[tuple[str, int], str],
) -> dict[int, list[str | None]]:
    """Map dynamic tracking slots to stable StatsBomb player IDs at selected frames."""

    wanted = sorted(set(int(frame) for frame in frames))
    output = {frame: [None] * 23 for frame in wanted}
    for shard in canonical_manifest["shards"]:
        selected = [
            frame
            for frame in wanted
            if int(shard["start_frame"]) <= frame <= int(shard["end_frame"])
        ]
        if not selected:
            continue
        frame = pd.read_parquet(
            manifest_path.parent / shard["path"],
            columns=["frame_id", "agent_id", "team_id", "jersey_number"],
            filters=[("frame_id", "in", selected)],
        )
        for row in frame.itertuples(index=False):
            entity_index = _slot_index(row.agent_id)
            jersey = _jersey_number(row.jersey_number)
            side = str(row.team_id)
            if entity_index is None or entity_index == 0 or jersey is None:
                continue
            output[int(row.frame_id)][entity_index] = lineup_by_slot.get((side, jersey))
    return output


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_frozen_encoder(
    checkpoint_path: Path,
    expected_sha256: str,
    *,
    device: str,
) -> tuple[SoccerStateEncoder, torch.device, dict[str, Any]]:
    actual_sha256 = file_sha256(checkpoint_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data_meta = dict(payload["data_meta"])
    if data_meta["feature_names"] != [
        "x_norm",
        "y_norm",
        "is_ball",
        "is_home",
        "is_away",
    ]:
        raise ValueError("Player-profile proof requires the position-only checkpoint.")
    cfg = dict(payload["config"]["model"])
    encoder = SoccerStateEncoder(
        context_steps=int(data_meta["context_steps"]),
        n_entities=int(data_meta["n_entities"]),
        n_features=int(data_meta["n_features"]),
        z_dim=int(cfg.get("z_dim", 128)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        n_layers=int(cfg.get("n_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        pooling=str(cfg.get("pooling", "mean")),
    )
    encoder.load_state_dict(payload["online_encoder"], strict=True)
    torch_device = _device(device)
    return encoder.to(torch_device).eval(), torch_device, {
        "path": str(checkpoint_path),
        "sha256": actual_sha256,
        "step": int(payload["step"]),
        "data_meta": data_meta,
    }


def _encode(
    encoder: SoccerStateEncoder,
    state: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_parts: list[torch.Tensor] = []
    entity_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(state), batch_size):
            stop = min(start + batch_size, len(state))
            batch_state = state[start:stop].to(device)
            batch_mask = mask[start:stop].to(device)
            global_parts.append(encoder(batch_state, batch_mask).cpu())
            entity_parts.append(encoder.encode_entity_tokens(batch_state, batch_mask).cpu())
    return torch.cat(global_parts), torch.cat(entity_parts)


def _raw_flat(state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = torch.nan_to_num(state) * mask.unsqueeze(-1).to(state.dtype)
    return torch.cat(
        [masked.flatten(start_dim=1), mask.float().flatten(start_dim=1)],
        dim=1,
    )


def _role_flat(player_ids: list[str | None], roles: dict[str, str]) -> torch.Tensor:
    features = torch.zeros(23, len(ROLE_NAMES), dtype=torch.float32)
    features[0, ROLE_NAMES.index("unknown")] = 1.0
    for index, player_id in enumerate(player_ids[1:], start=1):
        role = roles.get(str(player_id), "unknown") if player_id is not None else "unknown"
        features[index, ROLE_NAMES.index(role)] = 1.0
    return features.flatten()


def _match_profile(
    tokens: torch.Tensor,
    masks: torch.Tensor,
    player_ids: list[list[str | None]],
) -> dict[str, dict[str, Any]]:
    values: dict[str, list[torch.Tensor]] = defaultdict(list)
    for clip_index, ids in enumerate(player_ids):
        visible = masks[clip_index].any(dim=0)
        for entity_index, player_id in enumerate(ids):
            if entity_index == 0 or player_id is None or not bool(visible[entity_index]):
                continue
            values[str(player_id)].append(tokens[clip_index, entity_index])
    output: dict[str, dict[str, Any]] = {}
    for player_id, parts in values.items():
        stacked = torch.stack(parts)
        output[player_id] = {
            "mean": stacked.mean(dim=0),
            "variance": float(stacked.var(dim=0, unbiased=False).mean()),
            "clips": len(parts),
        }
    return output


def _support_rows(
    per_match: dict[str, dict[str, Any]],
    matches_by_id: dict[str, MatchInfo],
    *,
    player_id: str,
    query_datetime: datetime,
    k: int,
) -> list[tuple[MatchInfo, Any]]:
    rows = [
        (matches_by_id[match_id], values[player_id])
        for match_id, values in per_match.items()
        if player_id in values
        and matches_by_id[match_id].match_datetime < query_datetime
    ]
    rows.sort(key=lambda item: (item[0].match_datetime, item[0].pff_match_id))
    return rows[-k:]


def build_history_features(
    query_player_ids: list[list[str | None]],
    query_datetimes: list[datetime],
    *,
    k: int,
    match_profiles: dict[str, dict[str, Any]],
    match_event_stats: dict[str, dict[str, np.ndarray]],
    matches_by_id: dict[str, MatchInfo],
    embedding_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build slot-aligned event and latent histories using prior matches only."""

    event_features = torch.zeros(
        len(query_player_ids),
        23,
        len(EVENT_STAT_NAMES) + 2,
        dtype=torch.float32,
    )
    profile_features = torch.zeros(
        len(query_player_ids),
        23,
        embedding_dim + 3,
        dtype=torch.float32,
    )
    support_counts: list[int] = []
    covered_slots = 0
    player_slots = 0
    for row_index, (player_ids, query_datetime) in enumerate(
        zip(query_player_ids, query_datetimes, strict=True)
    ):
        for entity_index, player_id in enumerate(player_ids):
            if entity_index == 0 or player_id is None:
                continue
            player_slots += 1
            profile_support = _support_rows(
                match_profiles,
                matches_by_id,
                player_id=str(player_id),
                query_datetime=query_datetime,
                k=k,
            )
            stats_support = _support_rows(
                match_event_stats,
                matches_by_id,
                player_id=str(player_id),
                query_datetime=query_datetime,
                k=k,
            )
            if stats_support:
                stacked_stats = torch.tensor(
                    np.stack([value for _match, value in stats_support]),
                    dtype=torch.float32,
                )
                event_features[row_index, entity_index, : len(EVENT_STAT_NAMES)] = (
                    stacked_stats.mean(dim=0)
                )
                event_features[row_index, entity_index, -2] = math.log1p(
                    len(stats_support)
                )
                event_features[row_index, entity_index, -1] = 1.0
            support_counts.append(len(profile_support))
            if profile_support:
                covered_slots += 1
                means = torch.stack([value["mean"] for _match, value in profile_support])
                within_variance = float(
                    np.mean([value["variance"] for _match, value in profile_support])
                )
                between_variance = float(means.var(dim=0, unbiased=False).mean())
                profile_features[row_index, entity_index, :embedding_dim] = means.mean(dim=0)
                profile_features[row_index, entity_index, -3] = math.log1p(
                    len(profile_support)
                )
                profile_features[row_index, entity_index, -2] = math.log1p(
                    sum(value["clips"] for _match, value in profile_support)
                )
                profile_features[row_index, entity_index, -1] = math.log1p(
                    within_variance + between_variance
                )
    audit = {
        "k": int(k),
        "player_slots": player_slots,
        "covered_player_slots": covered_slots,
        "coverage": covered_slots / player_slots if player_slots else 0.0,
        "support_count_distribution": {
            str(value): support_counts.count(value) for value in sorted(set(support_counts))
        },
    }
    return event_features.flatten(start_dim=1), profile_features.flatten(start_dim=1), audit


def static_identity_features(
    query_player_ids: list[list[str | None]],
    split_names: list[str],
) -> tuple[torch.Tensor, dict[str, int]]:
    train_players = sorted(
        {
            str(player_id)
            for player_ids, split in zip(query_player_ids, split_names, strict=True)
            if split == "train"
            for player_id in player_ids[1:]
            if player_id is not None
        }
    )
    vocabulary = {player_id: index for index, player_id in enumerate(train_players)}
    features = torch.zeros(len(query_player_ids), len(vocabulary) * 2)
    for row_index, player_ids in enumerate(query_player_ids):
        for entity_index, player_id in enumerate(player_ids[1:], start=1):
            column = vocabulary.get(str(player_id))
            if column is None:
                continue
            side_offset = 0 if entity_index < 12 else len(vocabulary)
            features[row_index, side_offset + column] = 1.0
    return features, vocabulary


def _extended_binary_metrics(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    result = binary_metrics(labels, probabilities)
    probabilities = probabilities.float().clamp(1e-7, 1.0 - 1e-7)
    labels_float = labels.float()
    result["log_loss"] = float(
        torch.nn.functional.binary_cross_entropy(probabilities, labels_float)
    )
    ece = 0.0
    for lower in torch.linspace(0.0, 1.0, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        selected = (probabilities >= lower) & (
            probabilities < upper if upper < 1.0 else probabilities <= upper
        )
        if bool(selected.any()):
            ece += float(selected.float().mean()) * abs(
                float(probabilities[selected].mean()) - float(labels_float[selected].mean())
            )
    result["expected_calibration_error"] = ece
    return result


def _standardize(
    features: torch.Tensor,
    train_indices: list[int],
) -> torch.Tensor:
    train = features[train_indices].float()
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, unbiased=False, keepdim=True)
    # Later chronological splits can have larger support counts than training.
    # Unit scaling for train-constant columns prevents those legal values from
    # exploding solely because their training variance was zero.
    std = torch.where(std < 1e-3, torch.ones_like(std), std)
    return (features.float() - mean) / std


def fit_logistic_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    split_indices: dict[str, list[int]],
    *,
    device: str,
    max_iterations: int,
    history_size: int,
    l2_weight: float,
) -> dict[str, Any]:
    train_indices = [
        index for index in split_indices["train"] if bool(valid_mask[index])
    ]
    standardized = _standardize(features, train_indices)
    torch_device = _device(device)
    x_train = standardized[train_indices].to(torch_device)
    y_train = labels[train_indices].float().to(torch_device)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Probe training requires both classes.")
    model = torch.nn.Linear(features.shape[1], 1).to(torch_device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=max_iterations,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )
    pos_weight = torch.tensor(negatives / positives, device=torch_device)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train).view(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y_train,
            pos_weight=pos_weight,
        )
        loss = loss + 0.5 * l2_weight * model.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.inference_mode():
        probabilities = torch.sigmoid(
            model(standardized.to(torch_device)).view(-1)
        ).cpu()
    return {
        "probabilities": probabilities,
        "metrics": {
            split: _extended_binary_metrics(
                labels[[index for index in indices if bool(valid_mask[index])]],
                probabilities[[index for index in indices if bool(valid_mask[index])]],
            )
            for split, indices in split_indices.items()
        },
    }


def match_bootstrap_gain(
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    probabilities_a: torch.Tensor,
    probabilities_b: torch.Tensor,
    match_ids: list[str],
    test_indices: list[int],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_match = {
        match_id: [
            index
            for index in test_indices
            if match_ids[index] == match_id and bool(valid_mask[index])
        ]
        for match_id in sorted({match_ids[index] for index in test_indices})
    }
    by_match = {key: value for key, value in by_match.items() if value}
    rng = random.Random(seed)
    gains: list[float] = []
    match_names = list(by_match)
    for _ in range(samples):
        selected_matches = [rng.choice(match_names) for _ in match_names]
        indices = [
            index
            for match_id in selected_matches
            for index in by_match[match_id]
        ]
        metric_a = _extended_binary_metrics(labels[indices], probabilities_a[indices])
        metric_b = _extended_binary_metrics(labels[indices], probabilities_b[indices])
        if metric_a["macro_f1"] is not None and metric_b["macro_f1"] is not None:
            gains.append(float(metric_b["macro_f1"] - metric_a["macro_f1"]))
    ordered = sorted(gains)
    return {
        "samples": len(gains),
        "mean_macro_f1_gain": float(np.mean(gains)),
        "positive_fraction": float(np.mean(np.asarray(gains) > 0.0)),
        "ci95": [
            float(np.quantile(ordered, 0.025)),
            float(np.quantile(ordered, 0.975)),
        ],
    }


def _jsonable_probe(result: dict[str, Any]) -> dict[str, Any]:
    return {"metrics": result["metrics"]}


def _shuffle_profile_features(
    features: torch.Tensor,
    split_names: list[str],
    roles: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Shuffle slot profiles within broad-role and split groups."""

    generator = torch.Generator().manual_seed(seed)
    reshaped = features.view(len(features), 23, -1).clone()
    role_rows = roles.view(len(roles), 23, len(ROLE_NAMES)).argmax(dim=-1)
    for split in ("train", "val", "test"):
        row_indices = [index for index, value in enumerate(split_names) if value == split]
        for role_index in range(len(ROLE_NAMES)):
            slots = [
                (row_index, entity_index)
                for row_index in row_indices
                for entity_index in range(1, 23)
                if int(role_rows[row_index, entity_index]) == role_index
                and bool(reshaped[row_index, entity_index].abs().sum())
            ]
            if len(slots) < 2:
                continue
            order = torch.randperm(len(slots), generator=generator).tolist()
            source = [reshaped[row, entity].clone() for row, entity in slots]
            for target_index, (row, entity) in enumerate(slots):
                reshaped[row, entity] = source[order[target_index]]
    return reshaped.flatten(start_dim=1)


def build_feature_cache(
    config: dict[str, Any],
    *,
    workspace_root: str | Path,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build all causal query and prior-match features, then return tensor and audit payloads."""

    root = Path(workspace_root)
    data_cfg = dict(config["data"])
    pff_root = root / data_cfg["pff_raw_root"]
    canonical_root = root / data_cfg["canonical_root"]
    statsbomb_root = root / data_cfg["statsbomb_root"]
    checkpoint_path = root / config["encoder"]["checkpoint"]
    manifest_paths = [root / value for value in data_cfg["td_manifests"]]

    pff_files = {
        path.parent.stem: path
        for path in pff_root.glob("*.jsonl/*.jsonl")
        if path.is_file()
    }
    catalog = build_match_catalog(pff_files, statsbomb_root)
    split_path, split_hash = write_frozen_split_manifest(
        root / data_cfg["chronology_manifest"],
        catalog,
    )
    matches_by_id = {item.pff_match_id: item for item in catalog}
    td_entries, td_manifest_audit = _td_entries(manifest_paths)
    if set(td_entries) != set(matches_by_id):
        raise ValueError(
            "Combined train/val and confirmatory TD manifests do not cover the 64-match catalog."
        )
    encoder, torch_device, checkpoint_audit = load_frozen_encoder(
        checkpoint_path,
        str(config["encoder"]["sha256"]),
        device=device,
    )

    match_profiles: dict[str, dict[str, Any]] = {}
    match_event_stats: dict[str, dict[str, np.ndarray]] = {}
    query_state: list[torch.Tensor] = []
    query_mask: list[torch.Tensor] = []
    query_latent: list[torch.Tensor] = []
    query_roles: list[torch.Tensor] = []
    query_players: list[list[str | None]] = []
    query_datetimes: list[datetime] = []
    query_match_ids: list[str] = []
    query_split: list[str] = []
    query_period: list[int] = []
    query_possession: list[int] = []
    context_end_frames: list[int] = []
    phase_start_frames: list[int] = []
    labels = {target: [] for target in TARGETS}
    label_masks = {target: [] for target in TARGETS}
    player_names: dict[str, str] = {}
    match_audit: dict[str, Any] = {}

    for match_number, match in enumerate(catalog, start=1):
        print(
            f"[profile-cache {match_number:02d}/{len(catalog):02d}] "
            f"PFF {match.pff_match_id} ({match.split})",
            flush=True,
        )
        sb_events = _read_json(
            statsbomb_root / "events" / f"{match.statsbomb_match_id}.json"
        )
        lineup_by_slot, roles, names = lineup_lookup(statsbomb_root, match)
        player_names.update(names)
        match_event_stats[match.pff_match_id] = event_stats(sb_events, roles)
        opportunities = build_possession_opportunities(
            sb_events,
            horizon_seconds=float(data_cfg["prediction_horizon_seconds"]),
        )
        td = _load_match_td(td_entries[match.pff_match_id])
        canonical_path, canonical_manifest = _canonical_manifest(
            canonical_root,
            match.pff_match_id,
        )
        aligned = _align_opportunities(
            td,
            opportunities,
            canonical_manifest,
            max_gap_frames=int(data_cfg["query_max_alignment_gap_frames"]),
            min_visible_entities=int(data_cfg["min_visible_entities_at_anchor"]),
            require_ball_visible=bool(data_cfg["require_ball_visible_at_anchor"]),
        )
        profile_indices = _profile_indices(
            td,
            stride_examples=int(data_cfg["profile_stride_examples"]),
            min_visible_entities=int(data_cfg["min_visible_entities_at_anchor"]),
        )
        query_indices = [row[1] for row in aligned] if match.split != "support" else []
        selected_indices = profile_indices + query_indices
        selected_state = td["state"][selected_indices]
        selected_mask = td["mask"][selected_indices]
        global_latent, entity_tokens = _encode(
            encoder,
            selected_state,
            selected_mask,
            device=torch_device,
            batch_size=int(config["encoder"]["batch_size"]),
        )
        selected_frames = [
            int(td["context_end_frame"][index]) for index in selected_indices
        ]
        player_ids_by_frame = anchor_player_ids(
            canonical_path,
            canonical_manifest,
            selected_frames,
            lineup_by_slot,
        )
        profile_player_ids = [
            player_ids_by_frame[int(td["context_end_frame"][index])]
            for index in profile_indices
        ]
        match_profiles[match.pff_match_id] = _match_profile(
            entity_tokens[: len(profile_indices)],
            selected_mask[: len(profile_indices)],
            profile_player_ids,
        )

        if match.split != "support":
            offset = len(profile_indices)
            for local_index, (opportunity, source_index, phase_frame) in enumerate(aligned):
                anchor_frame = int(td["context_end_frame"][source_index])
                player_ids = player_ids_by_frame[anchor_frame]
                query_state.append(td["state"][source_index])
                query_mask.append(td["mask"][source_index])
                query_latent.append(global_latent[offset + local_index])
                query_roles.append(_role_flat(player_ids, roles))
                query_players.append(player_ids)
                query_datetimes.append(match.match_datetime)
                query_match_ids.append(match.pff_match_id)
                query_split.append(match.split)
                query_period.append(opportunity.period)
                query_possession.append(opportunity.possession)
                context_end_frames.append(anchor_frame)
                phase_start_frames.append(phase_frame)
                labels[TARGET_TURNOVER].append(opportunity.turnover_within_5s)
                labels[TARGET_PENALTY_ENTRY].append(
                    opportunity.penalty_area_entry_within_5s
                )
                label_masks[TARGET_TURNOVER].append(True)
                label_masks[TARGET_PENALTY_ENTRY].append(
                    opportunity.penalty_area_entry_valid
                )
        match_audit[match.pff_match_id] = {
            "split": match.split,
            "statsbomb_match_id": match.statsbomb_match_id,
            "opportunities": len(opportunities),
            "aligned_queries": len(aligned) if match.split != "support" else 0,
            "profile_clips": len(profile_indices),
            "profile_players": len(match_profiles[match.pff_match_id]),
        }

    state = torch.stack(query_state)
    mask = torch.stack(query_mask)
    role_features = torch.stack(query_roles)
    raw_features = _raw_flat(state, mask)
    current_latent = torch.stack(query_latent)
    static_features, static_vocabulary = static_identity_features(
        query_players,
        query_split,
    )
    split_indices = {
        split: [index for index, value in enumerate(query_split) if value == split]
        for split in ("train", "val", "test")
    }

    history: dict[int, dict[str, torch.Tensor]] = {}
    history_audit: dict[int, Any] = {}
    for k in [int(value) for value in config["profiles"]["support_sizes"]]:
        event_history, player_history, audit = build_history_features(
            query_players,
            query_datetimes,
            k=k,
            match_profiles=match_profiles,
            match_event_stats=match_event_stats,
            matches_by_id=matches_by_id,
            embedding_dim=int(config["encoder"]["entity_embedding_dim"]),
        )
        history[k] = {
            "event": event_history,
            "profile": player_history,
        }
        history_audit[k] = audit

    tensor_payload = {
        "version": 1,
        "experiment": config["experiment"],
        "raw": raw_features,
        "role": role_features,
        "current_latent": current_latent,
        "static_identity": static_features,
        "history": history,
        "labels": {
            target: torch.tensor(values, dtype=torch.long)
            for target, values in labels.items()
        },
        "label_masks": {
            target: torch.tensor(values, dtype=torch.bool)
            for target, values in label_masks.items()
        },
        "split_indices": split_indices,
        "match_id": query_match_ids,
        "split": query_split,
        "period": query_period,
        "possession": query_possession,
        "context_end_frame": context_end_frames,
        "phase_start_frame": phase_start_frames,
        "query_player_ids": query_players,
        "chronology_manifest_path": str(split_path),
        "chronology_manifest_payload_sha256": split_hash,
        "checkpoint": checkpoint_audit,
    }
    audit_payload = {
        "experiment": config["experiment"],
        "config_sha256": _stable_hash(config),
        "chronology_manifest_path": str(split_path),
        "chronology_manifest_payload_sha256": split_hash,
        "checkpoint": checkpoint_audit,
        "td_manifests": td_manifest_audit,
        "match_audit": match_audit,
        "history_audit": history_audit,
        "static_identity_vocabulary_size": len(static_vocabulary),
        "player_names_count": len(player_names),
        "split_counts": {
            split: len(indices) for split, indices in split_indices.items()
        },
        "label_support": {
            split: {
                target: {
                    "valid": int(
                        tensor_payload["label_masks"][target][indices].sum()
                    ),
                    "positive": int(
                        tensor_payload["labels"][target][indices][
                            tensor_payload["label_masks"][target][indices]
                        ].sum()
                    ),
                }
                for target in TARGETS
            }
            for split, indices in split_indices.items()
        },
        "causality": {
            "context_strictly_before_phase": all(
                end < start
                for end, start in zip(
                    context_end_frames,
                    phase_start_frames,
                    strict=True,
                )
            ),
            "profile_support_rule": "support_match_datetime < query_match_datetime",
            "same_match_profile_excluded": True,
        },
    }
    return tensor_payload, audit_payload


def evaluate_feature_cache(
    payload: dict[str, Any],
    config: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    """Run the A-F ladder, profile controls, support curve, and match bootstrap."""

    main_k = int(config["profiles"]["main_support_size"])
    raw = payload["raw"]
    role = payload["role"]
    current_latent = payload["current_latent"]
    static_identity = payload["static_identity"]
    history = payload["history"]
    split_names = payload["split"]
    features: dict[str, torch.Tensor] = {
        "A_raw_geometry": raw,
        "B_raw_plus_role": torch.cat([raw, role], dim=1),
        "C_raw_plus_current_latent": torch.cat([raw, current_latent], dim=1),
        "D_raw_plus_static_identity": torch.cat([raw, static_identity], dim=1),
    }
    for k, values in history.items():
        features[f"E_raw_plus_rolling_stats_k{k}"] = torch.cat(
            [raw, values["event"]],
            dim=1,
        )
        features[f"F_raw_plus_player_profile_k{k}"] = torch.cat(
            [raw, values["profile"]],
            dim=1,
        )
    shuffled = _shuffle_profile_features(
        history[main_k]["profile"],
        split_names,
        role,
        seed=int(config["evaluation"]["shuffle_seed"]),
    )
    features[f"F_control_same_role_shuffled_k{main_k}"] = torch.cat(
        [raw, shuffled],
        dim=1,
    )

    probe_cfg = config["probe"]
    rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str], dict[str, Any]] = {}
    for feature_name, feature_tensor in features.items():
        for target in TARGETS:
            result = fit_logistic_probe(
                feature_tensor,
                payload["labels"][target],
                payload["label_masks"][target],
                payload["split_indices"],
                device=device,
                max_iterations=int(probe_cfg["max_iterations"]),
                history_size=int(probe_cfg["history_size"]),
                l2_weight=float(probe_cfg["l2_weight"]),
            )
            fitted[(feature_name, target)] = result
            rows.append(
                {
                    "feature": feature_name,
                    "target": target,
                    "feature_dim": int(feature_tensor.shape[1]),
                    **_jsonable_probe(result),
                }
            )

    event_name = f"E_raw_plus_rolling_stats_k{main_k}"
    profile_name = f"F_raw_plus_player_profile_k{main_k}"
    raw_name = "A_raw_geometry"
    shuffled_name = f"F_control_same_role_shuffled_k{main_k}"
    gates: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    for target in TARGETS:
        event_result = fitted[(event_name, target)]
        profile_result = fitted[(profile_name, target)]
        raw_result = fitted[(raw_name, target)]
        shuffled_result = fitted[(shuffled_name, target)]
        event_test = event_result["metrics"]["test"]
        profile_test = profile_result["metrics"]["test"]
        raw_test = raw_result["metrics"]["test"]
        shuffled_test = shuffled_result["metrics"]["test"]
        bootstrap[target] = match_bootstrap_gain(
            payload["labels"][target],
            payload["label_masks"][target],
            event_result["probabilities"],
            profile_result["probabilities"],
            payload["match_id"],
            payload["split_indices"]["test"],
            samples=int(config["evaluation"]["match_bootstrap_samples"]),
            seed=int(config["evaluation"]["bootstrap_seed"]),
        )
        gates[target] = {
            "macro_f1_gain_f_over_e": float(
                profile_test["macro_f1"] - event_test["macro_f1"]
            ),
            "average_precision_gain_f_over_e": float(
                profile_test["average_precision"] - event_test["average_precision"]
            ),
            "macro_f1_gain_f_over_raw": float(
                profile_test["macro_f1"] - raw_test["macro_f1"]
            ),
            "macro_f1_gain_f_over_same_role_shuffle": float(
                profile_test["macro_f1"] - shuffled_test["macro_f1"]
            ),
            "average_precision_gain_f_over_same_role_shuffle": float(
                profile_test["average_precision"] - shuffled_test["average_precision"]
            ),
            "brier_improved": profile_test["brier"] < event_test["brier"],
            "log_loss_improved": profile_test["log_loss"] < event_test["log_loss"],
            "macro_f1_threshold": float(config["gates"]["macro_f1_gain"]),
        }
        gates[target]["task_passed"] = bool(
            (
                gates[target]["macro_f1_gain_f_over_e"]
                >= gates[target]["macro_f1_threshold"]
                or gates[target]["average_precision_gain_f_over_e"]
                >= float(config["gates"]["material_average_precision_gain"])
            )
            and (
                gates[target]["brier_improved"]
                or gates[target]["log_loss_improved"]
            )
            and gates[target]["macro_f1_gain_f_over_raw"] > 0.0
            and (
                gates[target]["macro_f1_gain_f_over_same_role_shuffle"] > 0.0
                or gates[target]["average_precision_gain_f_over_same_role_shuffle"] > 0.0
            )
            and bootstrap[target]["mean_macro_f1_gain"] > 0.0
        )

    return {
        "experiment": config["experiment"],
        "protocol_status": "frozen_single_checkpoint_bounded_proof",
        "rows": rows,
        "main_comparison": {
            "rolling_event_statistics": event_name,
            "history_derived_player_profiles": profile_name,
        },
        "match_bootstrap_f_over_e": bootstrap,
        "decision_gates": {
            "targets": gates,
            "tasks_passed": sum(value["task_passed"] for value in gates.values()),
            "required_tasks": int(config["gates"]["minimum_tasks_improved"]),
            "proceed_to_identity_aware_pretraining": (
                sum(value["task_passed"] for value in gates.values())
                >= int(config["gates"]["minimum_tasks_improved"])
            ),
        },
        "interpretation_limit": (
            "This is a frozen-encoder, lineup-slot-aggregated profile test on one "
            "competition. It tests incremental predictive value from prior-match "
            "player histories; it does not establish tactical understanding or "
            "full-match planning."
        ),
    }


def run_experiment(
    config_path: str | Path,
    *,
    workspace_root: str | Path,
    device: str = "auto",
    rebuild_cache: bool = False,
) -> dict[str, Path]:
    root = Path(workspace_root)
    config = load_config(config_path)
    run_dir = root / config["output"]["run_dir"]
    cache_path = run_dir / "feature_cache.pt"
    audit_path = run_dir / "feature_audit.json"
    result_path = run_dir / "results.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    if rebuild_cache or not cache_path.exists():
        payload, audit = build_feature_cache(
            config,
            workspace_root=root,
            device=device,
        )
        torch.save(payload, cache_path)
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    else:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit["config_sha256"] != _stable_hash(config):
            raise ValueError("Cached profile features were built under a different config.")
    result = evaluate_feature_cache(payload, config, device=device)
    result["feature_audit_path"] = str(audit_path)
    result["feature_cache_path"] = str(cache_path)
    result["config_path"] = str(config_path)
    result["config_sha256"] = _stable_hash(config)
    result["config_file_sha256"] = file_sha256(config_path)
    result["experiment_code_sha256"] = file_sha256(Path(__file__))
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {
        "run_dir": run_dir,
        "feature_cache": cache_path,
        "feature_audit": audit_path,
        "results": result_path,
    }
