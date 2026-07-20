"""Train-only diagnostics for GRF position discontinuities."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from footballq.constants import PITCH_LENGTH_M, PITCH_WIDTH_M
from footballq.io.gfootball import _extract_observation
from footballq.repro.splits import split_manifest_sha256

PLAYER_JUMP_THRESHOLD_M = 3.0
PLAYER_SPEED_THRESHOLD_MPS = 12.0
PLAYER_ACCELERATION_THRESHOLD_MPS2 = 100.0
BALL_JUMP_THRESHOLD_M = 10.0
BALL_SPEED_THRESHOLD_MPS = 60.0
BALL_ACCELERATION_THRESHOLD_MPS2 = 300.0
EVENT_WINDOW_FRAMES = 5

_METRIC_NAMES = ("position_jump_m", "causal_speed_mps", "causal_acceleration_mps2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xy_m(value: Any) -> tuple[float, float]:
    xy = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(xy) < 2 or not np.isfinite(xy[:2]).all():
        return float("nan"), float("nan")
    x_m = (float(xy[0]) + 1.0) / 2.0 * PITCH_LENGTH_M
    y_m = (float(xy[1]) + 0.42) / 0.84 * PITCH_WIDTH_M
    return x_m, y_m


def _score(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in np.asarray(value if value is not None else []).reshape(-1))


def _frame_from_record(record: dict[str, Any]) -> dict[str, Any]:
    obs = _extract_observation(record)
    entities: dict[str, dict[str, Any]] = {
        "ball": {"xy": _xy_m(obs.get("ball", [])), "active": True, "type": "ball"}
    }
    for team_key, active_key, prefix in (
        ("left_team", "left_team_active", "home"),
        ("right_team", "right_team_active", "away"),
    ):
        positions = np.asarray(obs.get(team_key, []), dtype=np.float64)
        if positions.size:
            positions = positions.reshape(-1, 2)
        else:
            positions = np.empty((0, 2), dtype=np.float64)
        active = np.asarray(obs.get(active_key, np.ones(len(positions))), dtype=bool).reshape(-1)
        for index, position in enumerate(positions):
            entities[f"{prefix}_{index:02d}"] = {
                "xy": _xy_m(position),
                "active": bool(active[index]) if index < len(active) else True,
                "type": "player",
            }
    return {
        "match_id": str(record.get("match_id", "")),
        "frame_id": int(record["frame_id"]),
        "time_s": float(record["time_s"]),
        "game_mode": int(obs.get("game_mode", 0)),
        "score": _score(obs.get("score")),
        "steps_left": int(obs.get("steps_left", -1)),
        "entities": entities,
    }


def _empty_metrics() -> dict[str, dict[str, list[float]]]:
    return {
        family: {name: [] for name in _METRIC_NAMES}
        for family in ("players", "ball")
    }


def _empty_attribution() -> dict[str, float | int]:
    return {
        "extreme_count": 0,
        "extreme_acceleration_mass": 0.0,
        "event_proximate_count": 0,
        "event_proximate_mass": 0.0,
        "jump_associated_count": 0,
        "jump_associated_mass": 0.0,
        "game_mode_change_nearby_count": 0,
        "game_mode_change_nearby_mass": 0.0,
        "nonzero_game_mode_nearby_count": 0,
        "nonzero_game_mode_nearby_mass": 0.0,
        "score_change_nearby_count": 0,
        "score_change_nearby_mass": 0.0,
    }


def _add_attribution(
    target: dict[str, float | int],
    *,
    acceleration: float,
    event_flags: dict[str, bool],
    jump_associated: bool,
) -> None:
    target["extreme_count"] += 1
    target["extreme_acceleration_mass"] += acceleration
    if event_flags["event_proximate"]:
        target["event_proximate_count"] += 1
        target["event_proximate_mass"] += acceleration
    if jump_associated:
        target["jump_associated_count"] += 1
        target["jump_associated_mass"] += acceleration
    for name in (
        "game_mode_change_nearby",
        "nonzero_game_mode_nearby",
        "score_change_nearby",
    ):
        if event_flags[name]:
            target[f"{name}_count"] += 1
            target[f"{name}_mass"] += acceleration


def _merge_attribution(
    target: dict[str, float | int], source: dict[str, float | int]
) -> None:
    for name, value in source.items():
        target[name] += value


def _finalize_attribution(values: dict[str, float | int]) -> dict[str, Any]:
    count = int(values["extreme_count"])
    mass = float(values["extreme_acceleration_mass"])
    result = dict(values)
    result.update(
        {
            "event_proximate_count_share": (
                float(values["event_proximate_count"]) / count if count else 0.0
            ),
            "event_proximate_mass_share": (
                float(values["event_proximate_mass"]) / mass if mass else 0.0
            ),
            "jump_associated_count_share": (
                float(values["jump_associated_count"]) / count if count else 0.0
            ),
            "jump_associated_mass_share": (
                float(values["jump_associated_mass"]) / mass if mass else 0.0
            ),
        }
    )
    for name in (
        "game_mode_change_nearby",
        "nonzero_game_mode_nearby",
        "score_change_nearby",
    ):
        result[f"{name}_count_share"] = (
            float(values[f"{name}_count"]) / count if count else 0.0
        )
        result[f"{name}_mass_share"] = (
            float(values[f"{name}_mass"]) / mass if mass else 0.0
        )
    return result


def _summarize(values: list[float], *, threshold: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "extreme_threshold": threshold, "extreme_count": 0}
    quantiles = np.quantile(array, [0.5, 0.95, 0.99, 0.999])
    extreme_count = int((array >= threshold).sum())
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p50": float(quantiles[0]),
        "p95": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "p999": float(quantiles[3]),
        "max": float(array.max()),
        "extreme_threshold": float(threshold),
        "extreme_count": extreme_count,
        "extreme_rate": extreme_count / len(array),
    }


def _thresholds(family: str) -> dict[str, float]:
    if family == "players":
        return {
            "position_jump_m": PLAYER_JUMP_THRESHOLD_M,
            "causal_speed_mps": PLAYER_SPEED_THRESHOLD_MPS,
            "causal_acceleration_mps2": PLAYER_ACCELERATION_THRESHOLD_MPS2,
        }
    return {
        "position_jump_m": BALL_JUMP_THRESHOLD_M,
        "causal_speed_mps": BALL_SPEED_THRESHOLD_MPS,
        "causal_acceleration_mps2": BALL_ACCELERATION_THRESHOLD_MPS2,
    }


def _summarize_metrics(metrics: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    return {
        family: {
            name: _summarize(values, threshold=_thresholds(family)[name])
            for name, values in family_metrics.items()
        }
        for family, family_metrics in metrics.items()
    }


def _merge_metrics(
    target: dict[str, dict[str, list[float]]],
    source: dict[str, dict[str, list[float]]],
) -> None:
    for family in target:
        for name in target[family]:
            target[family][name].extend(source[family][name])


def _event_flags(frames: list[dict[str, Any]], index: int) -> dict[str, bool]:
    start = max(0, index - EVENT_WINDOW_FRAMES)
    end = min(len(frames), index + EVENT_WINDOW_FRAMES + 1)
    window = frames[start:end]
    game_mode_change = any(
        frames[position]["game_mode"] != frames[position - 1]["game_mode"]
        for position in range(max(1, start), end)
    )
    score_change = any(
        frames[position]["score"] != frames[position - 1]["score"]
        for position in range(max(1, start), end)
    )
    nonzero_mode = any(frame["game_mode"] != 0 for frame in window)
    return {
        "game_mode_change_nearby": game_mode_change,
        "nonzero_game_mode_nearby": nonzero_mode,
        "score_change_nearby": score_change,
        "event_proximate": game_mode_change or nonzero_mode or score_change,
    }


def _top_push(
    heap: list[tuple[float, int, dict[str, Any]]],
    *,
    limit: int,
    counter: int,
    row: dict[str, Any],
) -> None:
    item = (float(row["causal_acceleration_mps2"]), counter, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def _process_episode(
    frames: list[dict[str, Any]],
    *,
    job_id: str,
    scenario: str,
    player_top: list[tuple[float, int, dict[str, Any]]],
    ball_top: list[tuple[float, int, dict[str, Any]]],
    top_player_limit: int,
    top_ball_limit: int,
    counter_start: int,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, float | int]], int]:
    metrics = _empty_metrics()
    attribution = {"players": _empty_attribution(), "ball": _empty_attribution()}
    counter = counter_start
    frames.sort(key=lambda frame: frame["frame_id"])
    previous_transitions: dict[str, dict[str, Any]] = {}
    for index in range(1, len(frames)):
        previous = frames[index - 1]
        current = frames[index]
        if current["frame_id"] != previous["frame_id"] + 1:
            previous_transitions = {}
            continue
        dt = current["time_s"] - previous["time_s"]
        if not np.isfinite(dt) or dt <= 0:
            previous_transitions = {}
            continue
        transitions: dict[str, dict[str, Any]] = {}
        for entity_id in sorted(set(previous["entities"]) & set(current["entities"])):
            before = previous["entities"][entity_id]
            after = current["entities"][entity_id]
            position_before = np.asarray(before["xy"], dtype=np.float64)
            position_after = np.asarray(after["xy"], dtype=np.float64)
            if not np.isfinite(position_before).all() or not np.isfinite(position_after).all():
                continue
            delta = position_after - position_before
            velocity = delta / dt
            jump = float(np.linalg.vector_norm(delta))
            speed = float(np.linalg.vector_norm(velocity))
            family = "ball" if after["type"] == "ball" else "players"
            metrics[family]["position_jump_m"].append(jump)
            metrics[family]["causal_speed_mps"].append(speed)
            transition = {
                "frame_before": previous["frame_id"],
                "frame_after": current["frame_id"],
                "position_before": position_before,
                "position_after": position_after,
                "velocity": velocity,
                "jump": jump,
                "active_before": bool(before["active"]),
                "active_after": bool(after["active"]),
            }
            transitions[entity_id] = transition
            earlier = previous_transitions.get(entity_id)
            if earlier is None or earlier["frame_after"] != previous["frame_id"]:
                continue
            acceleration = float(np.linalg.vector_norm(velocity - earlier["velocity"]) / dt)
            metrics[family]["causal_acceleration_mps2"].append(acceleration)
            acceleration_threshold = _thresholds(family)["causal_acceleration_mps2"]
            event_flags = _event_flags(frames, index)
            jump_threshold = _thresholds(family)["position_jump_m"]
            jump_associated = earlier["jump"] >= jump_threshold or jump >= jump_threshold
            if acceleration >= acceleration_threshold:
                _add_attribution(
                    attribution[family],
                    acceleration=acceleration,
                    event_flags=event_flags,
                    jump_associated=jump_associated,
                )
            row = {
                "job_id": job_id,
                "scenario": scenario,
                "match_id": current["match_id"],
                "entity_id": entity_id,
                "entity_type": after["type"],
                "frame_ids": [
                    int(earlier["frame_before"]),
                    int(previous["frame_id"]),
                    int(current["frame_id"]),
                ],
                "positions_m": [
                    earlier["position_before"].tolist(),
                    position_before.tolist(),
                    position_after.tolist(),
                ],
                "causal_velocities_mps": [
                    earlier["velocity"].tolist(),
                    velocity.tolist(),
                ],
                "position_jumps_m": [float(earlier["jump"]), jump],
                "causal_acceleration_mps2": acceleration,
                "active_flags": [
                    bool(earlier["active_before"]),
                    bool(earlier["active_after"]),
                    bool(after["active"]),
                ],
                "game_modes": [
                    int(frames[index - 2]["game_mode"]),
                    int(previous["game_mode"]),
                    int(current["game_mode"]),
                ],
                "scores": [
                    list(frames[index - 2]["score"]),
                    list(previous["score"]),
                    list(current["score"]),
                ],
                "steps_left": [
                    int(frames[index - 2]["steps_left"]),
                    int(previous["steps_left"]),
                    int(current["steps_left"]),
                ],
                **event_flags,
                "jump_associated": jump_associated,
            }
            counter += 1
            if family == "players":
                _top_push(
                    player_top,
                    limit=top_player_limit,
                    counter=counter,
                    row=row,
                )
            else:
                _top_push(ball_top, limit=top_ball_limit, counter=counter, row=row)
        previous_transitions = transitions
    return metrics, attribution, counter


def _resolve_source_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _decision(player_attribution: dict[str, Any]) -> dict[str, Any]:
    event_share = float(player_attribution["event_proximate_mass_share"])
    jump_share = float(player_attribution["jump_associated_mass_share"])
    if event_share >= 0.8:
        selected = "event_boundary_segmentation_or_masking"
    elif jump_share >= 0.8:
        selected = "generic_jump_boundary_mask"
    else:
        selected = "no_reset_aware_causal_position_redesign_authorized"
    return {
        "selected_next_candidate": selected,
        "event_proximate_mass_share": event_share,
        "event_proximate_minimum": 0.8,
        "jump_associated_mass_share": jump_share,
        "jump_associated_minimum": 0.8,
    }


def run_gfootball_position_discontinuity_audit(
    collection_manifest_path: str | Path,
    split_manifest_path: str | Path,
    *,
    repo_root: str | Path,
    expected_collection_plan_sha256: str | None = None,
    expected_split_manifest_sha256: str | None = None,
    top_player_limit: int = 100,
    top_ball_limit: int = 50,
) -> dict[str, Any]:
    """Audit raw GRF training positions without reading held-out jobs."""

    collection_path = Path(collection_manifest_path)
    split_path = Path(split_manifest_path)
    root = Path(repo_root)
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    actual_split_sha256 = split_manifest_sha256(split)
    collection_plan_sha256 = str(collection["collection_plan_sha256"])
    if (
        expected_collection_plan_sha256 is not None
        and collection_plan_sha256 != expected_collection_plan_sha256
    ):
        raise ValueError("Collection-plan SHA-256 does not match the frozen protocol.")
    if (
        expected_split_manifest_sha256 is not None
        and actual_split_sha256 != expected_split_manifest_sha256
    ):
        raise ValueError("Split-manifest SHA-256 does not match the frozen protocol.")

    train_ids = set(map(str, split["train_match_ids"]))
    held_out_ids = set(map(str, split.get("val_match_ids", []))) | set(
        map(str, split.get("test_match_ids", []))
    )
    train_jobs = [job for job in collection["jobs"] if job.get("split") == "train"]
    if not train_jobs:
        raise ValueError("Collection manifest has no train jobs.")

    global_metrics = _empty_metrics()
    global_attribution = {"players": _empty_attribution(), "ball": _empty_attribution()}
    scenario_metrics: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(_empty_metrics)
    scenario_attribution: dict[str, dict[str, dict[str, float | int]]] = defaultdict(
        lambda: {"players": _empty_attribution(), "ball": _empty_attribution()}
    )
    by_job: dict[str, Any] = {}
    observed_match_ids: set[str] = set()
    player_top: list[tuple[float, int, dict[str, Any]]] = []
    ball_top: list[tuple[float, int, dict[str, Any]]] = []
    counter = 0
    total_frames = 0

    for job in train_jobs:
        job_id = str(job["id"])
        source_path = _resolve_source_path(str(job["path"]), root).resolve()
        expected_train_root = (root / "data" / "raw" / "gfootball" / "v2_pilot" / "train").resolve()
        if expected_train_root not in source_path.parents:
            raise ValueError(f"Train job {job_id!r} is outside the frozen train root.")
        actual_source_sha256 = _sha256(source_path)
        if actual_source_sha256 != job["sha256"]:
            raise ValueError(f"Raw checksum mismatch for train job {job_id!r}.")
        episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        observed_scenarios: set[str] = set()
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("split") != "train":
                    raise ValueError(
                        f"Non-train record in {job_id!r} at line {line_number}."
                    )
                if str(record.get("collection_job_id")) != job_id:
                    raise ValueError(f"Job ID mismatch in {job_id!r} at line {line_number}.")
                match_id = str(record.get("match_id", ""))
                if match_id not in train_ids or match_id in held_out_ids:
                    raise ValueError(
                        f"Held-out or unknown episode in {job_id!r} at line {line_number}."
                    )
                frame = _frame_from_record(record)
                episodes[match_id].append(frame)
                observed_scenarios.add(str(record.get("env_name", "unknown")))
                observed_match_ids.add(match_id)
                total_frames += 1

        observed_job_frames = sum(len(frames) for frames in episodes.values())
        if "frames" in job and observed_job_frames != int(job["frames"]):
            raise ValueError(
                f"Frame-count mismatch for train job {job_id!r}: "
                f"expected {job['frames']}, observed {observed_job_frames}."
            )
        if len(observed_scenarios) != 1:
            raise ValueError(
                f"Train job {job_id!r} contains inconsistent scenarios: "
                f"{sorted(observed_scenarios)}"
            )

        job_metrics = _empty_metrics()
        job_attribution = {"players": _empty_attribution(), "ball": _empty_attribution()}
        scenario = next(iter(observed_scenarios))
        for match_id in sorted(episodes):
            episode_metrics, episode_attribution, counter = _process_episode(
                episodes[match_id],
                job_id=job_id,
                scenario=scenario,
                player_top=player_top,
                ball_top=ball_top,
                top_player_limit=top_player_limit,
                top_ball_limit=top_ball_limit,
                counter_start=counter,
            )
            _merge_metrics(job_metrics, episode_metrics)
            for family in ("players", "ball"):
                _merge_attribution(job_attribution[family], episode_attribution[family])
        _merge_metrics(global_metrics, job_metrics)
        _merge_metrics(scenario_metrics[scenario], job_metrics)
        for family in ("players", "ball"):
            _merge_attribution(global_attribution[family], job_attribution[family])
            _merge_attribution(scenario_attribution[scenario][family], job_attribution[family])
        by_job[job_id] = {
            "scenario": scenario,
            "path": str(job["path"]),
            "source_sha256": actual_source_sha256,
            "frame_count": observed_job_frames,
            "episode_count": len(episodes),
            "metrics": _summarize_metrics(job_metrics),
            "event_attribution": {
                family: _finalize_attribution(job_attribution[family])
                for family in ("players", "ball")
            },
        }

    if observed_match_ids != train_ids:
        missing = sorted(train_ids - observed_match_ids)
        extra = sorted(observed_match_ids - train_ids)
        raise ValueError(
            "Observed train episodes do not match split manifest: "
            f"{missing=}, {extra=}"
        )

    finalized_global_attribution = {
        family: _finalize_attribution(global_attribution[family])
        for family in ("players", "ball")
    }
    return {
        "status": "complete",
        "scope": "train_only_raw_grf",
        "inputs": {
            "collection_manifest_path": str(collection_path),
            "collection_manifest_sha256": _sha256(collection_path),
            "collection_plan_sha256": collection_plan_sha256,
            "split_manifest_path": str(split_path),
            "split_manifest_sha256": actual_split_sha256,
            "train_job_count": len(train_jobs),
            "train_episode_count": len(observed_match_ids),
            "train_frame_count": total_frames,
            "train_jobs": [str(job["id"]) for job in train_jobs],
            "held_out_jobs_read": [],
            "pff_sources_read": [],
        },
        "thresholds": {
            "event_window_frames_each_side": EVENT_WINDOW_FRAMES,
            "players": _thresholds("players"),
            "ball": _thresholds("ball"),
        },
        "global_metrics": _summarize_metrics(global_metrics),
        "global_event_attribution": finalized_global_attribution,
        "by_job": by_job,
        "by_scenario": {
            scenario: {
                "metrics": _summarize_metrics(metrics),
                "event_attribution": {
                    family: _finalize_attribution(scenario_attribution[scenario][family])
                    for family in ("players", "ball")
                },
            }
            for scenario, metrics in sorted(scenario_metrics.items())
        },
        "top_player_accelerations": [
            row for _, _, row in sorted(player_top, key=lambda item: item[:2], reverse=True)
        ],
        "top_ball_accelerations": [
            row for _, _, row in sorted(ball_top, key=lambda item: item[:2], reverse=True)
        ],
        "decision": _decision(finalized_global_attribution["players"]),
    }
