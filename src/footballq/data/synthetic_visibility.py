"""PFF-train-calibrated visibility profiles for synthetic tracking."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from footballq.repro.splits import stable_json_bytes

FRAME_KEYS = ["match_id", "period", "frame_id"]
DEFAULT_DISTANCE_BINS_M = [0.0, 10.0, 20.0, 30.0, 40.0, 55.0, 75.0, 110.0]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_bytes(payload)).hexdigest()


def build_pff_visibility_profile(
    canonical_root: str | Path,
    *,
    split: str = "train",
    frame_stride: int = 10,
    distance_bins_m: list[float] | None = None,
) -> dict[str, Any]:
    """Estimate observed-player and ball visibility from canonical PFF shards."""

    root = Path(canonical_root)
    dataset_manifest_path = root / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    bins = np.asarray(distance_bins_m or DEFAULT_DISTANCE_BINS_M, dtype=float)
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least one.")
    if len(bins) < 2 or not np.all(np.diff(bins) > 0):
        raise ValueError("distance bins must be strictly increasing.")

    matches = [item for item in dataset_manifest["matches"] if item["split"] == split]
    if not matches:
        raise ValueError(f"Canonical PFF manifest has no {split!r} matches.")

    count_histogram: Counter[int] = Counter()
    distance_total = np.zeros(len(bins) - 1, dtype=np.int64)
    distance_observed = np.zeros(len(bins) - 1, dtype=np.int64)
    ball_frames = 0
    ball_observed = 0
    player_rows = 0
    player_observed = 0

    for match in matches:
        match_id = str(match["match_id"])
        match_root = root / split / match_id
        match_manifest = json.loads(
            (match_root / "manifest.json").read_text(encoding="utf-8")
        )
        for shard in match_manifest["shards"]:
            frame = pd.read_parquet(
                match_root / shard["path"],
                columns=[*FRAME_KEYS, "agent_type", "x_m", "y_m", "is_observed"],
            )
            frame = frame[frame["frame_id"].astype(int) % int(frame_stride) == 0].copy()
            if frame.empty:
                continue
            frame["observed_xy"] = (
                frame["is_observed"].fillna(False).astype(bool)
                & frame["x_m"].notna()
                & frame["y_m"].notna()
            )
            players = frame[frame["agent_type"] == "player"].copy()
            balls = frame[frame["agent_type"] == "ball"].copy()

            counts = players.groupby(FRAME_KEYS, sort=False)["observed_xy"].sum().astype(int)
            count_histogram.update(counts.tolist())
            player_rows += len(players)
            player_observed += int(players["observed_xy"].sum())

            balls = balls.drop_duplicates(FRAME_KEYS)
            ball_frames += len(balls)
            ball_observed += int(balls["observed_xy"].sum())
            ball_positions = balls[[*FRAME_KEYS, "x_m", "y_m"]].rename(
                columns={"x_m": "ball_x_m", "y_m": "ball_y_m"}
            )
            distances = players.merge(ball_positions, on=FRAME_KEYS, how="inner")
            valid = distances[
                distances[["x_m", "y_m", "ball_x_m", "ball_y_m"]].notna().all(axis=1)
            ]
            if valid.empty:
                continue
            distance_m = np.hypot(
                valid["x_m"].to_numpy() - valid["ball_x_m"].to_numpy(),
                valid["y_m"].to_numpy() - valid["ball_y_m"].to_numpy(),
            )
            distance_total += np.histogram(distance_m, bins=bins)[0]
            distance_observed += np.histogram(
                distance_m,
                bins=bins,
                weights=valid["observed_xy"].astype(np.int64).to_numpy(),
            )[0].astype(np.int64)

    sampled_frames = sum(count_histogram.values())
    if sampled_frames == 0:
        raise ValueError("No PFF frames were available for visibility calibration.")
    global_player_rate = player_observed / player_rows if player_rows else 0.0
    distance_rates = [
        float(observed / total) if total else float(global_player_rate)
        for observed, total in zip(distance_observed, distance_total, strict=True)
    ]
    payload: dict[str, Any] = {
        "version": 1,
        "profile": "pff_train_observed_visibility_v1",
        "dataset": "pff_fc",
        "split": split,
        "frame_stride": int(frame_stride),
        "sampled_frame_count": sampled_frames,
        "train_match_ids": [str(item["match_id"]) for item in matches],
        "canonical_dataset_manifest_path": str(dataset_manifest_path),
        "canonical_dataset_manifest_file_sha256": _file_sha256(dataset_manifest_path),
        "canonical_dataset_manifest_payload_sha256": dataset_manifest.get(
            "manifest_payload_sha256"
        ),
        "split_manifest_path": dataset_manifest.get("split_manifest_path"),
        "split_manifest_sha256": dataset_manifest.get("split_manifest_sha256"),
        "observed_player_count_counts": {
            str(count): int(count_histogram[count]) for count in sorted(count_histogram)
        },
        "observed_player_count_probabilities": {
            str(count): count_histogram[count] / sampled_frames for count in sorted(count_histogram)
        },
        "player_observed_rate": float(global_player_rate),
        "ball_observed_rate": float(ball_observed / ball_frames) if ball_frames else 0.0,
        "distance_bin_edges_m": bins.tolist(),
        "distance_bin_player_counts": distance_total.tolist(),
        "distance_bin_observed_counts": distance_observed.tolist(),
        "distance_bin_observed_probabilities": distance_rates,
    }
    payload["profile_payload_sha256"] = _profile_hash(payload)
    return payload


def write_visibility_profile(profile: dict[str, Any], out: str | Path) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return path


def _frame_rng(seed: int, key: tuple[Any, ...]) -> np.random.Generator:
    material = "|".join([str(seed), *(str(value) for value in key)]).encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(stable_seed)


def _distance_probabilities(
    distance_m: np.ndarray,
    profile: dict[str, Any],
) -> np.ndarray:
    edges = np.asarray(profile["distance_bin_edges_m"], dtype=float)
    probabilities = np.asarray(profile["distance_bin_observed_probabilities"], dtype=float)
    indices = np.searchsorted(edges, distance_m, side="right") - 1
    indices = np.clip(indices, 0, len(probabilities) - 1)
    return np.clip(probabilities[indices], 1e-4, 1.0 - 1e-4)


def apply_pff_like_visibility(
    tracking: pd.DataFrame,
    profile: dict[str, Any],
    *,
    seed: int,
) -> pd.DataFrame:
    """Mask synthetic entities with deterministic PFF-train-calibrated visibility."""

    out = tracking.copy().reset_index(drop=True)
    counts = np.asarray(
        [int(value) for value in profile["observed_player_count_probabilities"]], dtype=int
    )
    count_probabilities = np.asarray(
        list(profile["observed_player_count_probabilities"].values()), dtype=float
    )
    count_probabilities /= count_probabilities.sum()
    ball_rate = float(profile["ball_observed_rate"])
    visible_column = "visible" if "visible" in out.columns else "is_visible"
    agent_type = out["agent_type"].astype(str).to_numpy()
    xy = out[["x_m", "y_m"]].to_numpy(dtype=float)
    available_visibility = out[visible_column].fillna(False).astype(bool).to_numpy()
    visibility_columns = {}
    for column in ("visible", "is_visible", "is_observed"):
        if column in out.columns:
            visibility_columns[column] = (
                out[column].fillna(False).astype(bool).to_numpy(copy=True)
            )
        else:
            visibility_columns[column] = np.zeros(len(out), dtype=bool)
    provider_visibility = (
        out["provider_visibility"].astype(object).to_numpy(copy=True)
        if "provider_visibility" in out.columns
        else None
    )

    for key, indices in out.groupby(FRAME_KEYS, sort=False).groups.items():
        indices = np.asarray(indices, dtype=int)
        rng = _frame_rng(seed, key if isinstance(key, tuple) else (key,))
        player_indices = indices[agent_type[indices] == "player"]
        ball_indices = indices[agent_type[indices] == "ball"]
        available = available_visibility[player_indices]
        finite = np.isfinite(xy[player_indices]).all(axis=1)
        eligible_indices = player_indices[available & finite]
        target_count = min(int(rng.choice(counts, p=count_probabilities)), len(eligible_indices))

        selected = np.asarray([], dtype=int)
        if target_count and len(eligible_indices):
            positions = xy[eligible_indices]
            if len(ball_indices):
                ball_xy = xy[ball_indices[0]]
            else:
                ball_xy = np.asarray([np.nan, np.nan])
            if np.isfinite(ball_xy).all():
                distance_m = np.linalg.norm(positions - ball_xy, axis=1)
                probability = _distance_probabilities(distance_m, profile)
            else:
                probability = np.full(len(eligible_indices), profile["player_observed_rate"])
            logits = np.log(probability / (1.0 - probability))
            scores = logits + rng.gumbel(size=len(eligible_indices))
            selected = eligible_indices[np.argpartition(scores, -target_count)[-target_count:]]

        for values in visibility_columns.values():
            values[player_indices] = False
        if len(selected):
            for values in visibility_columns.values():
                values[selected] = True
        if provider_visibility is not None:
            provider_visibility[player_indices] = "synthetic_masked"
            provider_visibility[selected] = "synthetic_observed"

        if len(ball_indices):
            ball_visible = bool(rng.random() < ball_rate)
            for values in visibility_columns.values():
                values[ball_indices] = ball_visible
            if provider_visibility is not None:
                provider_visibility[ball_indices] = (
                    "synthetic_observed" if ball_visible else "synthetic_masked"
                )
    for column, values in visibility_columns.items():
        out[column] = values
    if provider_visibility is not None:
        out["provider_visibility"] = provider_visibility
    return out
