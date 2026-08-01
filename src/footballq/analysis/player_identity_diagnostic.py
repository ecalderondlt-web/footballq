"""Frozen same-player retrieval diagnostic for chronological PFF profiles."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from footballq.analysis.player_profile_proof import (
    MatchInfo,
    _canonical_manifest,
    _encode,
    _load_match_td,
    _profile_indices,
    _stable_hash,
    _td_entries,
    anchor_player_ids,
    build_match_catalog,
    lineup_lookup,
    load_frozen_encoder,
)
from footballq.io.pff_shards import file_sha256


def load_config(path: str | Path) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def raw_clip_features(state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Summarize each entity's observed one-second geometry without identity fields."""

    output = torch.zeros(state.shape[0], state.shape[2], 11, dtype=torch.float32)
    for clip_index in range(state.shape[0]):
        for entity_index in range(state.shape[2]):
            visible = mask[clip_index, :, entity_index]
            if not bool(visible.any()):
                continue
            xy = state[clip_index, visible, entity_index, :2].float()
            mean = xy.mean(dim=0)
            std = xy.std(dim=0, unbiased=False)
            minimum = xy.min(dim=0).values
            maximum = xy.max(dim=0).values
            displacement = xy[-1] - xy[0]
            path_length = (
                torch.linalg.vector_norm(xy[1:] - xy[:-1], dim=1).sum()
                if len(xy) > 1
                else torch.tensor(0.0)
            )
            output[clip_index, entity_index] = torch.cat(
                [mean, std, minimum, maximum, displacement, path_length.view(1)]
            )
    return output


def aggregate_match_profiles(
    entity_tokens: torch.Tensor,
    raw_features: torch.Tensor,
    masks: torch.Tensor,
    player_ids: list[list[str | None]],
    *,
    roles: dict[str, str],
    teams: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Aggregate clip features into one latent and raw profile per match-player."""

    latent_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    raw_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    for clip_index, ids in enumerate(player_ids):
        visible = masks[clip_index].any(dim=0)
        for entity_index, player_id in enumerate(ids):
            if entity_index == 0 or player_id is None or not bool(visible[entity_index]):
                continue
            player_id = str(player_id)
            latent_values[player_id].append(entity_tokens[clip_index, entity_index])
            raw_values[player_id].append(raw_features[clip_index, entity_index])

    profiles: dict[str, dict[str, Any]] = {}
    for player_id, values in latent_values.items():
        if player_id not in roles or player_id not in teams:
            continue
        profiles[player_id] = {
            "latent": torch.stack(values).mean(dim=0),
            "raw": torch.stack(raw_values[player_id]).mean(dim=0),
            "role": roles[player_id],
            "team": teams[player_id],
            "clips": len(values),
        }
    return profiles


def _player_teams(
    lineup_by_slot: dict[tuple[str, int], str],
    match: MatchInfo,
) -> dict[str, str]:
    return {
        str(player_id): (
            match.home_team_name if side == "home" else match.away_team_name
        )
        for (side, _jersey), player_id in lineup_by_slot.items()
    }


def build_profile_cache(
    config: dict[str, Any],
    *,
    workspace_root: str | Path,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(workspace_root)
    data_cfg = config["data"]
    pff_root = root / data_cfg["pff_raw_root"]
    canonical_root = root / data_cfg["canonical_root"]
    statsbomb_root = root / data_cfg["statsbomb_root"]
    manifest_paths = [root / value for value in data_cfg["td_manifests"]]
    pff_files = {
        path.parent.stem: path
        for path in pff_root.glob("*.jsonl/*.jsonl")
        if path.is_file()
    }
    catalog = build_match_catalog(pff_files, statsbomb_root)
    allowed_splits = set(data_cfg["included_splits"])
    selected = [match for match in catalog if match.split in allowed_splits]
    if any(match.split == "test" for match in selected):
        raise ValueError("Identity diagnostic must not load knockout/test embeddings.")

    td_entries, td_audit = _td_entries(manifest_paths)
    checkpoint_path = root / config["encoder"]["checkpoint"]
    encoder, torch_device, checkpoint_audit = load_frozen_encoder(
        checkpoint_path,
        config["encoder"]["sha256"],
        device=device,
    )
    profiles: dict[str, dict[str, dict[str, Any]]] = {}
    match_audit: dict[str, Any] = {}
    for match_number, match in enumerate(selected, start=1):
        print(
            f"[identity-cache {match_number:02d}/{len(selected):02d}] "
            f"PFF {match.pff_match_id} ({match.split})",
            flush=True,
        )
        td = _load_match_td(td_entries[match.pff_match_id])
        indices = _profile_indices(
            td,
            stride_examples=int(data_cfg["profile_stride_examples"]),
            min_visible_entities=int(data_cfg["min_visible_entities_at_anchor"]),
        )
        states = td["state"][indices]
        masks = td["mask"][indices]
        _global, entity_tokens = _encode(
            encoder,
            states,
            masks,
            device=torch_device,
            batch_size=int(config["encoder"]["batch_size"]),
        )
        canonical_path, canonical_manifest = _canonical_manifest(
            canonical_root,
            match.pff_match_id,
        )
        lineup_by_slot, roles, _names = lineup_lookup(statsbomb_root, match)
        teams = _player_teams(lineup_by_slot, match)
        frames = [int(td["context_end_frame"][index]) for index in indices]
        by_frame = anchor_player_ids(
            canonical_path,
            canonical_manifest,
            frames,
            lineup_by_slot,
        )
        player_ids = [by_frame[frame] for frame in frames]
        profiles[match.pff_match_id] = aggregate_match_profiles(
            entity_tokens,
            raw_clip_features(states, masks),
            masks,
            player_ids,
            roles=roles,
            teams=teams,
        )
        match_audit[match.pff_match_id] = {
            "split": match.split,
            "profile_clips": len(indices),
            "profile_players": len(profiles[match.pff_match_id]),
        }

    payload = {
        "version": 1,
        "experiment": config["experiment"],
        "profiles": profiles,
        "matches": [match.to_dict() for match in selected],
        "checkpoint": checkpoint_audit,
        "included_splits": sorted(allowed_splits),
    }
    audit = {
        "experiment": config["experiment"],
        "config_sha256": _stable_hash(config),
        "checkpoint": checkpoint_audit,
        "td_manifests": td_audit,
        "included_splits": sorted(allowed_splits),
        "processed_match_count": len(selected),
        "processed_match_ids": [match.pff_match_id for match in selected],
        "test_match_ids_loaded": [],
        "match_audit": match_audit,
    }
    return payload, audit


def _matches_from_payload(payload: dict[str, Any]) -> dict[str, MatchInfo]:
    return {
        str(row["pff_match_id"]): MatchInfo(
            pff_match_id=str(row["pff_match_id"]),
            statsbomb_match_id=str(row["statsbomb_match_id"]),
            match_datetime=datetime.fromisoformat(row["match_datetime"]),
            stage=str(row["stage"]),
            match_week=int(row["match_week"]),
            split=str(row["split"]),
            home_team_name=str(row["home_team_name"]),
            away_team_name=str(row["away_team_name"]),
        )
        for row in payload["matches"]
    }


def _fit_profile_normalizer(
    profiles: dict[str, dict[str, dict[str, Any]]],
    matches: dict[str, MatchInfo],
    *,
    key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = [
        profile[key].float()
        for match_id, match_profiles in profiles.items()
        if matches[match_id].split in {"support", "train"}
        for profile in match_profiles.values()
    ]
    stacked = torch.stack(values)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-3, torch.ones_like(std), std)
    return mean, std


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def _support_profiles(
    profiles: dict[str, dict[str, dict[str, Any]]],
    matches: dict[str, MatchInfo],
    *,
    player_id: str,
    query_datetime: datetime,
    key: str,
    k: int,
    normalizer: tuple[torch.Tensor, torch.Tensor] | None,
) -> list[torch.Tensor]:
    rows = [
        (matches[match_id], values[player_id][key].float())
        for match_id, values in profiles.items()
        if player_id in values
        and matches[match_id].match_datetime < query_datetime
    ]
    rows.sort(key=lambda row: (row[0].match_datetime, row[0].pff_match_id))
    selected = [value for _match, value in rows[-k:]]
    if normalizer is not None:
        mean, std = normalizer
        selected = [(value - mean) / std for value in selected]
    return selected


def retrieval_rows(
    profiles: dict[str, dict[str, dict[str, Any]]],
    matches: dict[str, MatchInfo],
    *,
    key: str,
    query_splits: set[str],
    support_size: int,
    normalizer: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    """Build same-team, same-role chronological retrieval rows."""

    rows: list[dict[str, Any]] = []
    for match_id, query_profiles in profiles.items():
        match = matches[match_id]
        if match.split not in query_splits:
            continue
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for player_id, profile in query_profiles.items():
            grouped[(str(profile["team"]), str(profile["role"]))].append(player_id)
        for (team, role), current_players in grouped.items():
            prototypes: dict[str, torch.Tensor] = {}
            support_counts: dict[str, int] = {}
            for player_id in current_players:
                support = _support_profiles(
                    profiles,
                    matches,
                    player_id=player_id,
                    query_datetime=match.match_datetime,
                    key=key,
                    k=support_size,
                    normalizer=normalizer,
                )
                if support:
                    prototypes[player_id] = torch.stack(support).mean(dim=0)
                    support_counts[player_id] = len(support)
            if len(prototypes) < 2:
                continue
            for player_id in sorted(prototypes):
                query = query_profiles[player_id][key].float()
                if normalizer is not None:
                    mean, std = normalizer
                    query = (query - mean) / std
                similarities = {
                    candidate_id: _cosine(query, prototype)
                    for candidate_id, prototype in prototypes.items()
                }
                positive = similarities[player_id]
                negatives = [
                    value
                    for candidate_id, value in similarities.items()
                    if candidate_id != player_id
                ]
                ordered = sorted(
                    similarities,
                    key=lambda candidate_id: (-similarities[candidate_id], candidate_id),
                )
                rank = ordered.index(player_id) + 1
                pairwise = float(
                    np.mean(
                        [
                            1.0 if positive > negative else 0.5 if positive == negative else 0.0
                            for negative in negatives
                        ]
                    )
                )
                rows.append(
                    {
                        "query_id": f"{match_id}:{player_id}",
                        "match_id": match_id,
                        "split": match.split,
                        "player_id": player_id,
                        "team": team,
                        "role": role,
                        "candidate_count": len(prototypes),
                        "support_count": support_counts[player_id],
                        "positive_similarity": positive,
                        "negative_similarity_mean": float(np.mean(negatives)),
                        "pairwise_accuracy": pairwise,
                        "rank": rank,
                    }
                )
    return rows


def retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "queries": 0,
            "top1_accuracy": None,
            "chance_top1": None,
            "mean_reciprocal_rank": None,
            "pairwise_accuracy": None,
            "mean_similarity_margin": None,
        }
    return {
        "queries": len(rows),
        "unique_players": len({row["player_id"] for row in rows}),
        "top1_accuracy": float(np.mean([row["rank"] == 1 for row in rows])),
        "chance_top1": float(np.mean([1.0 / row["candidate_count"] for row in rows])),
        "mean_reciprocal_rank": float(np.mean([1.0 / row["rank"] for row in rows])),
        "pairwise_accuracy": float(np.mean([row["pairwise_accuracy"] for row in rows])),
        "mean_similarity_margin": float(
            np.mean(
                [
                    row["positive_similarity"] - row["negative_similarity_mean"]
                    for row in rows
                ]
            )
        ),
        "candidate_count_mean": float(
            np.mean([row["candidate_count"] for row in rows])
        ),
        "support_count_mean": float(np.mean([row["support_count"] for row in rows])),
    }


def _randomized_profiles(
    profiles: dict[str, dict[str, dict[str, Any]]],
    *,
    seed: int,
    mode: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    output = {
        match_id: {
            player_id: {
                **profile,
                "latent": profile["latent"].clone(),
            }
            for player_id, profile in values.items()
        }
        for match_id, values in profiles.items()
    }
    generator = torch.Generator().manual_seed(seed)
    if mode == "random":
        for values in output.values():
            for profile in values.values():
                profile["latent"] = torch.randn(
                    profile["latent"].shape,
                    generator=generator,
                )
        return output
    if mode != "shuffle":
        raise ValueError(f"Unknown profile randomization mode: {mode}")
    for values in output.values():
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for player_id, profile in values.items():
            groups[(profile["team"], profile["role"])].append(player_id)
        for player_ids in groups.values():
            if len(player_ids) < 2:
                continue
            order = torch.randperm(len(player_ids), generator=generator).tolist()
            source = [values[player_id]["latent"].clone() for player_id in player_ids]
            for target_index, player_id in enumerate(player_ids):
                values[player_id]["latent"] = source[order[target_index]]
    return output


def paired_player_bootstrap(
    latent_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    raw_by_id = {row["query_id"]: row for row in raw_rows}
    paired = [
        (row, raw_by_id[row["query_id"]])
        for row in latent_rows
        if row["query_id"] in raw_by_id
    ]
    by_player: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for latent, raw in paired:
        by_player[latent["player_id"]].append((latent, raw))
    player_ids = sorted(by_player)
    rng = random.Random(seed)
    gains: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(player_ids) for _ in player_ids]
        pairs = [pair for player_id in selected for pair in by_player[player_id]]
        gains.append(
            float(
                np.mean([latent["pairwise_accuracy"] for latent, _raw in pairs])
                - np.mean([raw["pairwise_accuracy"] for _latent, raw in pairs])
            )
        )
    return {
        "paired_queries": len(paired),
        "players": len(player_ids),
        "samples": len(gains),
        "mean_pairwise_gain": float(np.mean(gains)),
        "positive_fraction": float(np.mean(np.asarray(gains) > 0.0)),
        "ci95": [
            float(np.quantile(gains, 0.025)),
            float(np.quantile(gains, 0.975)),
        ],
    }


def evaluate_profile_cache(
    payload: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    profiles = payload["profiles"]
    matches = _matches_from_payload(payload)
    if any(match.split == "test" for match in matches.values()):
        raise ValueError("Test profiles are present in the identity diagnostic cache.")
    raw_normalizer = _fit_profile_normalizer(profiles, matches, key="raw")
    latent_normalizer = _fit_profile_normalizer(profiles, matches, key="latent")
    random_profiles = _randomized_profiles(
        profiles,
        seed=int(config["controls"]["random_seed"]),
        mode="random",
    )
    shuffled_profiles = _randomized_profiles(
        profiles,
        seed=int(config["controls"]["shuffle_seed"]),
        mode="shuffle",
    )
    random_normalizer = _fit_profile_normalizer(
        random_profiles,
        matches,
        key="latent",
    )
    main_support_size = int(config["evaluation"]["support_size"])
    support_sizes = sorted({1, main_support_size})
    model_specs = (
        ("raw_kinematics", profiles, "raw", raw_normalizer),
        ("frozen_td_jepa", profiles, "latent", latent_normalizer),
        ("random_profile", random_profiles, "latent", random_normalizer),
        (
            "same_team_role_identity_shuffle",
            shuffled_profiles,
            "latent",
            latent_normalizer,
        ),
    )
    rows_by_support: dict[int, dict[str, list[dict[str, Any]]]] = {}
    metrics_by_support: dict[int, dict[str, Any]] = {}
    for support_size in support_sizes:
        rows_by_model: dict[str, list[dict[str, Any]]] = {}
        for model_name, model_profiles, key, feature_normalizer in model_specs:
            rows_by_model[model_name] = retrieval_rows(
                model_profiles,
                matches,
                key=key,
                query_splits={"train", "val"},
                support_size=support_size,
                normalizer=feature_normalizer,
            )
        rows_by_support[support_size] = rows_by_model
        metrics_by_support[support_size] = {
            model: {
                split: retrieval_metrics(
                    [row for row in rows if row["split"] == split]
                )
                for split in ("train", "val")
            }
            for model, rows in rows_by_model.items()
        }
    rows_by_model = rows_by_support[main_support_size]
    metrics = metrics_by_support[main_support_size]
    bootstrap_by_support = {
        support_size: paired_player_bootstrap(
            [
                row
                for row in support_rows["frozen_td_jepa"]
                if row["split"] == "val"
            ],
            [
                row
                for row in support_rows["raw_kinematics"]
                if row["split"] == "val"
            ],
            samples=int(config["evaluation"]["player_bootstrap_samples"]),
            seed=int(config["evaluation"]["bootstrap_seed"]) + support_size,
        )
        for support_size, support_rows in rows_by_support.items()
    }
    bootstrap = bootstrap_by_support[main_support_size]
    val = {model: values["val"] for model, values in metrics.items()}
    latent_pairwise_gain = (
        val["frozen_td_jepa"]["pairwise_accuracy"]
        - val["raw_kinematics"]["pairwise_accuracy"]
    )
    latent_top1_gain = (
        val["frozen_td_jepa"]["top1_accuracy"]
        - val["raw_kinematics"]["top1_accuracy"]
    )
    shuffled_pairwise_gain = (
        val["frozen_td_jepa"]["pairwise_accuracy"]
        - val["same_team_role_identity_shuffle"]["pairwise_accuracy"]
    )
    thresholds = config["gates"]
    gates = {
        "val_pairwise_gain_over_raw": latent_pairwise_gain,
        "val_top1_gain_over_raw": latent_top1_gain,
        "val_pairwise_gain_over_identity_shuffle": shuffled_pairwise_gain,
        "val_top1_gain_over_chance": (
            val["frozen_td_jepa"]["top1_accuracy"]
            - val["frozen_td_jepa"]["chance_top1"]
        ),
        "player_bootstrap_ci95": bootstrap["ci95"],
    }
    gates["passed"] = bool(
        latent_pairwise_gain >= float(thresholds["minimum_pairwise_gain_over_raw"])
        and latent_top1_gain >= float(thresholds["minimum_top1_gain_over_raw"])
        and shuffled_pairwise_gain
        >= float(thresholds["minimum_pairwise_gain_over_shuffle"])
        and gates["val_top1_gain_over_chance"]
        >= float(thresholds["minimum_top1_gain_over_chance"])
        and bootstrap["ci95"][0] > 0.0
    )
    return {
        "experiment": config["experiment"],
        "scope": "support_train_validation_only_no_knockout_profiles",
        "metrics": metrics,
        "support_size_sensitivity": {
            str(k): values for k, values in metrics_by_support.items()
        },
        "main_support_size": main_support_size,
        "player_bootstrap_frozen_minus_raw": bootstrap,
        "player_bootstrap_by_support_size": {
            str(k): value for k, value in bootstrap_by_support.items()
        },
        "decision_gate": gates,
        "row_counts": {
            model: len(rows) for model, rows in rows_by_model.items()
        },
        "interpretation_limit": (
            "This diagnostic tests match-level player retrieval under same-team, "
            "same-role controls. It does not test event prediction, tactical "
            "planning, or cross-club identity transfer."
        ),
    }


def run_diagnostic(
    config_path: str | Path,
    *,
    workspace_root: str | Path,
    device: str = "auto",
    rebuild_cache: bool = False,
) -> dict[str, Path]:
    root = Path(workspace_root)
    config_path = Path(config_path)
    config = load_config(config_path)
    run_dir = root / config["output"]["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "profile_cache.pt"
    audit_path = run_dir / "profile_audit.json"
    result_path = run_dir / "results.json"
    if rebuild_cache or not cache_path.exists():
        payload, audit = build_profile_cache(
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
            raise ValueError("Cached identity profiles were built under another config.")
    result = evaluate_profile_cache(payload, config)
    result["config_path"] = str(config_path)
    result["config_file_sha256"] = file_sha256(config_path)
    result["experiment_code_sha256"] = file_sha256(Path(__file__))
    result["profile_cache_path"] = str(cache_path)
    result["profile_cache_sha256"] = file_sha256(cache_path)
    result["profile_audit_path"] = str(audit_path)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {
        "run_dir": run_dir,
        "profile_cache": cache_path,
        "profile_audit": audit_path,
        "results": result_path,
    }
