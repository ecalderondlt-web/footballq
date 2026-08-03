"""Sealed-test evaluation and series-blocked inference for RLCS matchups."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from footballq.models.identity_matchup_transformer import (
    IDENTITY_CONDITIONS,
    IdentityMatchupTransformer,
    permute_within_roster_identities,
)
from footballq.repro.manifest import file_sha256
from footballq.training.train import resolve_device
from footballq.training.train_matchup import (
    RLCSDecisionDataset,
    _model_from_config,
    _model_inputs,
    _to_device,
)


class TestUnlockError(PermissionError):
    """Raised if the one-time sealed-test authorization is absent or inconsistent."""


def validate_test_unlock(
    path: str | Path,
    *,
    dataset_manifest_path: str | Path,
    split_manifest_path: str | Path,
) -> dict[str, Any]:
    """Validate the frozen evaluation bundle before any test Parquet read."""

    unlock = json.loads(Path(path).read_text(encoding="utf-8"))
    if unlock.get("protocol") != "rlcs_identity_matchup_v1_sealed_test":
        raise TestUnlockError("Unlock protocol is missing or incorrect.")
    if unlock.get("status") != "unlocked_after_validation_gate":
        raise TestUnlockError("Test unlock has not passed the validation gate.")
    if unlock.get("dataset_manifest_sha256") != file_sha256(dataset_manifest_path):
        raise TestUnlockError("Dataset manifest changed after test authorization.")
    if unlock.get("split_manifest_sha256") != file_sha256(split_manifest_path):
        raise TestUnlockError("Split manifest changed after test authorization.")
    checkpoints = unlock.get("checkpoints", {})
    expected_seeds = {"17", "23", "41"}
    if set(checkpoints) != set(IDENTITY_CONDITIONS):
        raise TestUnlockError("Unlock must freeze all four model conditions.")
    for condition, seeds in checkpoints.items():
        if set(seeds) != expected_seeds:
            raise TestUnlockError(f"Unlock condition {condition} does not contain seeds 17/23/41.")
        for seed, descriptor in seeds.items():
            checkpoint_path = Path(descriptor["path"])
            if not checkpoint_path.exists():
                raise TestUnlockError(f"Missing frozen checkpoint: {checkpoint_path}")
            if file_sha256(checkpoint_path) != descriptor["sha256"]:
                raise TestUnlockError(f"Checkpoint hash mismatch for {condition}/{seed}.")
    if not str(unlock.get("nonce") or ""):
        raise TestUnlockError("Unlock requires a unique nonce.")
    return unlock


def consume_test_unlock(
    unlock_path: str | Path, *, output_dir: str | Path, resume: bool = False
) -> Path:
    """Create an exclusive receipt before opening test labels."""

    unlock = Path(unlock_path)
    receipt = unlock.with_suffix(unlock.suffix + ".consumed.json")
    payload = {
        "unlock_sha256": file_sha256(unlock),
        "output_dir": str(Path(output_dir).resolve()),
        "consumed_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    try:
        with receipt.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        old = json.loads(receipt.read_text(encoding="utf-8"))
        compatible = (
            old.get("unlock_sha256") == payload["unlock_sha256"]
            and old.get("output_dir") == payload["output_dir"]
        )
        if not resume or not compatible:
            raise TestUnlockError(
                f"Test unlock was already consumed; receipt exists at {receipt}."
            ) from exc
    return receipt


def _load_model(
    checkpoint_path: str | Path, *, device: torch.device
) -> tuple[IdentityMatchupTransformer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = _model_from_config(checkpoint["config"], int(checkpoint["vocabulary_size"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def _validate_checkpoint_bundle(
    descriptors: Mapping[str, Mapping[str, Mapping[str, str]]],
    *,
    dataset_manifest_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Require all twelve models to share data lineage and train-only normalization."""

    reference_mean: np.ndarray | None = None
    reference_std: np.ndarray | None = None
    for condition, seeds in descriptors.items():
        for seed, descriptor in seeds.items():
            checkpoint = torch.load(descriptor["path"], map_location="cpu", weights_only=False)
            if checkpoint.get("condition") != condition or int(checkpoint.get("seed", -1)) != int(
                seed
            ):
                raise ValueError(f"Checkpoint condition/seed mismatch: {condition}/{seed}.")
            if checkpoint.get("dataset_manifest_sha256") != dataset_manifest_sha256:
                raise ValueError(f"Checkpoint dataset lineage mismatch: {condition}/{seed}.")
            mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
            std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
            if reference_mean is None:
                reference_mean, reference_std = mean, std
            elif not np.array_equal(mean, reference_mean) or not np.array_equal(std, reference_std):
                raise ValueError("Matched checkpoints do not share identical train statistics.")
    if reference_mean is None or reference_std is None:
        raise ValueError("Checkpoint bundle is empty.")
    return reference_mean, reference_std


def _stage_bucket(group_path: str) -> str:
    lowered = str(group_path).casefold()
    for name in ("qualifier", "swiss", "group", "playoff", "quarterfinal", "semifinal", "final"):
        if name in lowered:
            return name
    return "other"


def matched_opponent_roster_map(
    dataset: RLCSDecisionDataset, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Choose another-series opponent identities in frozen matching strata."""

    strata: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        timestamp = pd.Timestamp(dataset.event_times[index])
        date_band = int(timestamp.to_julian_date() // 14) if not pd.isna(timestamp) else -1
        clock_band = int(float(dataset.seconds_remaining[index]) // 30)
        key = (
            str(dataset.regions[index]),
            _stage_bucket(str(dataset.group_paths[index])),
            date_band,
            int(dataset.score_diff[index]),
            clock_band,
        )
        strata[key].append(index)
    rng = np.random.default_rng(int(seed))
    replacement = np.zeros((len(dataset), 3), dtype=np.int64)
    matched = np.zeros(len(dataset), dtype=np.bool_)
    for indices in strata.values():
        by_series: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            by_series[str(dataset.series_ids[index])].append(index)
        series_names = sorted(by_series)
        if len(series_names) < 2:
            continue
        for index in indices:
            candidates = [
                value
                for series in series_names
                if series != str(dataset.series_ids[index])
                for value in by_series[series]
            ]
            selected = int(rng.choice(candidates))
            replacement[index] = dataset.identity_indices[selected, 3:6]
            matched[index] = True
    return replacement, matched


def _predict_probabilities(
    model: IdentityMatchupTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    condition: str,
    precision: str,
    identity_transform: Callable[[torch.Tensor, int, int], torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    entity_probabilities: list[np.ndarray] = []
    zone_probabilities: list[np.ndarray] = []
    offset = 0
    autocast = precision == "bf16" and device.type == "cuda"
    with torch.no_grad():
        for raw_batch in loader:
            batch_size = int(raw_batch["state"].shape[0])
            batch = _to_device(raw_batch, device)
            if identity_transform is not None:
                batch["identity_indices"] = identity_transform(
                    batch["identity_indices"], offset, offset + batch_size
                )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                output = model(**_model_inputs(batch), condition=condition)
            entity_probabilities.append(
                torch.softmax(output["next_touch_entity_logits"].float(), dim=-1).cpu().numpy()
            )
            zone_probabilities.append(
                torch.softmax(output["next_touch_zone_logits"].float(), dim=-1).cpu().numpy()
            )
            offset += batch_size
    return np.concatenate(entity_probabilities), np.concatenate(zone_probabilities)


def _ensemble_condition(
    descriptors: Mapping[str, Mapping[str, str]],
    loader: DataLoader,
    device: torch.device,
    *,
    condition: str,
    precision: str,
    identity_transform_factory: Callable[[int], Callable[[torch.Tensor, int, int], torch.Tensor]]
    | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    entity: list[np.ndarray] = []
    zone: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    individual: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for seed_text, descriptor in sorted(descriptors.items(), key=lambda item: int(item[0])):
        model, checkpoint = _load_model(descriptor["path"], device=device)
        if checkpoint["condition"] != condition or int(checkpoint["seed"]) != int(seed_text):
            raise ValueError(f"Checkpoint metadata mismatch for {condition}/{seed_text}.")
        transform = (
            identity_transform_factory(int(seed_text))
            if identity_transform_factory is not None
            else None
        )
        entity_prob, zone_prob = _predict_probabilities(
            model,
            loader,
            device,
            condition=condition,
            precision=precision,
            identity_transform=transform,
        )
        entity.append(entity_prob)
        zone.append(zone_prob)
        individual[str(seed_text)] = (entity_prob, zone_prob)
        metadata.append(
            {
                "seed": int(seed_text),
                "checkpoint": descriptor["path"],
                "checkpoint_sha256": descriptor["sha256"],
                "validation": checkpoint.get("validation"),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.mean(entity, axis=0), np.mean(zone, axis=0), metadata, individual


def _nll(entity_probability: np.ndarray, zone_probability: np.ndarray, dataset: Any) -> np.ndarray:
    index = np.arange(len(dataset))
    entity = np.clip(entity_probability[index, dataset.next_entity], 1e-9, 1.0)
    zone = np.clip(zone_probability[index, dataset.next_zone], 1e-9, 1.0)
    return -np.log(entity) - np.log(zone)


def sign_flip_pvalue(values: Sequence[float], *, permutations: int, seed: int) -> float:
    """One-sided paired sign-flip test on official-series mean differences."""

    differences = np.asarray(values, dtype=np.float64)
    if len(differences) == 0:
        raise ValueError("Sign-flip test requires at least one series.")
    observed = float(differences.mean())
    rng = np.random.default_rng(int(seed))
    exceed = 0
    remaining = int(permutations)
    while remaining:
        block = min(remaining, 2048)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(block, len(differences)))
        exceed += int((np.mean(signs * differences, axis=1) >= observed).sum())
        remaining -= block
    return (exceed + 1.0) / (int(permutations) + 1.0)


def bca_relative_lift_interval(
    comparison: Sequence[float],
    full: Sequence[float],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """BCa interval with official series as the resampling unit."""

    normal = NormalDist()
    comparison_array = np.asarray(comparison, dtype=np.float64)
    full_array = np.asarray(full, dtype=np.float64)
    if comparison_array.shape != full_array.shape or len(comparison_array) < 3:
        raise ValueError("BCa interval requires at least three paired series.")

    def statistic(indices: np.ndarray) -> float:
        baseline = float(comparison_array[indices].mean())
        return (baseline - float(full_array[indices].mean())) / baseline

    all_indices = np.arange(len(comparison_array))
    observed = statistic(all_indices)
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(resamples), dtype=np.float64)
    for start in range(0, int(resamples), 1024):
        count = min(1024, int(resamples) - start)
        indices = rng.integers(0, len(all_indices), size=(count, len(all_indices)))
        comp_mean = comparison_array[indices].mean(axis=1)
        full_mean = full_array[indices].mean(axis=1)
        bootstrap[start : start + count] = (comp_mean - full_mean) / comp_mean
    probability = np.clip(np.mean(bootstrap < observed), 1e-6, 1.0 - 1e-6)
    bias = float(normal.inv_cdf(probability))
    jackknife = np.asarray(
        [statistic(np.delete(all_indices, index)) for index in all_indices], dtype=np.float64
    )
    centered = jackknife.mean() - jackknife
    denominator = 6.0 * float(np.sum(centered**2) ** 1.5)
    acceleration = float(np.sum(centered**3) / denominator) if denominator > 0 else 0.0
    alpha = (1.0 - float(confidence)) / 2.0
    adjusted: list[float] = []
    for probability_alpha in (alpha, 1.0 - alpha):
        z_alpha = float(normal.inv_cdf(probability_alpha))
        adjusted.append(
            float(
                normal.cdf(
                    bias + (bias + z_alpha) / (1.0 - acceleration * (bias + z_alpha))
                )
            )
        )
    lower, upper = np.quantile(bootstrap, np.clip(adjusted, 0.0, 1.0))
    return float(lower), float(upper)


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjust the three preregistered main comparisons."""

    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def series_comparison(
    frame: pd.DataFrame,
    *,
    comparison_column: str,
    full_column: str = "nll_full",
    permutations: int,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    paired = frame[["series_id", comparison_column, full_column]].dropna()
    grouped = paired.groupby("series_id", sort=True)[[comparison_column, full_column]].mean()
    comparison = grouped[comparison_column].to_numpy()
    full = grouped[full_column].to_numpy()
    difference = comparison - full
    relative = (float(comparison.mean()) - float(full.mean())) / float(comparison.mean())
    lower, upper = bca_relative_lift_interval(
        comparison,
        full,
        resamples=resamples,
        seed=seed,
    )
    return {
        "series_count": int(len(grouped)),
        "comparison_nll": float(comparison.mean()),
        "full_nll": float(full.mean()),
        "relative_nll_reduction": relative,
        "bca_95pct": [lower, upper],
        "one_sided_sign_flip_p": sign_flip_pvalue(
            difference, permutations=permutations, seed=seed
        ),
    }


def _recurring_matchup_mask(dataset: RLCSDecisionDataset) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Repeated-matchup evaluation requires pyarrow.") from exc
    train_path = Path(dataset.manifest["splits"]["train"]["path"])
    train = pq.read_table(train_path, columns=["team_roster_hash", "opponent_roster_hash"])

    def key(first: Any, second: Any) -> tuple[bytes, bytes]:
        return tuple(sorted((bytes(first), bytes(second))))  # type: ignore[return-value]

    train_keys = {
        key(first, second)
        for first, second in zip(
            train["team_roster_hash"].to_pylist(),
            train["opponent_roster_hash"].to_pylist(),
            strict=True,
        )
    }
    return np.asarray(
        [
            key(first, second) in train_keys
            for first, second in zip(
                dataset.team_roster_hashes,
                dataset.opponent_roster_hashes,
                strict=True,
            )
        ],
        dtype=np.bool_,
    )


def evaluate_sealed_test(
    config_path: str | Path,
    *,
    unlock_path: str | Path,
    output_dir: str | Path,
    resume: bool = False,
) -> dict[str, Path]:
    """Perform the one preregistered test evaluation and write machine-readable evidence."""

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    evaluation_cfg = cfg["evaluation"]
    dataset_manifest_path = Path(data_cfg["manifest"])
    unlock = validate_test_unlock(
        unlock_path,
        dataset_manifest_path=dataset_manifest_path,
        split_manifest_path=data_cfg["split_manifest"],
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    receipt = consume_test_unlock(unlock_path, output_dir=output, resume=resume)
    descriptors = unlock["checkpoints"]
    mean, std = _validate_checkpoint_bundle(
        descriptors,
        dataset_manifest_sha256=file_sha256(dataset_manifest_path),
    )
    dataset = RLCSDecisionDataset(
        dataset_manifest_path,
        "test",
        allow_test=True,
        feature_mean=mean,
        feature_std=std,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 256)),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    device = resolve_device(str(training_cfg.get("device", "auto")))
    precision = str(training_cfg.get("precision", "bf16"))
    nll_by_name: dict[str, np.ndarray] = {}
    seed_nll_by_condition: dict[str, dict[str, np.ndarray]] = {}
    checkpoint_evidence: dict[str, Any] = {}
    for condition in IDENTITY_CONDITIONS:
        entity, zone, evidence, individual = _ensemble_condition(
            descriptors[condition],
            loader,
            device,
            condition=condition,
            precision=precision,
        )
        nll_by_name[condition] = _nll(entity, zone, dataset)
        seed_nll_by_condition[condition] = {
            seed: _nll(probabilities[0], probabilities[1], dataset)
            for seed, probabilities in individual.items()
        }
        checkpoint_evidence[condition] = evidence

    permutation_seeds = int(evaluation_cfg.get("permutation_seeds", 20))
    within_nlls: list[np.ndarray] = []
    opponent_nlls: list[np.ndarray] = []
    opponent_match_masks: list[np.ndarray] = []
    for control_seed in range(permutation_seeds):

        def within_factory(_model_seed: int) -> Callable[[torch.Tensor, int, int], torch.Tensor]:
            generator = torch.Generator(device="cpu").manual_seed(100_000 + control_seed)

            def transform(values: torch.Tensor, _start: int, _end: int) -> torch.Tensor:
                return permute_within_roster_identities(values, generator=generator)

            return transform

        entity, zone, _, _ = _ensemble_condition(
            descriptors["full"],
            loader,
            device,
            condition="full",
            precision=precision,
            identity_transform_factory=within_factory,
        )
        within_nlls.append(_nll(entity, zone, dataset))
        replacement, matched = matched_opponent_roster_map(dataset, seed=control_seed)

        def opponent_factory(_model_seed: int) -> Callable[[torch.Tensor, int, int], torch.Tensor]:
            def transform(values: torch.Tensor, start: int, end: int) -> torch.Tensor:
                output_values = values.clone()
                replacement_tensor = torch.from_numpy(replacement[start:end]).to(values.device)
                matched_tensor = torch.from_numpy(matched[start:end]).to(values.device)
                output_values[matched_tensor, 3:6] = replacement_tensor[matched_tensor]
                return output_values

            return transform

        entity, zone, _, _ = _ensemble_condition(
            descriptors["full"],
            loader,
            device,
            condition="full",
            precision=precision,
            identity_transform_factory=opponent_factory,
        )
        opponent_nlls.append(_nll(entity, zone, dataset))
        opponent_match_masks.append(matched)

    nll_by_name["within_roster_shuffle"] = np.mean(within_nlls, axis=0)
    opponent_stack = np.stack(opponent_nlls)
    opponent_mask = np.stack(opponent_match_masks)
    opponent_stack[~opponent_mask] = np.nan
    nll_by_name["matched_opponent_shuffle"] = np.nanmean(opponent_stack, axis=0)
    critical = (np.abs(dataset.score_diff) <= 1) & (
        (dataset.seconds_remaining <= 120.0) | dataset.overtime
    )
    all_known = dataset.known_masks.all(axis=1)
    primary = critical & all_known
    recurring = _recurring_matchup_mask(dataset)
    sample_frame = pd.DataFrame(
        {
            "sample_id": dataset.sample_ids,
            "replay_id": dataset.replay_ids,
            "series_id": dataset.series_ids,
            "region": dataset.regions,
            "critical": critical,
            "all_identities_known": all_known,
            "primary": primary,
            "recurring_matchup": recurring,
            **{f"nll_{name}": values for name, values in nll_by_name.items()},
        }
    )
    primary_frame = sample_frame.loc[primary].copy()
    if primary_frame.empty:
        raise ValueError("Primary critical all-identities-known test subset is empty.")
    permutations = int(evaluation_cfg.get("sign_flip_permutations", 10_000))
    resamples = int(evaluation_cfg.get("bootstrap_resamples", 10_000))
    comparisons: dict[str, Any] = {}
    for comparison in (
        "anonymous",
        "actor_only",
        "roster_only",
        "within_roster_shuffle",
        "matched_opponent_shuffle",
    ):
        comparisons[f"full_vs_{comparison}"] = series_comparison(
            primary_frame,
            comparison_column=f"nll_{comparison}",
            permutations=permutations,
            resamples=resamples,
            seed=73,
        )
    main_p = {
        name: comparisons[name]["one_sided_sign_flip_p"]
        for name in ("full_vs_anonymous", "full_vs_actor_only", "full_vs_roster_only")
    }
    adjusted = holm_adjust(main_p)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value
    repeated_frame = primary_frame.loc[primary_frame["recurring_matchup"]]
    repeated_lift = (
        float((repeated_frame["nll_anonymous"] - repeated_frame["nll_full"]).mean())
        if not repeated_frame.empty
        else None
    )
    gates_cfg = evaluation_cfg["gates"]
    seed_lifts = {}
    for seed in ("17", "23", "41"):
        anonymous_seed = seed_nll_by_condition["anonymous"][seed][primary]
        full_seed = seed_nll_by_condition["full"][seed][primary]
        seed_lifts[seed] = float(
            (anonymous_seed.mean() - full_seed.mean()) / anonymous_seed.mean()
        )
    series_differences = primary_frame.groupby("series_id", sort=True)[
        ["nll_anonymous", "nll_full"]
    ].mean()
    leave_one_series_out = []
    for series_id in series_differences.index:
        remainder = series_differences.drop(index=series_id)
        baseline = float(remainder["nll_anonymous"].mean())
        leave_one_series_out.append(
            (baseline - float(remainder["nll_full"].mean())) / baseline
        )
    known_coverage = float(all_known[critical].mean()) if bool(critical.any()) else 0.0
    gates = {
        "full_vs_anonymous_lift": comparisons["full_vs_anonymous"][
            "relative_nll_reduction"
        ]
        >= float(gates_cfg["full_vs_anonymous_relative_nll_reduction"]),
        "full_vs_anonymous_ci": comparisons["full_vs_anonymous"]["bca_95pct"][0]
        > float(gates_cfg["full_vs_anonymous_ci_lower"]),
        "full_vs_anonymous_holm_p": comparisons["full_vs_anonymous"]["holm_adjusted_p"]
        < float(evaluation_cfg.get("alpha", 0.01)),
        "full_vs_actor_only": comparisons["full_vs_actor_only"]["relative_nll_reduction"]
        >= float(gates_cfg["full_vs_actor_only_relative_nll_reduction"])
        and comparisons["full_vs_actor_only"]["bca_95pct"][0] > 0,
        "full_vs_roster_only": comparisons["full_vs_roster_only"][
            "relative_nll_reduction"
        ]
        >= float(gates_cfg["full_vs_roster_only_relative_nll_reduction"])
        and comparisons["full_vs_roster_only"]["bca_95pct"][0] > 0,
        "full_vs_within_roster_shuffle": comparisons["full_vs_within_roster_shuffle"][
            "relative_nll_reduction"
        ]
        >= float(gates_cfg["full_vs_within_roster_shuffle_relative_nll_reduction"])
        and comparisons["full_vs_within_roster_shuffle"]["bca_95pct"][0] > 0,
        "opponent_shuffle_separation": comparisons["full_vs_matched_opponent_shuffle"][
            "relative_nll_reduction"
        ]
        >= 0.01,
        "repeated_matchup_positive": repeated_lift is not None and repeated_lift > 0,
        "identity_coverage": known_coverage
        >= float(gates_cfg["minimum_known_identity_coverage"]),
        "positive_all_three_seeds": all(value > 0 for value in seed_lifts.values()),
        "not_driven_by_one_series": bool(leave_one_series_out)
        and min(leave_one_series_out) > 0,
    }
    results = {
        "version": 1,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "unlock_path": str(unlock_path),
        "unlock_sha256": file_sha256(unlock_path),
        "receipt_path": str(receipt),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "counts": {
            "test_samples": len(dataset),
            "critical_samples": int(critical.sum()),
            "primary_samples": int(primary.sum()),
            "primary_series": int(primary_frame["series_id"].nunique()),
            "known_identity_coverage_in_critical": float(all_known[critical].mean()),
            "opponent_shuffle_match_coverage_in_primary": float(
                np.isfinite(nll_by_name["matched_opponent_shuffle"])[primary].mean()
            ),
            "recurring_primary_samples": int((primary & recurring).sum()),
        },
        "mean_nll_primary": {
            name: float(primary_frame[f"nll_{name}"].mean()) for name in nll_by_name
        },
        "comparisons": comparisons,
        "full_vs_anonymous_relative_lift_by_seed": seed_lifts,
        "leave_one_series_out_min_relative_lift": (
            min(leave_one_series_out) if leave_one_series_out else None
        ),
        "repeated_matchup_full_minus_anonymous_improvement": repeated_lift,
        "gates": gates,
        "all_win_gates_pass": all(gates.values()),
        "checkpoint_evidence": checkpoint_evidence,
    }
    per_sample_path = output / "test_per_sample.parquet"
    sample_frame.to_parquet(per_sample_path, index=False, compression="zstd")
    results_path = output / "test_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return {"results": results_path, "per_sample": per_sample_path, "receipt": receipt}
