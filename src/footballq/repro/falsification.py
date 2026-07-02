"""Falsification controls for TD-JEPA-style batches."""

from __future__ import annotations

from typing import Any

import torch

CONTROL_CONDITIONS = {
    "correct_temporal_pairing",
    "shuffled_future_within_batch",
    "future_from_another_match",
    "reversed_time_context",
    "masked_ball",
    "team_swap",
    "team_label_swap",
    "target_team_label_swap",
    "pitch_reflection",
    "consistent_player_slot_permutation",
    "target_consistent_player_slot_permutation",
}


def consistent_player_slot_permutation(
    state: torch.Tensor,
    mask: torch.Tensor,
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Consistently permute player slots within each team across all context frames."""

    if state.shape[-2] != 23:
        raise ValueError("Expected canonical 23-entity state tensors.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    home = torch.arange(1, 12)
    away = torch.arange(12, 23)
    perm = torch.arange(23)
    perm[home] = home[torch.randperm(len(home), generator=generator)]
    perm[away] = away[torch.randperm(len(away), generator=generator)]
    return state[..., perm, :].contiguous(), mask[..., perm].contiguous(), perm


def _team_slot_swap(state: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    perm = torch.cat([torch.tensor([0]), torch.arange(12, 23), torch.arange(1, 12)])
    return state[..., perm, :].contiguous(), mask[..., perm].contiguous()


def _team_label_swap(state: torch.Tensor, feature_names: list[str] | None) -> torch.Tensor:
    if feature_names is None:
        raise ValueError("team_label_swap controls require feature_names.")
    names = [str(value) for value in feature_names]
    try:
        home_idx = names.index("is_home")
        away_idx = names.index("is_away")
    except ValueError as exc:
        raise ValueError("team_label_swap controls require is_home and is_away features.") from exc
    swapped = state.clone()
    swapped[..., home_idx] = state[..., away_idx]
    swapped[..., away_idx] = state[..., home_idx]
    return swapped


def apply_td_falsification_control(
    batch: dict[str, Any],
    condition: str,
    seed: int = 123,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Return a controlled copy of a TD-JEPA batch with condition metadata."""

    if condition not in CONTROL_CONDITIONS:
        raise ValueError(f"Unknown falsification control {condition!r}.")
    out = dict(batch)
    out["control_condition"] = condition
    if condition == "correct_temporal_pairing":
        return out
    if condition == "shuffled_future_within_batch":
        n = int(out["state_t_plus_delta"].shape[0])
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        perm = torch.randperm(n, generator=generator)
        out["state_t_plus_delta"] = out["state_t_plus_delta"][perm]
        out["mask_t_plus_delta"] = out["mask_t_plus_delta"][perm]
        out["control_permutation"] = perm
        return out
    if condition == "future_from_another_match":
        match_ids = [str(value) for value in out.get("match_id", [])]
        if not match_ids:
            raise ValueError("future_from_another_match requires batch match_id values.")
        perm_rows = []
        for row_idx, match_id in enumerate(match_ids):
            candidates = [idx for idx, other in enumerate(match_ids) if other != match_id]
            if not candidates:
                raise ValueError("future_from_another_match requires at least two match IDs.")
            perm_rows.append(candidates[(row_idx + int(seed)) % len(candidates)])
        perm = torch.tensor(perm_rows, dtype=torch.long)
        out["state_t_plus_delta"] = out["state_t_plus_delta"][perm]
        out["mask_t_plus_delta"] = out["mask_t_plus_delta"][perm]
        out["control_permutation"] = perm
        return out
    if condition == "reversed_time_context":
        out["state_t"] = torch.flip(out["state_t"], dims=[1])
        out["mask_t"] = torch.flip(out["mask_t"], dims=[1])
        return out
    if condition == "masked_ball":
        out["state_t"] = out["state_t"].clone()
        out["mask_t"] = out["mask_t"].clone()
        out["state_t"][:, :, 0, :] = 0.0
        out["mask_t"][:, :, 0] = False
        return out
    if condition == "team_swap":
        out["state_t"], out["mask_t"] = _team_slot_swap(out["state_t"], out["mask_t"])
        out["control_permutation"] = torch.cat(
            [torch.tensor([0]), torch.arange(12, 23), torch.arange(1, 12)]
        )
        return out
    if condition == "team_label_swap":
        out["state_t"] = _team_label_swap(out["state_t"], feature_names)
        return out
    if condition == "target_team_label_swap":
        out["state_t_plus_delta"] = _team_label_swap(
            out["state_t_plus_delta"],
            feature_names,
        )
        return out
    if condition == "pitch_reflection":
        reflected = out["state_t"].clone()
        reflected[..., 0] = -reflected[..., 0]
        out["state_t"] = reflected
        return out
    if condition == "target_consistent_player_slot_permutation":
        state, mask, perm = consistent_player_slot_permutation(
            out["state_t_plus_delta"],
            out["mask_t_plus_delta"],
            seed=seed,
        )
        out["state_t_plus_delta"] = state
        out["mask_t_plus_delta"] = mask
        out["control_permutation"] = perm
        return out
    state, mask, perm = consistent_player_slot_permutation(
        out["state_t"],
        out["mask_t"],
        seed=seed,
    )
    out["state_t"] = state
    out["mask_t"] = mask
    out["control_permutation"] = perm
    return out
