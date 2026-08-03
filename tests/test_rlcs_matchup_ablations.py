from __future__ import annotations

import numpy as np
import torch

from footballq.data.rlcs_player_profiles import PROFILE_DIMENSION
from footballq.models.player_matchup_value import PlayerMatchupValueModel, count_parameters
from footballq.training.train_rlcs_value import multiclass_metrics


def _inputs(batch: int = 2):
    generator = torch.Generator().manual_seed(11)
    return {
        "state": torch.randn(batch, 20, 7, 27, generator=generator),
        "state_mask": torch.ones(batch, 20, 7, dtype=torch.bool),
        "scalar_context": torch.randn(batch, 4, generator=generator),
        "team_form": torch.randn(batch, 6, generator=generator),
        "profiles": torch.randn(batch, 6, PROFILE_DIMENSION, generator=generator),
        "profile_uncertainty": torch.rand(
            batch, 6, PROFILE_DIMENSION, generator=generator
        ),
        "profile_effective_sample_size": torch.full((batch, 6), 10.0),
        "pair_geometry": torch.randn(batch, 3, 9, generator=generator),
        "teammate_geometry": torch.randn(batch, 2, 9, generator=generator),
    }


def test_conditions_mask_unavailable_profile_information():
    model = PlayerMatchupValueModel(dropout=0.0).eval()
    inputs = _inputs()
    altered = {key: value.clone() for key, value in inputs.items()}
    altered["profiles"][:, 3:6] += 100.0
    with torch.no_grad():
        state_a = model(**inputs, condition="state")["outcome_logits"]
        state_b = model(**altered, condition="state")["outcome_logits"]
        actor_a = model(**inputs, condition="actor_profile")["outcome_logits"]
        actor_b = model(**altered, condition="actor_profile")["outcome_logits"]
        full_a = model(**inputs, condition="full_matchup")["outcome_logits"]
        full_b = model(**altered, condition="full_matchup")["outcome_logits"]
    assert torch.equal(state_a, state_b)
    assert torch.equal(actor_a, actor_b)
    assert not torch.allclose(full_a, full_b)


def test_additive_is_opponent_permutation_invariant_but_matchup_is_not():
    model = PlayerMatchupValueModel(dropout=0.0).eval()
    inputs = _inputs()
    permuted = {key: value.clone() for key, value in inputs.items()}
    permuted["profiles"][:, 3:6] = inputs["profiles"][:, [5, 3, 4]]
    permuted["profile_uncertainty"][:, 3:6] = inputs["profile_uncertainty"][:, [5, 3, 4]]
    permuted["profile_effective_sample_size"][:, 3:6] = inputs[
        "profile_effective_sample_size"
    ][:, [5, 3, 4]]
    with torch.no_grad():
        additive_a = model(**inputs, condition="additive_profiles")["outcome_logits"]
        additive_b = model(**permuted, condition="additive_profiles")["outcome_logits"]
        full_a = model(**inputs, condition="full_matchup")["outcome_logits"]
        full_b = model(**permuted, condition="full_matchup")["outcome_logits"]
    assert torch.allclose(additive_a, additive_b, atol=1e-6)
    assert not torch.allclose(full_a, full_b)


def test_model_has_frozen_small_capacity_and_no_player_id_embedding():
    model = PlayerMatchupValueModel()
    assert 1_500_000 <= count_parameters(model) <= 2_000_000
    assert not hasattr(model, "identity_embedding")


def test_outcome_average_precision_does_not_require_sklearn():
    probabilities = np.asarray(
        [
            [0.05, 0.90, 0.05],
            [0.10, 0.80, 0.10],
            [0.70, 0.20, 0.10],
            [0.10, 0.20, 0.70],
        ],
        dtype=np.float64,
    )
    metrics = multiclass_metrics(probabilities, np.asarray([1, 0, 1, 2]))
    assert np.isclose(metrics["score_average_precision"], 5.0 / 6.0)
    assert metrics["concede_average_precision"] == 1.0
