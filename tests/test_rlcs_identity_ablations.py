from __future__ import annotations

import json

import numpy as np
import torch
from rlcs_test_utils import synthetic_replay

from footballq.data.rlcs_touch_windows import (
    build_replay_decisions,
    encode_decision_identities,
    fit_identity_vocabulary,
    save_identity_vocabulary,
    write_dataset_manifest,
    write_decision_parquet,
)
from footballq.models.identity_matchup_transformer import (
    IDENTITY_CONDITIONS,
    IdentityMatchupTransformer,
    apply_identity_condition,
    factorized_joint_nll,
    identity_matchup_loss,
    permute_within_roster_identities,
)
from footballq.repro.manifest import file_sha256
from footballq.training.eval_matchup import (
    bca_relative_lift_interval,
    holm_adjust,
    sign_flip_pvalue,
)
from footballq.training.train_matchup import train_matchup_from_config


def batch(batch_size: int = 3):
    torch.manual_seed(4)
    return {
        "state": torch.randn(batch_size, 20, 7, 27),
        "state_mask": torch.ones(batch_size, 20, 7, dtype=torch.bool),
        "identity_indices": torch.tensor([[1, 2, 3, 4, 5, 6]]).repeat(batch_size, 1),
        "seconds_remaining": torch.full((batch_size,), 80.0),
        "score_diff_actor": torch.zeros(batch_size, dtype=torch.long),
        "overtime": torch.zeros(batch_size, dtype=torch.bool),
    }


def test_four_conditions_mask_only_the_preregistered_identity_slots():
    identities = torch.tensor([[1, 2, 3, 4, 5, 6]])
    assert apply_identity_condition(identities, "anonymous").tolist() == [[0] * 6]
    assert apply_identity_condition(identities, "actor_only").tolist() == [[1, 0, 0, 0, 0, 0]]
    assert apply_identity_condition(identities, "roster_only").tolist() == [[0, 2, 3, 4, 5, 6]]
    assert apply_identity_condition(identities, "full").tolist() == identities.tolist()


def test_anonymous_predictions_are_invariant_to_input_identity_values():
    torch.manual_seed(7)
    model = IdentityMatchupTransformer(num_player_identities=20, dropout=0.0).eval()
    inputs = batch(2)
    first = model(**inputs, condition="anonymous")
    inputs["identity_indices"] = torch.tensor([[7, 8, 9, 10, 11, 12]]).repeat(2, 1)
    second = model(**inputs, condition="anonymous")
    for key in first:
        torch.testing.assert_close(first[key], second[key])


def test_all_conditions_use_one_identical_backbone_and_output_shapes():
    model = IdentityMatchupTransformer(num_player_identities=20, dropout=0.0).eval()
    outputs = {
        condition: model(**batch(2), condition=condition)
        for condition in IDENTITY_CONDITIONS
    }
    for result in outputs.values():
        assert result["next_touch_entity_logits"].shape == (2, 6)
        assert result["next_touch_zone_logits"].shape == (2, 18)
        assert result["retained_possession_logit"].shape == (2,)
        assert result["goal_within_8s_logit"].shape == (2,)


def test_within_roster_permutation_never_crosses_team_boundary():
    identities = torch.tensor([[1, 2, 3, 4, 5, 6]] * 5)
    output = permute_within_roster_identities(
        identities, generator=torch.Generator().manual_seed(8)
    )
    for row in output:
        assert set(row[:3].tolist()) == {1, 2, 3}
        assert set(row[3:].tolist()) == {4, 5, 6}


def test_multitask_loss_and_primary_nll_backpropagate():
    model = IdentityMatchupTransformer(
        num_player_identities=20,
        width=48,
        attention_heads=6,
        feed_forward_width=96,
        layers=1,
        dropout=0.0,
    )
    inputs = batch(2)
    outputs = model(**inputs, condition="full")
    targets = {
        "next_touch_entity": torch.tensor([0, 5]),
        "next_touch_zone": torch.tensor([1, 17]),
        "retained_possession": torch.tensor([True, False]),
        "goal_for_within_8s": torch.tensor([False, True]),
    }
    loss, parts = identity_matchup_loss(outputs, targets)
    assert set(parts) == {"loss", "entity_ce", "zone_ce", "retained_bce", "goal_focal_bce"}
    assert factorized_joint_nll(outputs, targets).shape == (2,)
    loss.backward()
    assert model.player_embedding.weight.grad is not None


def test_series_blocked_statistics_recover_known_five_percent_lift():
    rng = np.random.default_rng(9)
    anonymous = rng.normal(4.5, 0.2, size=80)
    full = anonymous * 0.95 + rng.normal(0, 0.005, size=80)
    lower, upper = bca_relative_lift_interval(
        anonymous, full, resamples=2_000, seed=3
    )
    assert lower > 0.04
    assert upper < 0.06
    assert sign_flip_pvalue(anonymous - full, permutations=2_000, seed=3) < 0.01


def test_holm_adjustment_is_multiplicity_aware():
    adjusted = holm_adjust({"a": 0.001, "b": 0.01, "c": 0.04})
    assert adjusted["a"] == 0.003
    assert adjusted["b"] == 0.02
    assert adjusted["c"] == 0.04


def test_one_step_training_writes_validation_only_checkpoint(tmp_path):
    parsed, observations, roster, inventory = synthetic_replay()
    base = build_replay_decisions(
        parsed,
        inventory=inventory,
        split="train",
        observations=observations,
        roster_ids=roster,
    )[0]
    vocabulary = fit_identity_vocabulary([base["player_ids"]])
    vocabulary_path = save_identity_vocabulary(vocabulary, tmp_path / "vocabulary.json")

    split_paths = {}
    for split, count in (("train", 8), ("val", 4), ("test", 4)):
        rows = []
        for index in range(count):
            row = dict(base)
            row["sample_id"] = f"{split}:{index}"
            row["replay_id"] = f"{split}-replay"
            row["series_id"] = f"{split}-series"
            row["split"] = split
            rows.append(row)
        split_paths[split] = write_decision_parquet(
            encode_decision_identities(rows, vocabulary), tmp_path / f"{split}.parquet"
        )
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps(
            {
                "name": "rlcs_smoke",
                "version": 1,
                "dataset": "rlcs",
                "protocol": "chronological_train_split1_val_s2r1_test_s2r2r3",
                "status": "frozen",
                "split_unit": "replay_id",
                "train_match_ids": ["train-replay"],
                "val_match_ids": ["val-replay"],
                "test_match_ids": ["test-replay"],
                "all_match_ids": ["train-replay", "val-replay", "test-replay"],
                "expected_count": 3,
            }
        ),
        encoding="utf-8",
    )
    quality = tmp_path / "quality.json"
    quality.write_text("{}", encoding="utf-8")
    manifest = write_dataset_manifest(
        output_dir=tmp_path,
        split_paths=split_paths,
        vocabulary_path=vocabulary_path,
        split_manifest_path=split_manifest,
        parser_version="test",
        quality_report_path=quality,
    )
    result = train_matchup_from_config(
        {
            "data": {
                "manifest": str(manifest),
                "split_manifest": str(split_manifest),
                "train_reflection_probability": 0.0,
            },
            "model": {
                "input_features": 27,
                "time_steps": 20,
                "entities": 7,
                "width": 48,
                "layers": 1,
                "attention_heads": 6,
                "feed_forward_width": 96,
                "dropout": 0.0,
                "identity_embedding_dim": 12,
            },
            "training": {
                "device": "cpu",
                "precision": "fp32",
                "batch_size": 4,
                "num_workers": 0,
                "learning_rate": 0.0003,
                "minimum_learning_rate": 0.00003,
                "betas": [0.9, 0.95],
                "weight_decay": 0.05,
                "gradient_clip_norm": 1.0,
                "warmup_steps": 1,
                "maximum_steps": 1,
                "validation_interval_steps": 1,
                "early_stop_patience_validations": 1,
                "run_root": str(tmp_path / "runs"),
            },
        },
        condition="full",
        seed=17,
    )
    assert result["best_step"] == 1
    assert result["best_checkpoint"].exists()
    run_manifest = json.loads(
        (result["run_dir"] / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["test_loaded"] is False
    assert run_manifest["dependency_lock_sha256"] == file_sha256("uv.lock")
