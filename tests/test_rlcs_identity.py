from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from footballq.data.rlcs_replay import (
    AliasEntry,
    IdentityObservation,
    IdentityResolutionError,
    IdentityResolver,
    load_alias_registry,
    normalize_handle,
)
from footballq.data.rlcs_touch_windows import fit_identity_vocabulary


def observation(
    *,
    replay: str = "r1",
    split: str = "train",
    prefix: str = "blue_player_1",
    handle: str = "Alpha",
    platform_id: str = "123",
) -> IdentityObservation:
    return IdentityObservation(
        replay_id=replay,
        split=split,
        event_time_utc="2025-02-01T00:00:00Z",
        group_id="g1",
        prefix=prefix,
        team="blue",
        handle=handle,
        normalized_handle=normalize_handle(handle),
        platform="steam",
        platform_id=platform_id,
    )


def test_handle_normalization_is_exact_nfkc_casefold_and_whitespace():
    assert normalize_handle("  ＡlPhA\t  PLAYER  ") == "alpha player"


def test_identity_fitting_rejects_validation_or_test_observations():
    with pytest.raises(ValueError, match="training observations only"):
        IdentityResolver.fit_training([observation(split="val")])


def test_stable_platform_id_resolves_but_unseen_vocab_is_unk():
    resolver = IdentityResolver.fit_training([observation()])
    result = resolver.resolve(observation(replay="r2", split="test"))
    assert result.canonical_player_id == "steam:123"
    vocabulary = fit_identity_vocabulary([["steam:999"]])
    assert vocabulary.encode(result.canonical_player_id) == (0, False)


def test_generic_platform_account_requires_exact_frozen_alias():
    alpha = observation(handle="Alpha")
    beta = observation(replay="r2", handle="Beta")
    resolver = IdentityResolver.fit_training([alpha, beta])
    assert resolver.resolve(alpha).method == "generic_platform_id_requires_alias"
    alias = AliasEntry(
        canonical_player_id="pro:alpha",
        canonical_handle="alpha",
        observed_handle="alpha",
        platform="steam",
        platform_id="123",
        valid_from="2025-01-01",
        valid_to="2025-12-31",
        evidence_group_id="g1",
        resolution_method="reviewed_roster",
        review_status="frozen",
    )
    resolver = IdentityResolver.fit_training([alpha, beta], [alias])
    assert resolver.resolve(alpha).canonical_player_id == "pro:alpha"
    assert resolver.resolve(beta).canonical_player_id is None


def test_alias_outside_date_range_does_not_resolve():
    alpha = observation(handle="Alpha")
    beta = observation(replay="r2", handle="Beta")
    alias = AliasEntry(
        canonical_player_id="pro:alpha",
        canonical_handle="alpha",
        observed_handle="alpha",
        platform="steam",
        platform_id="123",
        valid_from="2024-01-01",
        valid_to="2024-12-31",
        evidence_group_id="g1",
        resolution_method="reviewed_roster",
        review_status="frozen",
    )
    resolver = IdentityResolver.fit_training([alpha, beta], [alias])
    assert resolver.resolve(alpha).canonical_player_id is None


def test_metadata_only_corpus_audit_blocks_later_generic_account_collision():
    alpha = observation(handle="Alpha")
    later = observation(replay="test-r2", split="test", handle="Beta")
    resolver = IdentityResolver.fit_training([alpha]).with_collision_audit([alpha, later])
    assert resolver.resolve(later).method == "generic_platform_id_requires_alias"


def test_replay_roster_rejects_duplicate_canonical_identity():
    items = [
        observation(prefix="blue_player_1", platform_id="1", handle="A"),
        observation(prefix="blue_player_2", platform_id="1", handle="A"),
    ]
    resolver = IdentityResolver.fit_training(items)
    with pytest.raises(IdentityResolutionError, match="collision"):
        resolver.resolve_roster(items)


def test_many_to_one_handle_collision_is_not_fuzzy_resolved():
    first = observation(platform_id="1", handle="same")
    second = observation(replay="r2", platform_id="2", handle="SAME")
    resolver = IdentityResolver.fit_training([first, second])
    assert resolver.resolve(replace(first, replay_id="r3")).method == "many_to_one_handle_collision"


def test_frozen_rlcs_alias_registry_exactly_resolves_reviewed_platform_collisions():
    path = Path(__file__).resolve().parents[1] / "provenance" / "rlcs_identity_aliases_v1.csv"
    aliases = load_alias_registry(path)
    assert len(aliases) == 74
    assert len({f"{entry.platform}:{entry.platform_id}" for entry in aliases}) == 28
    observations = [
        IdentityObservation(
            replay_id=f"review-{index}",
            split="train",
            event_time_utc="2025-06-01T00:00:00Z",
            group_id=entry.evidence_group_id,
            prefix=f"reviewed_player_{index}",
            team="blue",
            handle=entry.observed_handle,
            normalized_handle=normalize_handle(entry.observed_handle),
            platform=entry.platform,
            platform_id=entry.platform_id,
        )
        for index, entry in enumerate(aliases)
    ]
    resolver = IdentityResolver.fit_training(observations, aliases)
    for item in observations:
        result = resolver.resolve(item)
        assert result.method == "frozen_alias"
        assert result.canonical_player_id == item.platform_key
