# RLCS Player-Matchup Critical Value V2: Pre-Outcome Amendment 01

Status: **frozen before V2 outcome construction**  
Frozen at: `2026-08-03T21:18:19Z`  
Base Git commit: `3f8b4351f9268ffaef091eb779b429bcd41197b1`  
Amendment scope: profile-stability entry gate only

## Reason for amendment

The original protocol required at least 60 players with 15 support games and explicitly required
a stop if that count failed. The complete mechanically eligible cohort contained 48 players, so
the original protocol stopped as required. The count was not quietly changed and the 15-game
requirement was not relaxed.

V2 conditions on continuous, chronology-safe profiles rather than fitting 48 identity embeddings.
Before any V2 ten-touch outcome row was constructed or examined, the user authorized this
reduced-cohort pilot amendment: replace the arbitrary 60-player count with stronger player-level
uncertainty and regional-stability requirements. At freeze time these files did not exist:

- `data/processed/rlcs_player_matchup_value_v2/train.parquet`;
- `data/processed/rlcs_player_matchup_value_v2/internal_development.parquet`;
- `data/processed/rlcs_player_matchup_value_v2/dataset_manifest.json`.

The original stop result remains an immutable historical record. This amendment controls only
whether work may resume after that stop.

## Complete cohort

Eligibility remains mechanical and unchanged: every player with at least 15 accepted games in the
frozen profile-support period is eligible. Use the complete available cohort with no manual
selection or removal.

The amended cohort gates are:

- eligible players: exactly 48, the complete available cohort;
- prior support games per eligible player: at least 15;
- eligible EU players: at least 20;
- eligible NA players: at least 20.

The audit records the sorted eligible-player cohort hash. Any cohort change invalidates this
amendment and requires a new documented freeze.

## Stability and uncertainty gates

The early/late profile construction, core traits, matched negative-player construction, and
profile features remain unchanged. Require all of:

| Gate | Requirement |
|---|---:|
| Same-player retrieval AUC point estimate | at least 0.75 |
| Retrieval AUC 95% player-bootstrap lower bound | at least 0.65 |
| Median split-half Spearman point estimate | at least 0.35 |
| Median Spearman 95% player-bootstrap lower bound | strictly above 0.20 |
| EU retrieval effect | AUC strictly above chance, 0.50 |
| NA retrieval effect | AUC strictly above chance, 0.50 |

“Positive separately in EU and NA” is operationalized before outcomes as retrieval AUC greater
than 0.50 in each region, because AUC minus 0.50 is the retrieval effect relative to chance.

## Frozen bootstrap

Use 10,000 deterministic percentile-bootstrap resamples with seed `20260803`. The player is the
resampling unit. Resample eligible players with replacement separately inside EU and NA, retaining
the observed regional composition in every replicate.

Each selected player contributes its early profile, late profile, same-player retrieval score,
and already matched different-player retrieval score. For every replicate:

1. calculate retrieval AUC from the resampled positive and negative player pairs;
2. calculate early/late Spearman correlation for each frozen core continuous trait;
3. take the median of the finite trait correlations.

The reported lower bound is the 2.5th percentile across the 10,000 player-bootstrap replicates.
No match, touch, or decision row is a bootstrap unit for this entry gate.

## Decision rule

Proceed to Split 1 V2 outcome construction and training only when every amended cohort,
point-estimate, confidence-bound, and regional gate passes. Stop if any one fails.

All downstream rules remain unchanged:

- score and concede must each have at least 5,000 positive training rows;
- full matchup must beat state plus team form, actor-only, additive profiles, and every required
  profile-shuffle control by the frozen margins;
- uncertainty remains at the official-series level downstream;
- EU and NA downstream point estimates must both be positive;
- Split 2 Regional 1 remains frozen validation and cannot be opened before Split 1 architecture
  freeze;
- Split 2 Regionals 2 and 3 remain sealed test and require a passing validation plus the one-use
  unlock.

## Claim boundary

This is a reduced-cohort pilot among established RLCS professionals. A positive result may support
identity-conditioned matchup value within that established-player population. It cannot support
broad generalization to new, lightly observed, lower-tier, or all Rocket League players.
