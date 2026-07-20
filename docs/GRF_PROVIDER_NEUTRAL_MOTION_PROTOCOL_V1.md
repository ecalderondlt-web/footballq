# GRF Provider-Neutral Motion Protocol V1

Status: frozen before implementation and data rebuilding on 2026-07-13.

## Motivation

The train-only domain-gap audit at `docs/GRF_PFF_TRAIN_DOMAIN_GAP_RESULT_V1.md` selected player
acceleration under its prespecified rule. GRF mean player acceleration is `7.4006 m/s^2` versus
`0.8177 m/s^2` in PFF, and the gap is largest across every full-match GRF difficulty.

The current sources construct velocity differently: PFF lacks x/y velocity components and uses
causal position differences, while GRF supplies simulator direction vectors that are converted to
metres per second. This protocol removes that provider-specific difference before changing model
architecture, loss weights, simulator policy, or data volume.

## Frozen Transformation

- source: balanced GRF V2 collection and immutable episode split
- visibility masks: unchanged
- positions: unchanged
- sample identities and context/target frame indices: unchanged
- before tensor feature construction, ignore GRF-provided `vx_mps` and `vy_mps`
- recompute both components causally from consecutive x/y positions and timestamps
- use the existing `_with_causal_velocity` finite-difference implementation
- no smoothing, clipping, interpolation, learned calibration, or validation-fit parameters
- rebuild geometry-only, future-nonoverlap shards under a separately named profile

The transformation applies to ball and players for implementation consistency, but the acceptance
gate is selected from player-motion metrics. It must not change coordinates, visibility, splits, or
example inclusion.

## Frozen Train-Only Preflight

Rerun `docs/GRF_PFF_TRAIN_DOMAIN_GAP_PROTOCOL_V1.md` with the same PFF train manifest, seed,
24,576-context budget, shard selection, metrics, and scenario caps. The rebuilt GRF data passes
preflight only when all conditions hold:

1. example count, sample IDs, frame indices, and masks match the current V2 manifest exactly
2. player-acceleration gap score is below `1.0`
3. player-acceleration gap score falls by at least 25% from `1.4508`
4. player-turn gap score does not exceed 110% of its `1.1005` baseline
5. player-speed gap score does not exceed 110% of its `0.8294` baseline
6. no validation or test tensor is read by the preflight audit

Failure stops the redesign without model training. Passing preflight permits writing a separate
matched synthetic-pretraining and PFF-validation protocol; it does not itself authorize training.

## Visibility Boundary

The context-conditioned visibility mismatch found by the audit is recorded but deliberately not
changed here. Combining motion and visibility changes would prevent attribution. A separate
train-only visibility-calibration protocol may follow after this motion ablation.

## Claim Boundary

A preflight pass would show only that provider-neutral velocity construction reduces a measured
train-domain mismatch. It would not show improved representation learning, real-game validation,
tactical understanding, or downstream value. PFF validation and test remain untouched throughout
this protocol.
