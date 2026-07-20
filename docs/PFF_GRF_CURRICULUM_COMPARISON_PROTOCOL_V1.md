# PFF-GRF Curriculum Comparison Protocol V1

Status: frozen before training on 2026-07-13.

## Purpose

Test whether the balanced GRF V2 curriculum transfers to real PFF validation data better than the
saved easy-11v11 curriculum when masking, objective, architecture, optimizer, synthetic compute,
real-data compute, and random seeds are held fixed.

This is a validation-only optimization and representation diagnostic. It cannot establish tactical
concepts, semantic understanding, or downstream usefulness.

## Frozen Synthetic Data

Baseline family:

- source: saved `11_vs_11_easy_stochastic` episodes only
- dataset manifest payload SHA-256:
  `bd7085fd1c4e02e54b41f683c3f60ca215eefd9c069d7354e8e0793bb5929e2f`
- examples: 4,860 train, 2,430 validation, 2,430 untouched synthetic test
- config: `configs/td_jepa_gfootball_saved_easy_masked_matched_v1.yaml`

Candidate family:

- source: balanced GRF V2 full-match and academy curriculum
- dataset manifest payload SHA-256:
  `f8d524157196ab007c21c0662a15b18e9458b2468b5a2af349895cec4bb981db`
- examples: 33,591 train, 2,066 validation, 1,076 untouched synthetic test
- config: `configs/td_jepa_gfootball_v2_masked_matched_v1.yaml`

Both families use the PFF-train-only visibility profile with semantic SHA-256
`3bd3e96d0c449e3f6a57e69a37001af71e863b82ca7613b3ba7022738280cd40`. Both use geometry-only
`future_nonoverlap_context_only` examples with a one-second context, one-second gap, separate
one-second target, and 10 fps sampling. Neither synthetic test split is used.

## Frozen Synthetic Training

- paired seeds: 7, 11, and 23
- batch size: 128
- optimizer updates: exactly 263 per run
- examples per run: exactly 33,664; partial training batches are dropped
- architecture, objective, loss weights, optimizer, and EMA settings: identical
- checkpoint transferred to PFF: `latest.pt` after update 263, never validation-selected `best.pt`
- the easy-only source repeats as needed; V2 receives one nearly complete pass plus one batch

The latest-checkpoint rule prevents the different dataset sizes from creating unequal checkpoint
selection opportunities.

## Frozen Real-Data Fine-Tuning

- data: PFF observed-only tracking
- dataset manifest SHA-256:
  `ca53ef656470aea212ec9365881ffcad996a8995615532d152e1cd5df00ebac2`
- split manifest SHA-256:
  `bee86c2c8f917a52a007dbdb92f6082a50f6a33fd278cc8e3f52213cb31f381b`
- split: 48 train, 8 validation, 8 untouched test matches
- config: `configs/td_jepa_pff_wc2022_grf_curriculum_comparison_v1.yaml`
- budget: 2,000 training batches and 500 validation batches per run
- initialization: matching-seed synthetic `latest.pt`, model weights only, fresh optimizer
- data order, model, losses, and all real-data settings are paired by seed

## Frozen Validation Gate

V2 passes as a material curriculum improvement only when all conditions hold:

1. all six PFF runs finish with finite validation metrics
2. V2 has lower total validation loss in at least two of three paired seeds
3. V2 lowers mean total validation loss by at least 2% relative to easy-only initialization
4. V2 mean narrow TD loss does not exceed easy-only mean narrow TD loss
5. every run has validation `z_online_std_mean` above 0.05

Failure blocks test evaluation and downstream claims. A pass permits freezing the six PFF
checkpoints and designing the next held-out control step; it does not automatically authorize test
access, probes, discovery, or interpretation.

## Claim Boundary

A pass supports only the statement that, under this fixed compute budget, the V2 synthetic
curriculum initializes PFF training better than the matched easy-only synthetic source. A blocked
result means the current variation and masking redesign has not demonstrated a persistent
validation advantage. PFF test matches remain untouched in either case during this protocol.
