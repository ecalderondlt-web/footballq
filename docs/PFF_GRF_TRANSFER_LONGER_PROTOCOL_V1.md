# PFF-GRF Longer Transfer Protocol V1

Status: frozen before training on 2026-07-13.

## Purpose

Test whether the early GRF transfer benefit persists under a longer real-data budget without using
PFF test matches for training, model selection, threshold design, or checkpoint selection.

This protocol measures optimization and representation diagnostics only. It cannot establish
tactical concepts, semantic understanding, or downstream usefulness.

## Frozen Data And Objective

- primary view: PFF observed-only tracking
- dataset manifest SHA-256:
  `ca53ef656470aea212ec9365881ffcad996a8995615532d152e1cd5df00ebac2`
- split manifest SHA-256:
  `bee86c2c8f917a52a007dbdb92f6082a50f6a33fd278cc8e3f52213cb31f381b`
- split: 48 train matches, 8 validation matches, 8 untouched test matches
- objective: geometry-only `future_nonoverlap_context_only`
- timing: one-second context, one-second gap, separate one-second target at 10 fps
- config: `configs/td_jepa_pff_wc2022_observed_only_longer_v1.yaml`

## Frozen Training Design

- paired seeds: 7, 11, and 23
- training budget: 2,000 batches, 256,000 examples per run
- validation budget: 500 batches, 64,000 examples per run
- scratch family: random initialization
- transfer family: matching-seed GRF geometry-only checkpoint
- transfer optimizer: fresh optimizer; model weights only are transferred
- all architecture, loss, optimizer, data-order, and validation settings are paired by seed

GRF checkpoints:

- seed 7: `runs/td_jepa/20260709_231250/best.pt`
- seed 11: `runs/td_jepa/20260709_231619/best.pt`
- seed 23: `runs/td_jepa/20260709_231638/best.pt`

## Validation Gate

The longer transfer gate passes only when all of these conditions hold:

1. all six runs finish with finite validation metrics
2. transfer has lower total validation loss in at least two of three paired seeds
3. transfer lowers mean total validation loss by at least 5% relative to scratch
4. transfer mean narrow TD loss does not exceed scratch mean narrow TD loss
5. every run has validation `z_online_std_mean` above 0.05

Failure blocks test evaluation and downstream work. A pass freezes the six checkpoints and permits
one paired held-out test evaluation. It does not permit changing the model or thresholds.

## Falsification Gate

Before test access, run the frozen match-invariant v1 policy on 100 validation batches for all
three selected transfer checkpoints. Conditions and thresholds remain exactly as specified in
`docs/FALSIFICATION_POLICY_MATCH_INVARIANT_V1.md`.

If validation falsification has a blocker, do not inspect test results. If validation passes, apply
the same policy once to 100 test batches for the frozen transfer family. Evaluate the paired
scratch and transfer test losses once without checkpoint reselection.

## Claim Boundary

Passing this protocol supports only a repeatable GRF-to-PFF optimization benefit and satisfaction
of the predefined control battery. Incremental probes and discovery comparisons against raw, PCA,
and random baselines remain separate mandatory gates before any representational claim.
