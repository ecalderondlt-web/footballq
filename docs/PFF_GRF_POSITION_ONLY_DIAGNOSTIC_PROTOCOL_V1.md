# PFF-GRF Position-Only Diagnostic Protocol V1

Status: frozen before data projection or model training on 2026-07-14.

## Purpose

Test whether the early GRF warm-start benefit survives when both GRF and PFF use positions and
entity identity only, with no explicit velocity channels. This is an optimization diagnostic, not
evidence of tactical concepts, semantic understanding, final model quality, or downstream value.

## Frozen Data

- GRF source manifest payload SHA-256:
  `88eebc0646f2ca50d2e8d12945e86635487fc39aa7f789af70f36765239e7e7c`
- GRF source population: train only, 30,134 examples, 89.709% retention after actual-jump
  segmentation
- PFF source manifest payload SHA-256:
  `ca53ef656470aea212ec9365881ffcad996a8995615532d152e1cd5df00ebac2`
- PFF split manifest SHA-256:
  `bee86c2c8f917a52a007dbdb92f6082a50f6a33fd278cc8e3f52213cb31f381b`
- PFF projection population: the frozen 48 train and 8 validation matches only
- ordered channels: `x_norm`, `y_norm`, `is_ball`, `is_home`, `is_away`
- objective: `future_nonoverlap_context_only`
- timing: one-second context, one-second gap, separate one-second target at 10 fps
- visibility: PFF observed-only; GRF uses the frozen PFF-train observed-visibility profile

The PFF projection must preserve sample IDs, masks, timing indices, coordinates, split labels, and
example counts exactly. It may only remove `vx_norm` and `vy_norm`. PFF test tensor shards must not
be opened or copied.

## Frozen Training Design

- paired seeds: 7, 11, and 23
- GRF pretraining: 263 optimizer updates per seed, batch size 128, latest fixed-budget checkpoint
- GRF validation: disabled because the selected candidate was built train-only
- PFF diagnostic budget: 100 training batches and 50 validation batches per run
- scratch family: random initialization
- transfer family: matching-seed position-only GRF checkpoint with a fresh optimizer
- architecture, loss, optimizer, batch order, validation order, and budgets are paired by seed
- model input and target feature view are position-only in both families
- automatic embedding export is disabled for every run

The trainer may read only GRF train tensors and PFF train/validation tensors. The source split
manifest and dataset manifests may be read for provenance, but PFF test tensor files may not be
loaded. GRF pretraining uses `latest.pt`; it does not select a checkpoint from training loss.

## Validation Gate

The diagnostic passes only when all of these conditions hold:

1. all six PFF runs finish with finite validation metrics
2. transfer has lower total validation loss in at least two of three paired seeds
3. transfer lowers mean total validation loss by at least 5% relative to scratch
4. transfer mean narrow TD loss does not exceed scratch mean narrow TD loss
5. every run has validation `z_online_std_mean` above 0.05

A pass permits a separately frozen 2,000-update position-only repeat. A block stops this branch of
the experiment at validation and does not permit PFF test evaluation.

## Seven-Channel Reference

The previous observed-only seven-channel 100-update diagnostic reported a 10.3% mean total-loss
benefit and a 23.5% mean narrow-TD benefit from GRF initialization. Those relative effects are a
descriptive reference only. Absolute losses cannot be compared across the five- and seven-channel
views because their reconstruction targets have different dimensionality.

## Claim Boundary

Passing this protocol supports only a repeatable early optimization benefit from position-only GRF
initialization. It does not establish a persistent long-budget benefit, tactical concepts, tactical
surprise, semantic understanding, or downstream utility.
