# PFF-GRF Position-Only Longer Protocol V1

Status: frozen before longer training on 2026-07-14.

## Purpose

Test whether the position-only GRF benefit that passed at 100 PFF updates persists at 2,000 PFF
updates. This remains a validation-only optimization and representation diagnostic.

## Frozen Inputs

- diagnostic gate result SHA-256:
  `ed197fdfa6fb8150c15b05f43479971d3b3750829965766faaa69afa5c5c6259`
- PFF projected manifest payload SHA-256:
  `37acb8a6a00e4842a8aef8dce2700417fd7dfa24c827c3a9f46c7dac782c24ae`
- PFF split manifest SHA-256:
  `bee86c2c8f917a52a007dbdb92f6082a50f6a33fd278cc8e3f52213cb31f381b`
- feature view: `x_norm`, `y_norm`, `is_ball`, `is_home`, `is_away`
- objective: position-only `future_nonoverlap_context_only`
- timing: one-second context, one-second gap, separate one-second target at 10 fps
- PFF population: 48 train matches and 8 validation matches; no projected test tensors
- config: `configs/td_jepa_pff_wc2022_position_only_longer_v1.yaml`

Matching-seed GRF initialization checkpoints:

- seed 7: `runs/position_only_diagnostic_v1/td_jepa/20260714_223442/latest.pt`, SHA-256
  `e0d8821c986994b0212a7f26dc1921b5ad3ad1e3fd39fd2216e86d4248102406`
- seed 11: `runs/position_only_diagnostic_v1/td_jepa/20260714_223518/latest.pt`, SHA-256
  `c0a0dbf07510fbf9a23268daab6962f7a52d8c0628e4d14b6e76db9b60225de6`
- seed 23: `runs/position_only_diagnostic_v1/td_jepa/20260714_223552/latest.pt`, SHA-256
  `e70bfde87c9fe2dffd89eee511d3102902d8c11d4931dec2793bb2a92bbb7423`

## Frozen Training Design

- paired seeds: 7, 11, and 23
- training budget: 2,000 batches, 256,000 examples per run
- validation budget: 500 batches, 64,000 examples per run
- scratch family: random initialization
- transfer family: matching-seed GRF checkpoint with a fresh optimizer
- all architecture, objective, optimizer, batch-order, validation-order, and budget settings paired
- validation split: `val`
- embedding sample split: disabled
- PFF test tensor files must not be loaded

## Validation Gate

The longer repeat passes only when all conditions hold:

1. all six runs finish with finite validation metrics
2. transfer has lower total validation loss in at least two of three paired seeds
3. transfer lowers mean total validation loss by at least 5% relative to scratch
4. transfer mean narrow TD loss does not exceed scratch mean narrow TD loss
5. every run has validation `z_online_std_mean` above 0.05

Failure blocks this feature-view path and all PFF test access. A pass permits a separately frozen
validation falsification check; it does not itself authorize test evaluation.

## Claim Boundary

Passing supports only a persistent validation optimization benefit under this model, objective,
data view, and budget. It does not establish tactical concepts, tactical surprise, semantic
understanding, downstream utility, or superiority to raw/PCA/random baselines.
