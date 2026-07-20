# PFF-GRF Transfer Diagnostic V1

## Question

Does geometry-only GRF pretraining improve early adaptation to PFF World Cup 2022 tracking when
the real-data budget, model, optimizer, split, and batch order are matched?

This is an optimization diagnostic. It is not evidence of tactical concepts, semantic
understanding, final model quality, or downstream usefulness.

## Frozen Inputs

- PFF split: `splits/pff_wc2022_64match_inductive_v1.json`
- split SHA-256: `bee86c2c8f917a52a007dbdb92f6082a50f6a33fd278cc8e3f52213cb31f381b`
- all-available manifest SHA-256:
  `5e8e9365afc60069423f9150537c724023ab061973ee5945a988c63e7a8ac47b`
- observed-only manifest SHA-256:
  `ca53ef656470aea212ec9365881ffcad996a8995615532d152e1cd5df00ebac2`
- objective: geometry-only, one second of context, one-second gap, separate one-second target
- split: 48 train matches, 8 validation matches, and 8 reserved test matches; the legacy runner
  later auto-exported one test-batch embedding, as corrected in
  `docs/TEST_SPLIT_ACCESS_AUDIT_2026_07_14.md`
- training budget: 100 batches, 12,800 examples
- validation budget: 50 batches, 6,400 examples
- seeds: 7, 11, and 23
- transfer mode: pretrained model weights with a fresh optimizer

The matched configs are:

- `configs/td_jepa_pff_wc2022_matched_diagnostic.yaml`
- `configs/td_jepa_pff_wc2022_observed_only_matched_diagnostic.yaml`

GRF checkpoints were independently trained with the same seeds at:

- seed 7: `runs/td_jepa/20260709_231250/best.pt`
- seed 11: `runs/td_jepa/20260709_231619/best.pt`
- seed 23: `runs/td_jepa/20260709_231638/best.pt`

## All-Available Results

| Seed | Scratch run | Scratch total | GRF run | GRF total | Relative change |
| --- | --- | ---: | --- | ---: | ---: |
| 7 | `20260712_234339` | 0.029427 | `20260712_234405` | 0.026526 | -9.9% |
| 11 | `20260712_234513` | 0.036609 | `20260712_234538` | 0.031491 | -14.0% |
| 23 | `20260712_234605` | 0.034479 | `20260712_234628` | 0.030259 | -12.2% |
| Mean | | 0.033505 | | 0.029426 | -12.2% |

GRF initialization lowers the combined validation objective in all three seeds. Mean slot and
context reconstruction losses improve by 9.8% and 8.6%, and mean online latent standard deviation
is 29.6% higher. However, mean narrow latent TD loss worsens by 8.0%. The all-available result
therefore supports faster optimization of the combined model, not better latent future prediction.

## Observed-Only Results

The observed-only dataset contains 1,135,478 unique examples in 2,039 hashed shards:

- train: 844,195
- validation: 141,054
- test: 150,229

| Seed | Scratch run | Scratch total | GRF run | GRF total | Relative change |
| --- | --- | ---: | --- | ---: | ---: |
| 7 | `20260712_234831` | 0.016887 | `20260712_234857` | 0.015743 | -6.8% |
| 11 | `20260712_234923` | 0.016979 | `20260712_234947` | 0.015130 | -10.9% |
| 23 | `20260712_235011` | 0.012796 | `20260712_235035` | 0.010960 | -14.3% |
| Mean | | 0.015554 | | 0.013944 | -10.3% |

The transfer effect survives removal of provider-estimated positions. In this stricter view, GRF
initialization also lowers mean narrow latent TD loss by 23.5%, anti-collapse loss by 48.2%, slot
reconstruction loss by 7.9%, and context reconstruction loss by 7.0%. Mean online latent standard
deviation is 10.2% higher.

Absolute losses must not be compared between visibility views because their masks and usable
example populations differ. Only scratch-versus-GRF comparisons within the same view are matched.

## Conclusion And Next Gate

GRF initialization gives a repeatable early-training benefit on PFF tracking under both visibility
policies. The observed-only result is the stronger integrity control because it does not rely on
provider-estimated player positions and improves both the combined objective and narrow latent
prediction error.

This result authorizes a prespecified longer matched repeat. It does not authorize tactical or
semantic interpretation. Before model selection or downstream claims, the longer repeat must use
multiple seeds, bounded validation selection, a single frozen test application, the existing
condition-aware falsification policy, and raw/PCA/random incremental-probe and discovery controls.
