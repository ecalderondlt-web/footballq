# SkillCorner Tactical Transfer Pilot V1

## Status

Complete. This is an exploratory two-match transfer test, not confirmatory evidence of
general tactical understanding.

## Question

Do position-only encoders trained on PFF, or initialized with increasing amounts of GRF before
PFF fine-tuning, make later real-match outcomes easier to predict from geometry available before
a possession phase starts?

## Frozen protocol

- Real data: all 10 SkillCorner Open Data matches under the immutable 6/2/2 match split.
- Encoder input: `x_norm`, `y_norm`, `is_ball`, `is_home`, and `is_away` only.
- Context: a one-second window ending strictly before the labeled phase starts.
- Outcomes: turnover within five seconds and penalty-area entry within five seconds.
- Train/validation/test examples: 2,193 / 668 / 791.
- Compared encoders: PFF-only, GRF 1x + PFF, GRF 4x + PFF, and GRF 8x + PFF.
- Encoder seeds: 7, 11, and 23 for every family.
- Controls: raw flattened geometry, train-only PCA, random projection, and random noise.
- Probe: the same frozen class-balanced linear logistic probe for every feature source.
- Primary metric: macro F1. Average precision is also reported because penalty-area entry is rare.

The frozen manifest records the split, source tensor, 20 label files, 12 checkpoints, protocol
configuration, and implementation hashes. Test model predictions were not opened until after the
train/validation preflight passed. Test label support had been manually audited before the freeze,
which is recorded as a limitation rather than described as a fully untouched test set.

## Test results

### Latent-only probes

| Feature source | Turnover macro F1 | Turnover AP | Penalty entry macro F1 | Penalty entry AP |
|---|---:|---:|---:|---:|
| Raw geometry | 0.502 | 0.358 | 0.314 | 0.070 |
| Raw PCA-128 | 0.506 | 0.387 | 0.229 | 0.072 |
| PFF-only latent | 0.534 | 0.435 | 0.509 | 0.183 |
| GRF 1x + PFF latent | 0.550 | 0.432 | 0.494 | 0.210 |
| GRF 4x + PFF latent | 0.544 | 0.417 | 0.523 | 0.189 |
| GRF 8x + PFF latent | 0.546 | 0.438 | 0.523 | 0.184 |

The compact learned latents make both outcomes more linearly decodable than raw, PCA, random
projection, and random-noise controls. The clearest effect is penalty-area entry. This is positive
representation evidence, but it can still reflect a useful nonlinear reorganization of geometry
rather than learned tactical concepts.

### Incremental raw-plus-latent gate

Macro-F1 gain over the raw-geometry probe:

| Encoder family | Turnover gain | Penalty-entry gain |
|---|---:|---:|
| PFF-only | -0.010 | +0.025 |
| GRF 1x + PFF | +0.002 | +0.028 |
| GRF 4x + PFF | -0.002 | +0.025 |
| GRF 8x + PFF | -0.015 | +0.042 |

Penalty-area entry passes the predeclared +0.01 incremental threshold for every encoder family.
Turnover does not. The incremental result is therefore mixed, not a general tactical-learning pass.
The concatenated probe is also much higher-dimensional than the latent-only probe, so its weaker
absolute performance should not be interpreted as evidence that raw geometry destroys latent
information.

## GRF decision

Under the frozen raw-plus-latent gate, the best GRF gain over PFF-only was:

- `+0.012` macro F1 for turnover, from GRF 1x + PFF.
- `+0.017` macro F1 for penalty-area entry, from GRF 8x + PFF.

Both are below the predeclared material-gain threshold of `+0.020`. GRF therefore remains useful as
a plausible initialization source, but this experiment does not show a material or monotonic GRF
advantage. Increasing GRF from 1x to 4x to 8x did not produce consistent tactical-transfer gains.

## Conclusion

The architecture has learned a compact representation that makes two later football outcomes easier
to decode from pre-phase geometry. That is stronger evidence than the earlier trajectory-only tests,
but it is not yet evidence of inherent tactical understanding.

The next experiment should prioritize a larger, newly frozen set of real matches with aligned event
labels. It should repeat the same causal outcomes with match-level uncertainty and a pre-frozen,
regularization-matched incremental probe. More GRF scaling should wait until that larger real-data
benchmark can distinguish a small synthetic-pretraining effect from two-match variation.

## Artifacts

- `runs/skillcorner_tactical_transfer_v1/frozen_protocol_manifest.json`
- `runs/skillcorner_tactical_transfer_v1/preflight_examples.pt`
- `runs/skillcorner_tactical_transfer_v1/tactical_examples.pt`
- `runs/skillcorner_tactical_transfer_v1/embeddings.pt`
- `runs/skillcorner_tactical_transfer_v1/results.json`
