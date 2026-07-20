# PFF StatsBomb Context Residual Result V1

## Decision

Status: `blocked`. The operational family remains `tracking`.

The frozen StatsBomb event representation lowers mean validation TD loss relative to the matched
tracking-only residual in all three seeds, but the mean improvement is only 0.457%, below the
frozen 1% materiality threshold. Raw PFF event summaries also win all three seeds by only 0.336%,
below their 1% fallback threshold. Pretraining improves only 0.121% over raw events and 0.378% over
the random frozen encoder, missing both materiality thresholds.

The frozen protocol is `docs/PFF_STATSBOMB_CONTEXT_RESIDUAL_PROTOCOL_V1.md`, with SHA-256
`7f56e3a24a07616e0d7f0bcac07f0becb2753102944fee5806adff7182fe6031`. The machine-readable
result is `runs/pff_statsbomb_context_residual_v1/gate_summary.json`.

## Data And Runs

- PFF train-only event audit: 48 matches and 71,300 retained events
- explicitly mapped events: 66,725 (93.58%)
- retained unknown events: 4,575
- excluded generic `OTB` interval markers: 60,778
- final event tensors: 48 train and 8 validation matches, 83,412 events total
- causal event history: last 32 same-match, same-period events ending no later than the observed
  tracking context
- families: tracking, raw, frozen random encoder, and frozen pretrained encoder
- seeds: 7, 11, and 23
- trainable parameters: 66,432 in every family
- training: 2,000 updates per run
- final validation: the same 64,000 examples per run

All 12 runs use the final checkpoint. No best step or individual seed replaces the paired result.

## Final Validation

Lower normalized latent TD loss is better.

| Seed | Tracking | Raw events | Random encoder | Pretrained encoder |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.000112077 | 0.000111770 | 0.000112037 | **0.000111625** |
| 11 | 0.000122614 | **0.000122147** | 0.000122530 | 0.000122562 |
| 23 | 0.000115208 | 0.000114807 | 0.000115057 | **0.000114115** |
| **Mean** | 0.000116633 | 0.000116241 | 0.000116541 | **0.000116101** |

The pretrained family is numerically best on the mean and beats tracking in every seed. The effect
is small: the absolute mean reduction is approximately `5.33e-7` TD-loss units.

## Frozen Gate

| Criterion | Result | Decision |
| --- | ---: | --- |
| finite metrics and exact endpoints | yes | pass |
| matching event-history counts | yes | pass |
| identical frozen base loss per seed | yes | pass |
| pretrained wins vs tracking | 3 of 3 | pass |
| pretrained mean improvement vs tracking | 0.457% | **block; minimum 1%** |
| pretrained wins vs raw | 2 of 3 | pass |
| pretrained mean improvement vs raw | 0.121% | **block; minimum 0.5%** |
| pretrained wins vs random | 2 of 3 | pass |
| pretrained mean improvement vs random | 0.378% | **block; minimum 1%** |
| correct context wins vs event ablation | 3 of 3 | pass |
| mean improvement vs event ablation | 30.758% | pass; minimum 1% |
| raw fallback wins vs tracking | 3 of 3 | pass |
| raw fallback mean improvement | 0.336% | **not selected; minimum 1%** |

## Interpretation

PFF event history carries a small, repeatable signal for this frozen future-latent objective. Both
raw and pretrained context beat the tracking residual in every seed. However, the gain is too small
to meet the prespecified materiality rule, and most of it is already available from the raw PFF
event summary. The StatsBomb-pretrained representation adds very little beyond those raw labels.

The large event-ablation gap shows that an event-conditioned head learns to depend on its input. It
does not show that the input produces a material improvement over the matched tracking model. A
trained conditional head receiving an all-zero vector is deliberately out of distribution, so the
ablation is a dependence check, not the primary value comparison.

Do not add the current StatsBomb context residual to the operational tracking model. The next full
model path should retain tracking-only conditioning. PFF events remain useful as aligned labels,
stratification variables, or event-boundary evaluation slices. A future event integration study
would need a more targeted objective, such as event-transition prediction or event-local motion
evaluation, rather than a larger version of this residual head.

## Integrity And Boundaries

- The provider mapping was audited on all 48 PFF training matches before validation preparation.
- All 56 train/validation event shards passed hash, shape, ordering, mapping, split, and finite-value
  checks.
- All 12 result-bearing runs used only train and validation tensors and exported no embeddings.
- PFF test event files and test tracking tensors were not loaded, prepared, evaluated, or selected
  against in this study.
- StatsBomb and PFF matches were never joined. The frozen StatsBomb vocabulary and weights were
  applied to causal PFF events from the same PFF match as each tracking example.
- All checkpoints, metrics, curves, manifests, and frozen inputs passed the post-run artifact audit.

This result supports only a small event-context effect on a frozen latent-prediction task. It is not
evidence of tactical concepts, semantic understanding, player intent, or downstream match value.
