# StatsBomb Semantic Event Pretraining Result V1

## Decision

Status: `blocked` for 360 integration. The operational semantic family is `event_only`.

The event-sequence encoder itself passes its raw-control test decisively: mean final event-type NLL
is 0.530884, a 47.978% improvement over the train-fitted first-order Markov baseline of 1.020508.
However, adding sparse StatsBomb 360 geometry does not improve the frozen three-seed objective. It
worsens mean anchored event NLL by 0.132%, worsens anchored location MAE by 6.667%, and loses the
anchored location comparison in all three seeds.

The frozen protocol is `docs/STATSBOMB_SEMANTIC_PRETRAIN_PROTOCOL_V1.md`, with execution SHA-256
`f5ab883b837983d151b85179b6ce0333c96123f7b8c200ea2ea7523062bebdc6`. The machine-readable
result is `runs/statsbomb_semantic_pretrain_v1/gate_summary.json`.

## Data And Runs

- source snapshot: StatsBomb Open Data commit
  `b0bc9f22dd77c206ddedc1d742893b3bbe64baec`
- split: 3,388 train / 425 validation / 422 test matches
- prepared data: 11,890,025 train events and 1,503,962 validation events
- causal windows: 739,046 train and 93,479 validation
- model inputs: 32 events per window
- final training: 5,700 updates and 729,600 windows per run
- seeds: 7, 11, and 23 for both feature views
- final validation: all 93,479 windows, 2,991,328 event targets, and 278,803 360-anchored targets

All six final evaluations are finite and use the fixed endpoint. No best checkpoint or individual
seed replaces the final paired result.

## Final Validation

Lower NLL and MAE are better.

| Seed | View | Overall event NLL | Overall location MAE | Anchored event NLL | Anchored location MAE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | event only | 0.529614 | 0.077128 | 0.446344 | 0.066182 |
| 7 | event + 360 | 0.530487 | 0.079066 | 0.451026 | 0.074657 |
| 11 | event only | 0.529920 | 0.078491 | 0.445815 | 0.067667 |
| 11 | event + 360 | 0.532623 | 0.080932 | 0.445392 | 0.072743 |
| 23 | event only | 0.533118 | 0.079881 | 0.448289 | 0.069615 |
| 23 | event + 360 | 0.532635 | 0.078886 | 0.445803 | 0.069628 |
| **Mean** | **event only** | **0.530884** | **0.078500** | **0.446816** | **0.067821** |
| **Mean** | **event + 360** | **0.531915** | **0.079628** | **0.447407** | **0.072343** |

The event-only seeds are tightly grouped. The 360 event-NLL effect is mixed by seed and nearly
neutral on the mean. The location effect is consistently unfavorable: 360 is worse in every seed
on anchored location MAE.

## Raw Controls

| Control or model | Event-type NLL | Change vs Markov |
| --- | ---: | ---: |
| global event frequency | 1.944380 | +90.53% |
| first-order Markov transition | 1.020508 | reference |
| event-only neural mean | **0.530884** | **-47.98%** |
| event + 360 neural mean | 0.531915 | -47.88% |

This is strong validation evidence that the causal encoder uses event context beyond a one-step
event-type transition table. It is not, by itself, evidence that its latent representation contains
tactical concepts or will improve the tracking model.

The event-only location MAE is also 56.37% below the copy-current-location baseline of 0.179904.
That comparison is descriptive because the frozen representation-value control is event NLL.

## Frozen Gate

| Criterion | Result | Decision |
| --- | ---: | --- |
| all six runs finite | yes | pass |
| event only improvement over Markov | 47.978% | pass; minimum 1% |
| 360 anchored event-NLL wins | 2 of 3 | pass; minimum 2 |
| mean anchored event-NLL improvement | -0.132% | **block; minimum +1%** |
| 360 anchored location-MAE wins | 0 of 3 | **block; minimum 2** |
| mean anchored location-MAE improvement | -6.667% | **block; minimum +1%** |
| mean overall event-NLL change with 360 | +0.194% | pass; maximum +1% |
| mean overall location-MAE change with 360 | +1.437% | **block; maximum +1%** |
| exact final coverage and access integrity | yes | pass |

A negative improvement means the 360 family is worse. The gate is blocked by the material
anchored-event threshold and the anchored/overall location criteria, not by instability, incomplete
runs, missing anchors, or test leakage.

## Interpretation

The sequential-event phase is useful. A causal Transformer learns substantially more predictive
structure than global frequencies or the immediately preceding event type, and that result repeats
across three seeds.

The current 360 integration is not useful. A single static visible-player set attached to the
current event does not improve the next-event location objective. Plausible explanations include
sparse coverage, camera-dependent visibility, stale provider joins, clipped out-of-pitch
coordinates, and a mismatch between static geometry and the next-event target. The frozen study
does not distinguish those explanations.

Do not discard StatsBomb 360, but do not concatenate this geometry path into the operational
semantic encoder. A future 360 study should change the objective: geometry reconstruction,
contrastive event/freeze-frame alignment, balanced anchored batches, or late fusion after separate
pretraining. It should not repeat the same next-event attachment at greater scale.

## Integrity And Boundaries

- All 3,813 processed shards passed hash, shape, finite-value, vocabulary, split, and period-window
  checks.
- A post-run artifact audit rechecks every frozen input and all checkpoint, metric, curve, and run
  manifest hashes for the six result-bearing runs. Its report is
  `runs/integrity/statsbomb_semantic_pretrain_v1_artifact_audit.json`.
- Every result-bearing run loaded only train and validation tensors and exported no embeddings.
- No StatsBomb test event was tensorized, trained on, evaluated, or used for selection.
- Before protocol freeze, the 42 test 360 JSON files received a structural syntax parse only; no
  values or metrics were retained. The test split is outcome- and metric-sealed, not byte-untouched.
- PFF data and the scratch tracking backbone were not loaded by this phase.
- StatsBomb and PFF matches are not aligned, so no cross-provider timestamp join was attempted.

The next integration candidate is the `event_only` encoder, evaluated as a frozen semantic context
source against raw event and tracking-only baselines. No tactical, semantic-understanding,
tactical-surprise, or downstream tracking claim follows from this pretraining result alone.
