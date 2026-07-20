# PFF 4x Tracking Backbone Complete Result V1

## Decision

Status: `blocked` on validation. The operational tracking-backbone family is `scratch`.

After exactly 10,000 PFF training updates, 4x GRF initialization worsens mean total validation
loss by 3.930% and loses the total-loss comparison in all three paired seeds. It improves mean
narrow TD loss by 32.375%. The frozen gate therefore rejects 4x GRF as the initializer for the
complete tracking backbone, while retaining evidence that it teaches a persistent
motion-prediction advantage.

The frozen protocol is `docs/PFF_4X_TRACKING_BACKBONE_COMPLETE_PROTOCOL_V1.md`, with execution
SHA-256 `569c11a065b9b0dbbfa8a4e234751a928af088fc9844e2022da096dc4f1985ab`. The
machine-readable result is `runs/pff_4x_tracking_complete_v1/gate_summary.json`.

## Final Validation

Each run used 10,000 training batches of 128 examples. Final evaluation used 500 validation
batches, or 64,000 examples, at the fixed endpoint. Lower values are better.

| Seed | Scratch total | 4x GRF total | Total change | Scratch TD | 4x GRF TD | TD change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.0116271 | 0.0118590 | +1.994% | 0.00012269 | 0.00010039 | -18.173% |
| 11 | 0.0121967 | 0.0128041 | +4.981% | 0.00013173 | 0.00010420 | -20.905% |
| 23 | 0.0114494 | 0.0119962 | +4.776% | 0.00012888 | 0.00005462 | -57.618% |
| **Mean** | **0.0117577** | **0.0122198** | **+3.930%** | **0.00012777** | **0.00008640** | **-32.375%** |

The direction is consistent across seeds: scratch is better on the combined objective, while GRF
initialization is better on the narrow temporal-difference term.

## Frozen Gate

| Criterion | Result | Decision |
| --- | ---: | --- |
| finite metrics for all six runs | yes | pass |
| 4x GRF total-loss wins | 0 of 3 | **block; minimum 2** |
| mean total-loss improvement | -3.930% | **block; minimum +1%** |
| mean narrow TD relative change | -32.375% | pass; maximum 0% |
| minimum `z_online_std_mean` | 0.827 | pass; must exceed 0.05 |
| run access and curve integrity | all checks passed | pass |

Here, a negative total-loss improvement means that the 4x GRF family is worse than scratch. The
gate is blocked by the seed-win and total-improvement rules, not by numerical instability,
representation collapse, missing artifacts, or test leakage.

## Learning Curve

These descriptive means use the first 50 validation batches. They did not select checkpoints or
change the decision. The final decision above uses the larger 500-batch evaluation.

| PFF updates | Scratch total | 4x GRF total | 4x total change | Scratch TD | 4x GRF TD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.020302 | 0.017028 | -16.127% | 0.00036454 | 0.00035516 |
| 250 | 0.016389 | 0.015418 | -5.926% | 0.00018878 | 0.00021302 |
| 500 | 0.016759 | 0.015702 | -6.306% | 0.00018343 | 0.00015826 |
| 1,000 | 0.013631 | 0.013072 | -4.103% | 0.00017168 | 0.00011596 |
| 2,000 | 0.014057 | 0.013607 | -3.199% | 0.00016193 | 0.00011441 |
| 5,000 | 0.012550 | 0.012109 | -3.516% | 0.00018797 | 0.00011530 |
| 10,000 | 0.011705 | 0.012732 | +8.769% | 0.00012680 | 0.00008429 |

GRF provides a clear early head start on the combined objective through 5,000 updates. Scratch
then catches and passes it by 10,000 updates. The TD advantage remains, so the endpoint is a
tradeoff rather than simple forgetting of the synthetic initialization.

## Objective Breakdown

At final validation, the 4x GRF family changes the mean loss components relative to scratch as
follows:

| Component | Change with 4x GRF |
| --- | ---: |
| narrow TD prediction | **-32.375%** |
| player-slot reconstruction | +4.437% |
| context reconstruction | +5.690% |
| ball-dynamics reconstruction | +19.055% |

This explains the split result. Current GRF pretraining leaves the model substantially better at
the narrow motion task, but worse at reconstructing the broader real-game state, particularly the
ball component. Those broader terms outweigh the TD gain in the complete objective.

## Integrity

- All six runs recorded exactly the seven frozen curve steps and a final step of 10,000.
- Every run loaded only PFF `train` and `val` tensors.
- Embedding export was disabled.
- The PFF test split was not loaded, evaluated, or used for selection.
- Final latent spread remains well above the frozen anti-collapse floor in every run.
- Matching-seed 4x source checkpoint hashes and all final artifact hashes are recorded in the
  execution manifest.

## Conclusion

Use the scratch family as the operational real-tracking backbone. Do not discard simulation: the
32.4% TD improvement is strong evidence that GRF teaches useful motion structure. However, do not
initialize the whole backbone from the current GRF objective for final training, because its
broader real-game representation is worse after convergence.

The next GRF work should isolate the useful motion signal instead of scaling the same pretraining:
for example, an auxiliary motion head, a partially transferred encoder, or a loss-balanced adapter
that can preserve TD gains without degrading player, context, and ball reconstruction. Semantic
event integration remains a separate phase and requires local StatsBomb or Wyscout event files and
a separately frozen alignment protocol.

This result is not evidence of tactical concepts, tactical surprise, semantic understanding, or
downstream utility.
