# PFF-GRF Position-Only Longer Result V1

## Decision

Status: `blocked` on validation by the material total-loss threshold.

The frozen protocol is `docs/PFF_GRF_POSITION_ONLY_LONGER_PROTOCOL_V1.md`. Its SHA-256 at
execution was `299ef4f60e07ac14ec40fcc154de5f39030a818afe386d549c49dc31713ff7ee`.
The machine-readable result is
`runs/integrity/pff_grf_position_only_longer_v1_gate_summary.json`.

No PFF test tensor was projected or loaded. All six run manifests record only `train` and `val`,
and automatic embedding export was disabled.

## Runs

Every run used 2,000 training batches (256,000 examples) and 500 validation batches (64,000
examples) from the finalized position-only PFF manifest.

| Seed | Scratch run | Scratch total | GRF run | GRF total | Relative change |
| --- | --- | ---: | --- | ---: | ---: |
| 7 | `20260714_224323` | 0.013215 | `20260714_224633` | 0.013016 | -1.5% |
| 11 | `20260714_225003` | 0.013129 | `20260714_225505` | 0.012829 | -2.3% |
| 23 | `20260714_230005` | 0.012912 | `20260714_230608` | 0.012808 | -0.8% |
| Mean | | 0.013086 | | 0.012884 | -1.5% |

Every checkpoint records exactly 2,000 optimizer updates and the five-channel `position_only`
feature view. The transfer runs record the matching-seed GRF initialization checkpoint; scratch
runs record no initialization.

## Frozen Gate

| Criterion | Result | Decision |
| --- | ---: | --- |
| finite metrics for all six runs | yes | pass |
| transfer total-loss wins | 3 of 3 | pass; minimum 2 |
| mean total-loss improvement | 1.540% | block; minimum 5% |
| mean narrow TD relative change | -10.522% | pass; maximum 0% |
| minimum `z_online_std_mean` | 0.457 | pass; must exceed 0.05 |

## Interpretation

The early position-only benefit shrinks from 14.7% at 100 PFF updates to 1.5% at 2,000 updates.
Scratch training therefore catches most of the combined-objective lead, and the remaining effect
is below the frozen meaningful-effect threshold.

Position-only differs usefully from the old seven-channel longer result: mean narrow TD loss
improves by 10.5% instead of worsening by 10.0%. It improves in every seed, and transfer latent
spread is also higher. This suggests that omitting explicit velocity removes a source of temporal
mismatch, but it does not produce a material persistent gain in the complete validation objective.

Because the gate is blocked, no validation falsification follow-up and no PFF test evaluation are
authorized. The defensible conclusion remains that current GRF pretraining is a strong early warm
start with a small longer-budget residual benefit, not a demonstrated persistent representation
advantage.

## Claim Boundary

This result is not evidence of tactical concepts, tactical surprise, semantic understanding,
downstream utility, or superiority to raw/PCA/random controls.
