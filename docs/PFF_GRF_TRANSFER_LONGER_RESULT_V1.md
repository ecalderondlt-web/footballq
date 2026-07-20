# PFF-GRF Longer Transfer Result V1

## Decision

Status: `blocked` on validation.

The frozen protocol is `docs/PFF_GRF_TRANSFER_LONGER_PROTOCOL_V1.md`. Its SHA-256 at execution was
`28551fe32689274780493c2bdb691cefda183935df7ca39e2b094417d1542c45`.
The machine-readable result is:

```text
runs/integrity/pff_grf_transfer_longer_v1_gate_summary.json
```

Because the validation gate failed, no PFF test loss metrics or held-out falsification controls were
run. A 2026-07-14 audit found that the legacy trainer nevertheless exported an embedding from its
first test batch after training. That forward pass did not affect weights, validation metrics, or
checkpoint selection, but the test split was not completely untouched. See
`docs/TEST_SPLIT_ACCESS_AUDIT_2026_07_14.md`.

## Runs

Each run used 2,000 training batches (256,000 examples) and 500 validation batches (64,000
examples) from the finalized observed-only PFF manifest.

| Seed | Scratch run | Scratch total | GRF run | GRF total | Relative change |
| --- | --- | ---: | --- | ---: | ---: |
| 7 | `20260713_001245` | 0.009717 | `20260713_001653` | 0.009587 | -1.3% |
| 11 | `20260713_002143` | 0.009555 | `20260713_002643` | 0.009471 | -0.9% |
| 23 | `20260713_003209` | 0.009690 | `20260713_003737` | 0.009528 | -1.7% |
| Mean | | 0.009654 | | 0.009529 | -1.3% |

## Frozen Gate Evaluation

| Criterion | Result | Decision |
| --- | ---: | --- |
| finite metrics for all six runs | yes | pass |
| transfer total-loss wins | 3 of 3 | pass; minimum 2 |
| mean total-loss improvement | 1.297% | block; minimum 5% |
| mean narrow TD relative change | +10.005% | block; maximum 0% |
| minimum `z_online_std_mean` | 0.364 | pass; must exceed 0.05 |

The narrow TD loss is worse for GRF initialization in every seed at this longer budget. The
representations do not show the monitored collapse symptom, and GRF retains a small combined-loss
advantage, but that advantage is below the prespecified meaningful-effect threshold.

## Interpretation

The 100-update observed-only diagnostic showed a 10.3% mean combined-loss benefit and a 23.5%
narrow-TD benefit. At 2,000 updates, the combined benefit shrinks to 1.3% and narrow TD is 10.0%
worse. The defensible interpretation is that current GRF pretraining is primarily a warm start:
it accelerates early adaptation, while scratch training mostly catches up after more real-data
updates.

This does not support a persistent representation-quality benefit from the current GRF setup. It
also does not show that GRF is useless; faster early optimization may still reduce real-data
compute. The next validation-only study should measure a fixed learning curve at prespecified
checkpoints or redesign the synthetic objective/domain variation. Do not access the frozen PFF
test split to resolve this blocked result.
