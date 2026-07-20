# PFF-GRF Curriculum Comparison Result V1

## Decision

Status: `blocked` on validation.

The frozen protocol is `docs/PFF_GRF_CURRICULUM_COMPARISON_PROTOCOL_V1.md`. Its SHA-256 at
execution was `cf5b0290ab8da493f78a26bc76e3f5f22ed70208fe64286ff8f5214351de40bf`.
The machine-readable result is:

```text
runs/integrity/pff_grf_curriculum_comparison_v1_gate_summary.json
```

Because the validation gate failed, no PFF or synthetic test loss metrics were computed. A later
audit found that the legacy trainer automatically exported a first-test-batch embedding after each
run. Those forward passes did not affect weights, validation metrics, or checkpoint selection, but
the test tensors were not completely untouched. See
`docs/TEST_SPLIT_ACCESS_AUDIT_2026_07_14.md`.

## Matched Synthetic Pretraining

Every synthetic run used batch size 128, dropped partial training batches, and stopped at exactly
263 optimizer updates. PFF initialization used `latest.pt`, never the validation-selected synthetic
checkpoint.

| Seed | Easy-only synthetic run | Balanced V2 synthetic run |
| --- | --- | --- |
| 7 | `20260713_150131` | `20260713_150316` |
| 11 | `20260713_150207` | `20260713_150347` |
| 23 | `20260713_150242` | `20260713_150420` |

## PFF Validation Runs

Every PFF run used 2,000 training batches and 500 validation batches from the observed-only
manifest. Each pair used the same seed, model, optimizer, data order, and real-data budget.

| Seed | Easy-only run | Easy-only total | Balanced V2 run | V2 total | V2 relative improvement |
| --- | --- | ---: | --- | ---: | ---: |
| 7 | `20260713_150515` | 0.009406 | `20260713_150747` | 0.009584 | -1.89% |
| 11 | `20260713_151006` | 0.009305 | `20260713_151244` | 0.009272 | +0.36% |
| 23 | `20260713_151534` | 0.009498 | `20260713_151826` | 0.009380 | +1.24% |
| Mean | | 0.009403 | | 0.009412 | -0.09% |

## Frozen Gate Evaluation

| Criterion | Result | Decision |
| --- | ---: | --- |
| finite metrics for all six runs | yes | pass |
| Balanced V2 total-loss wins | 2 of 3 | pass; minimum 2 |
| mean total-loss improvement | -0.092% | block; minimum +2% |
| mean narrow TD relative change | +4.572% | block; maximum 0% |
| minimum `z_online_std_mean` | 0.467 | pass; must exceed 0.05 |

Balanced V2 wins two seeds, but the seed-7 regression is larger than the two improvements. The
mean total loss is effectively tied and slightly worse for V2. Mean narrow TD loss also worsens,
from `0.00012897` to `0.00013487`. There is no monitored latent-spread collapse.

## Interpretation

Under equal synthetic and real-data compute, the current balanced V2 curriculum does not provide a
persistent validation advantage over the PFF-masked easy-only source. The result does not show that
scenario diversity is useless. It shows that this particular natural-frequency mixture, objective,
and 263-update budget did not convert the added diversity into a stable PFF benefit.

The V2 manifest is dominated by full-match windows, while several short academy drills contribute
few examples. A future validation-only study may freeze a scenario-aware sampler or a learning-curve
comparison before training. It must not tune weights from these six validation outcomes, and it must
not inspect PFF test, run downstream probes, or make tactical claims from this blocked result.
