# PFF-GRF Sampler Comparison Result V1

## Decision

Status: `blocked` on validation.

The frozen exploratory protocol is `docs/PFF_GRF_SAMPLER_COMPARISON_PROTOCOL_V1.md`. Its SHA-256
at execution was `780cee79e85dc5621a026df3c8fcda1b63c634439128edb92f874917eb61dcdc`.
The machine-readable result is:

```text
runs/integrity/pff_grf_sampler_comparison_v1_gate_summary.json
```

No PFF or synthetic test loss metrics were computed. A later audit found that the legacy trainer
automatically exported a first-test-batch embedding after each run. Those forward passes did not
affect weights, validation metrics, or checkpoint selection, but the test tensors were not
completely untouched. See `docs/TEST_SPLIT_ACCESS_AUDIT_2026_07_14.md`.

## Frozen Sampler

The candidate assigned shard probability mass proportional to the square root of source example
count. This changed synthetic training exposure from 88.5% full-match / 11.5% academy under
natural frequency to 72.3% / 27.7%. The allocation was frozen before candidate training at
`runs/integrity/gfootball_v2_sqrt_sampler_v1_allocation_plan.json`.

Every candidate synthetic run consumed exactly 33,664 samples in 263 updates:

| Seed | Square-root synthetic run |
| --- | --- |
| 7 | `20260713_153347` |
| 11 | `20260713_153422` |
| 23 | `20260713_153456` |

The transferred checkpoint was `latest.pt` at update 263, not a synthetic-validation selection.

## PFF Validation Runs

The natural-V2 baseline runs were frozen by the preceding curriculum comparison. Each candidate
used the same seed and 2,000-update observed-only PFF configuration.

| Seed | Natural V2 run | Natural total | Square-root run | Square-root total | Relative improvement |
| --- | --- | ---: | --- | ---: | ---: |
| 7 | `20260713_150747` | 0.009584 | `20260713_153559` | 0.009446 | +1.44% |
| 11 | `20260713_151244` | 0.009272 | `20260713_153844` | 0.009325 | -0.58% |
| 23 | `20260713_151826` | 0.009380 | `20260713_154154` | 0.009414 | -0.36% |
| Mean | | 0.009412 | | 0.009395 | +0.18% |

## Frozen Gate Evaluation

| Criterion | Result | Decision |
| --- | ---: | --- |
| finite metrics for all six runs | yes | pass |
| square-root total-loss wins | 1 of 3 | block; minimum 2 |
| mean total-loss improvement | +0.181% | block; minimum +2% |
| mean narrow TD relative change | +5.783% | block; maximum 0% |
| minimum `z_online_std_mean` | 0.522 | pass; must exceed 0.05 |

The seed-7 improvement is not stable across seeds. Its magnitude produces a small favorable mean,
but the mean is far below the frozen material-effect threshold. Narrow TD loss worsens from
`0.00013487` to `0.00014267`, and there is no monitored latent-spread collapse.

## Interpretation

Increasing academy exposure through this fixed square-root sampler does not solve the transfer
problem. Together with the preceding natural-frequency curriculum result, this points away from
simple scenario under-sampling as the sole explanation. It does not prove that all reweighting or
scenario diversity is useless.

Further sampler tuning against these same validation runs would overfit the study. The next
defensible work is a train-only simulator-to-real gap audit of motion and visibility distributions,
followed by a new objective or data-generation protocol frozen from those train-only diagnostics.
Do not access PFF test, run semantic probes, or make tactical claims from this result.
