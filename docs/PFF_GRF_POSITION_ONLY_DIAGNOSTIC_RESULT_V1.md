# PFF-GRF Position-Only Diagnostic Result V1

Status: validation controls passed on 2026-07-14.

The frozen protocol is `docs/PFF_GRF_POSITION_ONLY_DIAGNOSTIC_PROTOCOL_V1.md`. Its SHA-256 at
execution was `b1f880ebfbec99f59fad38b966c51346cf0bb496b3d31486be71a4e6444b1cc9`.
The machine-readable gate result is
`runs/integrity/pff_grf_position_only_diagnostic_v1_gate_summary.json`.

## Data Integrity

The projected PFF manifest has payload SHA-256
`37acb8a6a00e4842a8aef8dce2700417fd7dfa24c827c3a9f46c7dac782c24ae` and contains:

- train: 844,195 examples
- validation: 141,054 examples
- total: 985,249 examples in 1,766 shards from 56 matches
- test: zero projected shards and zero loaded tensor files
- ordered channels: `x_norm`, `y_norm`, `is_ball`, `is_home`, `is_away`

All 1,766 projected shards map one-to-one to a selected source tensor hash and preserve source
split, match, period, and example-count metadata. The projector mechanically preserves sample
identities, masks, timing indices, and coordinates while removing only velocity channels.

## Runs

Every PFF run used 100 training batches and 50 validation batches. Automatic embedding export was
disabled, and each run manifest records only `train` and `val` as loaded tensor splits.

| Seed | Scratch run | Scratch total | GRF run | GRF total | Relative change |
| --- | --- | ---: | --- | ---: | ---: |
| 7 | `20260714_223720` | 0.022467 | `20260714_223748` | 0.019443 | -13.5% |
| 11 | `20260714_223816` | 0.022053 | `20260714_223842` | 0.019267 | -12.6% |
| 23 | `20260714_223910` | 0.016386 | `20260714_223936` | 0.013222 | -19.3% |
| Mean | | 0.020302 | | 0.017311 | -14.7% |

## Frozen Gate

| Criterion | Result | Decision |
| --- | ---: | --- |
| finite metrics for all six runs | yes | pass |
| transfer total-loss wins | 3 of 3 | pass; minimum 2 |
| mean total-loss improvement | 14.735% | pass; minimum 5% |
| mean narrow TD relative change | -21.209% | pass; maximum 0% |
| minimum `z_online_std_mean` | 0.239 | pass; must exceed 0.05 |

The position-only transfer benefit is therefore not carried solely by explicit provider velocity
channels. Relative to the earlier seven-channel 100-update result, the mean total-loss improvement
is larger (14.7% versus 10.3%), while the narrow-TD improvement is slightly smaller (21.2% versus
23.5%). These are descriptive relative-effect comparisons; absolute losses are not comparable
across feature dimensions.

## Decision

This result authorizes the separately frozen 2,000-update position-only validation repeat. It does
not authorize PFF test access, tactical interpretation, model selection claims, or downstream use.
The current evidence remains an early-optimization result until the longer repeat is complete.
