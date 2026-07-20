# GRF Position-Only Volume Scaling Results V1

## Outcome

The frozen primary gate is **blocked**. All data and training integrity checks passed, but the 8x
family did not reach either prespecified materiality threshold:

- 8x improved mean final PFF total validation loss by 2.62% versus scratch; the threshold was 5%.
- 8x improved mean total loss by 0.87% versus equal-compute 1x replay; the threshold was 2%.

Blocked means the volume hypothesis did not pass its test. It does not mean model training failed
or that simulation had no effect.

## Final Validation

Each value is the mean across paired seeds 7, 11, and 23 after exactly 2,000 PFF updates. Final
evaluation used 500 validation batches, or 64,000 examples, per run.

| Family | Synthetic examples | Synthetic updates | Mean total loss | Change vs scratch | Mean narrow TD loss | TD change vs scratch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch | 0 | 0 | 0.0130860 | reference | 0.00015769 | reference |
| 1x | 29,453 | 263 | 0.0128448 | -1.84% | 0.00014379 | -8.82% |
| 1x replay | 29,453 | 2,104 | 0.0128542 | -1.77% | 0.00014442 | -8.42% |
| 4x | 120,177 | 1,052 | **0.0127415** | **-2.63%** | **0.00011615** | **-26.35%** |
| 8x | 240,337 | 2,104 | 0.0127426 | -2.62% | 0.00013743 | -12.85% |

Lower is better. The 4x and 8x mean total losses differ by only 0.009% relative to 8x, with 4x
fractionally lower. The 8x family beat scratch total loss in two of three paired seeds and passed
the narrow-TD and latent-spread safeguards.

## Frozen Gate

| Criterion | Result | Status |
| --- | --- | --- |
| all 15 PFF runs finite | all finite | pass |
| 8x total wins vs scratch | 2 of 3 | pass |
| 8x mean total improvement vs scratch | 2.62%, minimum 5% | **block** |
| 8x mean TD no worse than scratch | 12.85% lower | pass |
| every final latent spread above 0.05 | minimum 0.456 | pass |
| 8x mean total improvement vs replay | 0.87%, minimum 2% | **block** |
| 8x mean TD no worse than replay | 4.84% lower | pass |

## Learning Curves

The values below are descriptive means from the first 50 validation batches. They did not select
checkpoints or change the final decision.

| Updates | scratch | 1x | 1x replay | 4x | 8x |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.020302 | 0.017130 | 0.016703 | 0.017028 | **0.016405** |
| 250 | 0.016389 | 0.015673 | **0.013833** | 0.015418 | 0.014584 |
| 500 | 0.016759 | 0.015997 | **0.014739** | 0.015702 | 0.015410 |
| 1,000 | 0.013631 | 0.013227 | **0.012900** | 0.013062 | 0.013786 |
| 2,000 | 0.014059 | 0.013795 | 0.013841 | 0.013609 | **0.013597** |

Simulation provides an early optimization advantage, but more unique volume does not produce a
clean monotonic dose response. The equal-compute replay control is often competitive early, and
the final 4x and 8x means are effectively tied.

## Data Integrity

- The 1x, 4x, and 8x datasets contain 29,453, 120,177, and 240,337 unique train examples.
- Retention after frozen jump segmentation is 88.59%, 89.39%, and 89.48%, above the 75% floor.
- Full-field tensor nesting passed for 1x in 4x, 4x in 8x, and 1x in 8x.
- Boundary-crossing and unsafe-frame references are zero.
- Synthetic tensors are train-only. PFF runs loaded only train and validation tensors.
- Embedding export was disabled, and the PFF test split was not loaded or evaluated.

Machine-readable evidence:

- `runs/grf_position_scale_v1/gate_summary.json`
- `runs/grf_position_scale_v1/execution_manifest.json`
- `runs/integrity/gfootball_position_scale_v1_tensor_nesting_audit.json`
- `runs/integrity/gfootball_position_scale_v1_run_access_audit.json`

## Conclusion

Do not discard GRF: every simulation family improved mean final total loss over scratch, and the
4x family gave the best mean total and narrow-TD results. However, do not scale this same scenario
mixture further on the evidence here. The gain saturates by 4x and remains below the frozen
materiality bar.

The next simulation experiment should change diversity or realism rather than raw episode count:
stronger and more varied policies, longer tactical phases, score/time conditioning, human-like
speed and action distributions, and scenario coverage selected from measured GRF-to-PFF gaps.
Real tracking remains the final grounding source. No tactical or semantic claim follows from this
validation-loss experiment.
