# GRF Provider-Neutral Motion Protocol V2

Status: frozen on 2026-07-13 after detecting the V1 scorer defect and before corrected rescoring.

## Reason For V2

The V1 preflight used a nominally robust gap score, but its implementation allowed one quarter of
the pooled standard deviation to replace the pooled interquartile scale. Rare, extremely large
candidate acceleration values therefore enlarged the denominator and made the gap appear smaller.
The resulting V1 formal pass is invalid and does not authorize model training.

V2 changes only score normalization and adds direct physical sanity checks. It does not change the
provider-neutral tensors, frozen samples, train-only inputs, or candidate transformation.

## Frozen Corrected Score

- continuous metrics use the mean of the real and synthetic interquartile ranges as their scale
- if that scale is zero, use the corresponding pooled median-absolute-deviation scale
- standard deviation is never used to reduce a gap score
- binary-rate metrics use the fixed probability range `1.0`
- quantile Wasserstein distance, sampling, metrics, seed, and scenario caps remain unchanged
- both the original GRF baseline and provider-neutral candidate are rescored by this implementation

## Frozen Train-Only Gate

The candidate passes only when every condition holds:

1. train example identities, match/period/frame identities, frame indices, x/y values, masks,
   splits, collection hashes, and visibility-profile hashes match the original GRF V2 tensors
2. velocity values differ, confirming that the requested transformation was applied
3. corrected player-acceleration gap score is below `1.0`
4. corrected player-acceleration gap score falls by at least 25% from the corrected original-GRF
   score computed with the same frozen inputs
5. candidate mean player acceleration does not exceed the original-GRF value
   `7.400551014128501 m/s^2`
6. candidate player-acceleration p99 does not exceed the original-GRF value
   `30.639189758300784 m/s^2`
7. corrected player-turn gap does not exceed 110% of its corrected original-GRF score
8. corrected player-speed gap does not exceed 110% of its corrected original-GRF score
9. neither audit nor invariant comparison reads a validation or test tensor

Failure blocks this redesign without model training. Passing permits a separate model-training
protocol; it does not itself authorize training or establish representation quality.

## Claim Boundary

This is an integrity repair and train-domain preprocessing check. It cannot support claims about
tactical concepts, tactical surprise, emergent understanding, validation performance, or
real-world downstream value. PFF validation and test remain untouched.
