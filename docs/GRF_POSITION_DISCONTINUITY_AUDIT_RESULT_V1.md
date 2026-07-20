# GRF Position Discontinuity Audit Result V1

Status: complete on 2026-07-13. The frozen decision selects
`event_boundary_segmentation_or_masking` as the next preprocessing candidate.

## Integrity Boundary

The audit read all and only the ten raw GRF jobs marked `train`: 92 episodes and 69,773 frames.
Every source file matched the SHA-256 in the immutable collection manifest, every record carried a
train split/job identity, and the observed episode set exactly matched the split manifest's train
IDs. No GRF validation/test job and no PFF source was read.

Machine-readable result:
`runs/integrity/grf_position_discontinuity_audit_v1.json`.

## Main Result

Across 1,427,619 three-frame player transitions, causal position differences produce:

| Statistic | Player acceleration |
| --- | ---: |
| Mean | 23.6252 m/s^2 |
| Median | 4.5679 m/s^2 |
| p95 | 21.5618 m/s^2 |
| p99 | 30.0404 m/s^2 |
| p99.9 | 4,343.8393 m/s^2 |
| Maximum | 7,868.2454 m/s^2 |

There are 10,478 extreme player accelerations at or above the frozen `100 m/s^2` threshold,
representing 0.734% of measured transitions.

- 100% of extreme records and 100% of extreme acceleration mass are within five frames of a
  recorded game-mode or score event
- 96.125% of extreme records and 99.671% of extreme mass contain an adjacent player jump of at
  least 3 metres
- 76.216% of extreme mass is near a transition into or out of a nonzero game mode
- 23.784% is near a score change
- the largest retained examples relocate a player by roughly 70-78 metres in one 0.1-second frame

All extreme player records occur in the four full-match jobs. None occur in the six academy jobs.
The ball has 311 extreme accelerations at or above `300 m/s^2`; 98.322% of their mass is also
event-proximate.

## Interpretation

The raw evidence confirms that simulator event repositioning causes the rare catastrophic
finite-difference spikes. It is not an episode-boundary artifact because motion is differentiated
only inside one episode with adjacent frame IDs. Score changes and nonzero game modes identify all
extreme player records under the frozen event window.

This explains why the unsmoothed provider-neutral reconstruction produced a huge mean and standard
deviation and why the defective V1 gap score appeared to pass. It does not explain away the full
motion-domain mismatch: median and p95 player acceleration remain far above PFF even below the
catastrophic tail. Event masking is therefore a necessary repair for causal reconstruction, not a
complete solution or evidence that the corrected domain-gap gate will pass.

## Frozen Decision

The protocol selects event-boundary segmentation or masking because event-proximate mass exceeds
the frozen 80% threshold. The next candidate should exclude transitions around recorded score and
game-mode boundaries before calculating causal velocities and before building temporal examples.
Its exact mask/window and sample-identity consequences must be frozen before implementation.

After rebuilding, the candidate must repeat train-identity checks and the corrected provider-
neutral V2 preflight. No model training is authorized by this audit.

## Claim Boundary

This result is a raw train-data preprocessing diagnosis. It is not evidence about representation
quality, validation performance, tactical concepts, tactical surprise, emergent understanding, or
downstream value.
