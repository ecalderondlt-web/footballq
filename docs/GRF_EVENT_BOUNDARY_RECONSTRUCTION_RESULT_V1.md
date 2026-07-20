# GRF Event-Boundary Reconstruction Result V1

Status: `blocked` on 2026-07-13. No model training is authorized.

## Candidate

The candidate removes frames within five frames of a nonzero or changing GRF game mode or a score
change. Retained frames are partitioned into consecutive temporal segments. Causal velocity is
calculated only inside a segment, and a TD-JEPA example is retained only when its full temporal
extent belongs to one segment and its original episode stride phase is preserved.

Only the ten GRF train jobs were read. The resulting manifest is:

`data/processed/gfootball_v2_event_segmented_v1/pff_train_observed_visibility_v1/dataset_manifest.json`

Manifest payload SHA-256:
`93ba1dfa07c2590e644b50f9c818e61836d8ea17d093f16d33cb0be9b0015893`.

## Integrity Result

- original train examples: 33,591
- retained train examples: 16,785
- retention: 49.969%
- total raw train frames: 69,773
- unsafe frames removed: 3,758
- retained raw frames: 66,015 (94.614%)
- retained temporal segments: 331
- unsafe frame references in tensors: 0
- validation examples built/read: 0
- test examples built/read: 0

Every retained sample ID is an original train sample ID. Match/period/frame identities, context and
target frame indices, x/y values, and visibility masks match exactly. There are no new or duplicate
IDs. Velocity reconstruction changes 4,882,826 retained tensor values.

Although only 5.386% of raw frames are removed, the 3-second context/gap/target footprint means one
event boundary invalidates many possible windows. The candidate therefore fails the frozen 75%
example-retention floor.

## Corrected Motion Gate

| Criterion | Candidate | Limit | Result |
| --- | ---: | ---: | --- |
| Player-acceleration gap | 1.4281 | below 1.0 | fail |
| Acceleration-gap reduction | 1.563% | at least 25% | fail |
| Mean player acceleration | 7.0095 m/s^2 | at most 7.4006 | pass |
| Player-acceleration p99 | 27.3842 m/s^2 | at most 30.6392 | pass |
| Player-turn gap | 1.2586 | at most 1.3913 | pass |
| Player-speed gap | 0.8471 | at most 0.9123 | pass |
| Train example retention | 49.969% | at least 75% | fail |

The candidate contains only 16,785 contexts, so the audit's frozen 24,576-context cap resolves to
16,785 shared contexts per source. All 48 PFF train matches remain represented. The machine result
is `runs/integrity/grf_event_segmented_preflight_v1.json`.

## Interpretation

Event segmentation repairs the pathological tail: mean sampled GRF acceleration falls from
`22.9713` to `7.0095 m/s^2`, and p99 also improves. This confirms that score/game-mode repositioning
was the cause of the catastrophic causal finite differences.

It does not solve the typical motion mismatch. The robust acceleration gap changes only from
`1.4508` to `1.4281`, far below the prespecified improvement. It also discards half of the eligible
temporal examples despite retaining almost 95% of frames. The dataset is therefore unsuitable for
the next model comparison in its current form.

An initial engineering build restarted stride phase inside each segment and created non-baseline
sample IDs. The subset invariant caught that build before domain scoring. The final candidate above
uses episode-level stride phase and is the only authoritative result.

## Next Scientific Step

Do not tune the event window against this result. The next protocol should compare train-only
position-only or lower-frequency motion feature views that preserve temporal examples while
addressing the broad acceleration mismatch. Event-boundary exclusion should remain an integrity
rule for any causal finite-difference view, but it is not itself a sufficient domain adaptation.

## Claim Boundary

This result concerns train-data preprocessing. It is not evidence about model quality, validation
performance, tactical concepts, tactical surprise, emergent understanding, or downstream value.
