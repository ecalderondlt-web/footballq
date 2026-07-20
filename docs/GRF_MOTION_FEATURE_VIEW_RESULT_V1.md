# GRF Motion Feature-View Result V1

Status: position-only selected on 2026-07-14 for a future frozen model protocol. Model training is
not authorized by this result alone.

## Shared Candidate Data

Actual-jump segmentation uses the frozen `3 m` player and `10 m` ball thresholds. A detected jump
starts a new temporal segment, but no surrounding frames are removed. Both feature candidates use
the exact same samples and are train-only.

- original GRF train examples: 33,591
- retained candidate examples: 30,134
- retention: 89.709%
- raw train frames retained: 69,773/69,773
- unique jump-boundary frames: 240
- temporal segments: 332
- examples crossing a boundary: 0
- validation/test examples built or read: 0

Every retained sample ID is from the original manifest. Match/period/frame identities, temporal
indices, x/y coordinates, and visibility masks match exactly. The integrity gate passes for both
candidates.

## Candidate B: 0.5-Second Motion

Manifest payload SHA-256:
`a79c45e5ea8894bb58b8336c313d70f6767c4544b73dd89db83fa1e331fa8a76`.

The five-frame causal lag improves several physical summaries but fails the robust acceleration
gate:

| Criterion | Candidate | Frozen limit | Result |
| --- | ---: | ---: | --- |
| Player-acceleration gap | 1.5434 | below 1.0 | fail |
| Gap change from baseline | -6.379% | at least +25% | fail |
| Mean player acceleration | 5.2646 m/s^2 | at most 7.4006 | pass |
| Player-acceleration p99 | 16.3089 m/s^2 | at most 30.6392 | pass |
| Player-turn gap | 1.0393 | at most 1.3913 | pass |
| Player-speed gap | 0.7648 | at most 0.9123 | pass |
| Example retention | 89.709% | at least 75% | pass |

The mean is lower, but the corrected robust score compares the full quantile shape relative to
pooled interquartile spread. That distribution remains unlike PFF and scores worse than the
original provider-velocity baseline. The 0.5-second view is therefore blocked.

## Candidate A: Position-Only

Manifest payload SHA-256:
`88eebc0646f2ca50d2e8d12945e86635487fc39aa7f789af70f36765239e7e7c`.

The position-only candidate is a mechanical projection of the accepted 0.5-second tensors and has
exactly these ordered channels: `x_norm`, `y_norm`, `is_ball`, `is_home`, and `is_away`. It contains
no velocity or possession-derived channel. Sample identities, masks, coordinates, temporal indices,
and boundary handling are identical to the lagged candidate.

Position-only passes its frozen integrity and retention requirements. Under the frozen selection
rule it is selected for a separately frozen matched model comparison because the lower-frequency
motion candidate failed its motion gate.

Machine-readable selection:
`runs/integrity/grf_motion_feature_view_selection_v1.json`.

## Interpretation

Longer causal differencing reduces acceleration magnitude but does not align the full GRF motion
distribution with PFF. The safer next test is to omit explicit velocity and let the temporal encoder
infer movement from position sequences.

Selection does not show that position-only improves representation learning. The next experiment
must project the matching PFF train/validation tensors, freeze identical seeds and update budgets,
and compare against the current seven-channel reference. PFF test must remain untouched.

## Claim Boundary

This result selects an input feature view. It is not evidence of validation improvement, tactical
concepts, tactical surprise, emergent understanding, or downstream value.
