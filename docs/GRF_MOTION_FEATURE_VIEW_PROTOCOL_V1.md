# GRF Motion Feature-View Protocol V1

Status: frozen on 2026-07-14 before implementation, candidate construction, or result computation.

## Purpose

The five-frame event-boundary candidate repairs simulator-reset acceleration spikes but retains only
49.969% of temporal examples and reduces the robust acceleration gap by only 1.563%. This protocol
tests whether a narrower observed-jump boundary and a less provider-specific feature view can
preserve training examples without reintroducing discontinuities.

## Frozen Inputs And Scope

- original balanced-GRF manifest payload SHA-256:
  `f8d524157196ab007c21c0662a15b18e9458b2468b5a2af349895cec4bb981db`
- collection-plan canonical SHA-256:
  `cba4b38f44ed78dce9b14fcf8d67cb2170552952ae9b44195ec29e0b97d7ee90`
- episode split-manifest SHA-256:
  `55b5db0bb003ee3ee4b11180903403ddf1da3df0e22e451311054cca58368e71`
- PFF-train visibility-profile payload SHA-256:
  `3bd3e96d0c449e3f6a57e69a37001af71e863b82ca7613b3ba7022738280cd40`
- only GRF train jobs and existing PFF train tensors may be read
- no GRF validation/test job, PFF validation/test tensor, or model outcome may be read

## Frozen Actual-Jump Boundary

Within each episode, period, and entity slot, compare adjacent frame IDs:

- player discontinuity: position jump at least `3.0 m`
- ball discontinuity: position jump at least `10.0 m`

If any entity crosses its threshold, the current frame begins a new temporal segment. No surrounding
frame window is removed. Causal motion is calculated independently inside each segment, and a
temporal example is retained only when its entire context/gap/target footprint lies in one segment.
Original episode stride phase and sample IDs must be preserved.

## Candidate A: Position-Only

The frozen ordered feature channels are:

1. `x_norm`
2. `y_norm`
3. `is_ball`
4. `is_home`
5. `is_away`

This view is projected mechanically from the exact accepted lower-frequency tensors below, so
sample IDs, masks, coordinates, splits, and temporal indices are identical between candidates.
No velocity or possession-derived channel is present.

## Candidate B: 0.5-Second Causal Motion

The frozen ordered channels remain the seven-channel `geometry_only` view. For each entity at frame
`t`, velocity is displacement from the earliest available frame up to five frames in the past,
divided by elapsed time. Once five prior frames exist, the lag remains exactly five frames
(`0.5 s` at 10 Hz). The first frame in a temporal segment receives zero velocity. No future frame,
smoothing, clipping, interpolation, or learned calibration is permitted.

## Frozen Integrity And Retention Gate

Both candidates must satisfy:

- train sample IDs are a strict subset of the original 33,591 IDs
- no new or duplicate ID
- retained match/period/frame identities, temporal indices, x/y values, and masks match exactly
- no sample crosses an actual-jump segment boundary
- at least 75% of original train examples remain
- candidate A has exactly the five frozen position-only channels
- candidate B has exactly the seven frozen geometry channels and changed velocity values
- manifests report boundary frames and temporal segments by job
- all paths are train-only

## Frozen Lower-Frequency Motion Gate

Candidate B is compared with PFF train under the corrected audit and the existing frozen limits:

- player-acceleration gap below `1.0`
- at least 25% reduction from `1.4508298714`
- mean player acceleration at most `7.4005510141 m/s^2`
- player-acceleration p99 at most `30.6391897583 m/s^2`
- player-turn gap at most `1.3912653342`
- player-speed gap at most `0.9123399937`

Use the same seed, 24,576-context cap, all 48 PFF train matches, four real shards per match, and
5,000-context scenario cap. Candidate A has no velocity-domain gate because velocity is absent.

## Frozen Selection Rule

1. If candidate B passes integrity, retention, and every corrected motion criterion, select the
   0.5-second view for a separately frozen short model comparison.
2. Otherwise, if candidate A passes integrity and retention, select position-only for that future
   comparison. This is an authorization to test it, not evidence that it improves a model.
3. If neither passes, stop without model training.

## Claim Boundary

This comparison can select a train-data feature view only. It cannot support claims about model
quality, validation performance, tactical concepts, tactical surprise, emergent understanding, or
downstream value.
