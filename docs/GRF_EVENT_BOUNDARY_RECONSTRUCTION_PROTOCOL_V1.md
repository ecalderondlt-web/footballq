# GRF Event-Boundary Reconstruction Protocol V1

Status: frozen on 2026-07-13 before implementation, rebuilding, and result computation.

## Motivation

The frozen raw train-only discontinuity audit found that 100% of extreme player-acceleration mass
is within five frames of a score or game-mode event. The provider-neutral causal-position candidate
must therefore prevent simulator event repositioning from entering velocity construction or
temporal examples before its motion-domain gate is evaluated again.

## Frozen Inputs And Scope

- balanced GRF V2 collection-plan canonical SHA-256:
  `cba4b38f44ed78dce9b14fcf8d67cb2170552952ae9b44195ec29e0b97d7ee90`
- episode split-manifest SHA-256:
  `55b5db0bb003ee3ee4b11180903403ddf1da3df0e22e451311054cca58368e71`
- PFF-train visibility profile payload SHA-256:
  `3bd3e96d0c449e3f6a57e69a37001af71e863b82ca7613b3ba7022738280cd40`
- only the ten GRF jobs marked `train` are read and rebuilt
- GRF validation/test raw files and all PFF tensors remain unread
- geometry-only, future-nonoverlap configuration and visibility seeds remain unchanged

## Frozen Event Rule

Add raw home score, away score, and steps-remaining metadata to canonical GRF rows. Within each
episode and period, a frame is an event signal when any condition holds:

- `game_mode` is nonzero
- `game_mode` differs from the preceding frame
- home or away score differs from the preceding frame

Mark a frame unsafe when its frame ID is within five frames, inclusive, of any event-signal frame
in the same episode and period. Remove unsafe frames before causal velocity construction and before
temporal example construction.

Partition retained frames into maximal runs of consecutive frame IDs. Causal x/y velocity is
computed independently inside each run. A context/target example is permitted only when all of its
frames lie inside one retained run. No smoothing, clipping, interpolation, learned calibration, or
validation-fit parameter is permitted.

## Frozen Identity And Data-Loss Gate

- candidate train sample IDs must be a strict subset of the original balanced-GRF train IDs
- no new or duplicate sample ID is permitted
- for every retained ID, match/period/frame identity, context and target frame indices, x/y values,
  and visibility masks must exactly match the original tensors
- velocity channels must differ, confirming reconstruction occurred
- no tensor may reference an unsafe frame
- at least 75% of the original 33,591 train examples must remain
- manifest metadata must report signal, unsafe, retained-frame, and segment counts by job

## Frozen Corrected Motion Gate

Rerun the corrected train-only GRF-to-PFF audit with the same 24,576-context cap, all 48 PFF train
matches, four PFF shards per match, seed `20260713`, and scenario cap `5,000`.

The candidate passes only if every condition holds:

1. identity/data-loss gate passes
2. player-acceleration gap is below `1.0`
3. player-acceleration gap falls by at least 25% from corrected original-GRF score `1.4508298714`
4. mean player acceleration does not exceed `7.4005510141 m/s^2`
5. player-acceleration p99 does not exceed `30.6391897583 m/s^2`
6. player-turn gap does not exceed `1.3912653342`
7. player-speed gap does not exceed `0.9123399937`
8. audit and invariant paths are train-only

Failure blocks this candidate without model training. Passing permits a separately frozen model
comparison; it does not itself authorize training.

## Claim Boundary

This is a train-data preprocessing test. It cannot support claims about validation performance,
representation quality, tactical concepts, tactical surprise, emergent understanding, or
downstream value.
