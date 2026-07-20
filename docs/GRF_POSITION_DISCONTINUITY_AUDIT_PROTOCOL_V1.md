# GRF Position Discontinuity Audit Protocol V1

Status: frozen on 2026-07-13 before implementation and result computation.

## Purpose

The provider-neutral causal-position redesign is blocked because mean sampled GRF player
acceleration rose from `7.4006` to `22.9713 m/s^2`, while its p99 remained nearly unchanged and
its standard deviation rose to `226.77 m/s^2`. This pattern is consistent with rare finite-
difference spikes, but their source has not been established.

This audit locates those spikes in immutable raw GRF training episodes before any further data
transformation or model run.

## Frozen Inputs

- collection manifest: `data/raw/gfootball/v2_pilot/collection_manifest.json`
- collection-plan canonical SHA-256:
  `cba4b38f44ed78dce9b14fcf8d67cb2170552952ae9b44195ec29e0b97d7ee90`
- split manifest: `splits/gfootball_v2_pilot_episode_split.json`
- split-manifest canonical SHA-256:
  `55b5db0bb003ee3ee4b11180903403ddf1da3df0e22e451311054cca58368e71`
- scope: the ten collection jobs marked `train` and only episode IDs listed in
  `train_match_ids`
- each raw JSONL file must match the SHA-256 stored in the collection manifest

The audit must reject any validation/test job, path, record, or episode. It does not read PFF data.

## Frozen Measurements

Use raw positions converted by the existing GRF-to-metre coordinate conversion. For each entity
slot within one episode, require adjacent frame IDs and positive adjacent timestamps.

- position jump: Euclidean distance between frames `t-1` and `t`
- causal velocity: position difference divided by elapsed time
- causal acceleration: Euclidean difference between causal velocities at `t-1` and `t`, divided
  by elapsed time
- player extreme jump: at least `3.0 m` in one adjacent 10 Hz step
- player extreme causal speed: at least `12.0 m/s`
- player extreme causal acceleration: at least `100.0 m/s^2`
- ball extreme jump: at least `10.0 m`
- ball extreme causal speed: at least `60.0 m/s`
- ball extreme causal acceleration: at least `300.0 m/s^2`

Report counts, rates, mean, standard deviation, p50, p95, p99, p99.9, and maximum globally and by
job/scenario. Retain the 100 largest player-acceleration and 50 largest ball-acceleration records,
including episode, frames, entity slot, positions, velocities, active flags, game modes, score,
and steps remaining.

## Frozen Event Attribution

For each extreme acceleration record, inspect the inclusive `t-5` through `t+5` frame window in
the same episode. Record independently whether that window contains:

- a `game_mode` value change
- any nonzero `game_mode`
- a score change

An extreme is `event_proximate` when any of those conditions is true. It is `jump_associated` when
either of its two velocity-producing position jumps crosses the corresponding frozen jump
threshold.

Compute both record-count shares and acceleration-mass shares, where acceleration mass is the sum
of extreme acceleration magnitudes.

## Frozen Decision Rule

1. If at least 80% of extreme player-acceleration mass is event-proximate, the next permitted
   candidate is event-boundary segmentation or masking.
2. Otherwise, if at least 80% is jump-associated, the next permitted candidate is a generic
   jump-boundary mask, with its threshold frozen independently before rebuilding tensors.
3. Otherwise, no reset-aware causal-position redesign is authorized; retain provider velocities
   while investigating another source of the motion gap.

This audit selects a preprocessing hypothesis only. Any candidate still requires invariant checks
and the corrected provider-neutral V2 train-only preflight before model training.

## Claim Boundary

The result cannot support claims about representation quality, validation performance, tactical
concepts, tactical surprise, emergent understanding, or downstream value. GRF validation/test and
all PFF validation/test data remain untouched.
