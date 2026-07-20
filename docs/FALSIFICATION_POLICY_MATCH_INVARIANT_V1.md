# Match-Invariant Falsification Policy v1

Status: frozen for the first held-out test evaluation on 2026-07-10.

This policy was designed from validation diagnostics and is therefore exploratory on
validation. It must not be changed after inspecting test results. The first held-out
test application is confirmatory for this exact policy and selected model family.

## Selected Model Family

The selected family is the three-seed GRF-initialized temporal-transition TD-JEPA:

- seed 7: `runs/td_jepa/20260710_165038/best.pt`
- seed 11: `runs/td_jepa/20260710_165753/best.pt`
- seed 23: `runs/td_jepa/20260710_170316/best.pt`

No checkpoint, loss weight, threshold, or metric may be changed before the test
evaluation.

## Condition Metrics

- shuffled future, wrong-match future, reversed time, and pitch reflection: `td_loss`
- masked ball: `ball_dynamic_reconstruction_loss`
- no-motion predictor: `total_loss`
- team/label/within-team player-slot permutations: `td_loss`

The ball metric uses only the ball slot and dynamic `x,y,vx,vy` channels. The no-motion
metric includes the explicitly trained future-transition reconstruction. Other causal
corruptions use the narrow latent prediction error.

## Expectations And Thresholds

Higher-than-correct conditions pass when the minimum ratio across seeds is at least
`1.25`, are caution from `1.05` to below `1.25`, and fail below `1.05`.

Player-slot, team-slot, and home/away-label transformations are symmetry checks for a
global match-invariant representation. They pass when every seed ratio lies in
`[0.80, 1.25]`, are caution within `[0.50, 1.50]`, and fail outside that range.
Requiring these transformations to increase loss would reward arbitrary tensor-slot
memorization and conflict with the match-invariance objective.

Every condition is blocking under this policy. The legacy `legacy_higher_v1` policy
and its artifacts remain unchanged for comparison.

## Claim Boundary

A pass means only that the selected representation satisfies this predefined control
battery on held-out tracking windows. It does not establish semantic or tactical
understanding and does not by itself authorize interpretation or annotation. Probe and
discovery controls remain separate required gates.

## First Held-Out Test Result

The frozen policy was applied once to the untouched test split with no intervening
model, metric, threshold, or policy changes. The three-seed aggregate is stored at
`runs/td_jepa/redesign_transfer_transition_seed7_11_23_falsification_gate_match_invariant_v1_test/`
and reports `controls_passed` with no blocking conditions.

Minimum ratios across seeds are `1.513x` for masked-ball dynamic reconstruction,
`1.299x` for no-motion total loss, `1.365x` for reversed-time TD loss, `4.716x` for
shuffled-future TD loss, and `4.129x` for pitch-reflection TD loss. Every symmetry
condition remains inside `[0.80, 1.25]`; the widest is target player-slot permutation
at a maximum ratio of `1.227x`.

This confirms the frozen falsification policy only. Incremental probes and discovery
controls were subsequently completed and remain blocking, as recorded in
`runs/integrity/grf_transition_v1_gate_summary.json`.
