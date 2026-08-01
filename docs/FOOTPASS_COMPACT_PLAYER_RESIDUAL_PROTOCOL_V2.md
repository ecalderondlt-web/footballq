# FOOTPASS Compact Player-Residual Protocol V2

## Status and Motivation

This is a development-only redesign informed by the failed V1 development
experiment. V1 found that a large player-history block improved over a weak
rolling baseline, but did not beat role-mean and all shuffled-identity
controls. Geometry plus role remained the strongest view.

V2 therefore asks a narrower question:

> Does a strongly shrunk, compact summary of a specific player's strictly
> earlier matches improve prediction beyond current geometry, current role,
> and generic same-team role history?

The V1 validation outcomes have already been seen and are not described as a
fresh test. V2 adds predeclared internal match-group cross-validation as a
stability requirement. FOOTPASS confirmation matches 22, 40, and 43 remain
the only untouched outcome evaluation.

## Data and Chronology

- Source: the immutable V1 development extraction cache.
- Source HDF5 hash:
  `bcc02cd2f05509d1e82ba16a81ada5349895410a349aeedcb13e539339379058`.
- Split: `splits/footpass_player_history_chronological_v1.json`.
- Identity manifests: Bayern, Napoli, and Lazio V1 manifests.
- Support-only matches: 6 and 10.
- Development fit: 2, 11, 12, 13, 27, 28, 32, 35, and 38.
- Development validation: 14, 15, and 33.
- Confirmation: 22, 40, and 43.

For every query, history includes all available focal-team appearances with a
verified date strictly earlier than the query. The query match and all later
matches are excluded. Sample IDs remain period-aware.

The V2 runner must refuse any cache containing a confirmation match ID and
does not implement confirmatory unsealing.

## Targets

Targets and opportunity construction are unchanged from V1:

1. Primary: a focal-team action-location proxy inside the attacking penalty
   area within ten seconds and before the opponent's next possession action.
2. Secondary: the first subsequent possession action within five seconds is
   by the opponent.

FOOTPASS has no ball coordinates. The primary target is not a directly
observed ball entry.

## Compact History

Each prior player-match is summarized using 28 behavioral values:

- four action-group probabilities: drive, pass-like, shot, and defensive;
- event means for attacking x, y, attacking vx, vy, and speed;
- event standard deviations for attacking x, y, and speed;
- prior turnover and penalty-area-action rates;
- nine off-ball tracking means;
- five off-ball tracking standard deviations.

The player profile is the equally weighted mean of all strictly earlier
player-match summaries. Equal match weighting prevents one high-event match
from dominating the profile.

For each current role, a generic prior is built from other active focal-team
players in the same broad role. The focal player is excluded. If that pool is
empty, the prior falls back to other active focal-team players and then to the
neutral smoothed profile.

The player-specific feature is the deviation from that generic role prior:

`shrunk_deviation = m / (m + lambda) * (player_profile - role_prior)`

where `m` is the number of strictly earlier matches available for the player.
The frozen main shrinkage is `lambda = 3`. Sensitivities use 1 and 5. Missing
history produces a zero deviation with explicit support and availability
features.

## Feature Views

1. `geometry_role`: current geometry, action class, and tactical role.
2. `role_context`: `geometry_role` plus the compact generic role prior.
3. `player_residual`: `role_context` plus the shrunk focal-player deviation,
   support count, availability, and shrinkage weight.
4. `player_residual_lambda1` and `player_residual_lambda5`: fixed shrinkage
   sensitivities.
5. `player_residual_unshrunk`: leakage-free but deliberately unregularized
   profile sensitivity.
6. `shuffled_player_residual_seed_*`: same-team, broad-role donor residuals
   with seeds 7, 11, 23, 41, and 73.
7. `geometry_role_identity`: static-identity control.

The main scientific comparison is `player_residual` versus `geometry_role`.
The player view must also beat `role_context` and every shuffled-player
control. This prevents generic team or role history from being mislabeled as
specific-player knowledge.

## Probe

- Deterministic unweighted binary logistic regression.
- Train-only centering and scaling.
- L2 penalty `0.01 * 0.5 * sum(weight^2)`.
- CPU LBFGS, maximum 250 iterations.
- No validation-tuned regularization, feature selection, or class weighting.

## Development Evaluation

The original development split is retained for comparability:

- fit on the nine development-fit matches;
- evaluate on matches 14, 15, and 33.

Three additional predeclared match-group folds cover every fit match once:

- fold A validation: 2, 11, and 28;
- fold B validation: 27, 32, and 38;
- fold C validation: 12, 13, and 35.

Each fold trains on the other six development-fit matches. These folds are
development stability checks, not untouched temporal tests.

## V2 Development Gate

All checks must pass:

1. Primary relative NLL improvement over `geometry_role` is at least 0.5%.
2. The 95% match-period blocked-bootstrap lower bound is above zero.
3. Primary NLL gain is positive in at least two of three V1 validation
   matches.
4. `player_residual` has lower primary NLL than `role_context`.
5. `player_residual` has lower primary NLL than every shuffled-player
   residual.
6. Primary Brier score is no worse than `geometry_role`.
7. Secondary relative NLL change versus `geometry_role` is no worse than
   -0.5%.
8. Pooled internal-CV primary NLL gain versus `geometry_role` is positive.
9. Internal-CV primary NLL gain is positive in at least two of three folds.
10. All chronology, identity, finite-value, and cache-sealing audits pass.

Failure is a valid null result and leaves confirmation sealed. Passing this
gate establishes only eligibility to create a separate confirmatory freeze;
this V2 development runner never opens confirmation.
