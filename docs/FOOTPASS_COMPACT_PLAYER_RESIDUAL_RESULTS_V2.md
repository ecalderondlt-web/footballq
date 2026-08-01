# FOOTPASS Compact Player-Residual Results V2

## Status

- Development gate: failed.
- Claim status: development-only null result.
- Confirmation eligibility: no.
- Confirmation matches 22, 40, and 43: sealed and unread.
- Confirmatory freeze or eligibility artifact created: no.

V2 was designed after the V1 development result and therefore does not treat
the V1 validation matches as a fresh test. It adds three predeclared internal
match-group folds as a stability check.

## Question

Does a compact, strongly shrunk summary of the focal player's strictly earlier
matches improve action-outcome prediction beyond:

- current geometry and role;
- generic same-team role history; and
- same-role shuffled-player controls?

## Data and Integrity

- Total development opportunities: 11,908.
- Development-fit opportunities: 9,049.
- V1 validation opportunities: 2,859.
- Compact behavior dimensions: 28.
- Mean strictly earlier matches per actor: 2.14.
- Median strictly earlier matches per actor: 2.
- Maximum strictly earlier matches per actor: 5.
- Missing actor history: 687 opportunities (5.77%).
- Mean main shrinkage weight: 0.379.
- Mean leave-one-player-out role-prior peers: 2.42.
- Chronology violations: 0.
- Duplicate period-aware sample IDs: 0.
- Non-finite feature components: 0.
- Confirmation IDs loaded: none.

All earlier matches were weighted equally. The focal player was excluded from
the generic role prior. Every shuffled control found a valid donor.

## V1 Validation Result

The main player model used three match-equivalents of shrinkage.

| View | Primary NLL | Brier | ROC AUC | Average precision |
| --- | ---: | ---: | ---: | ---: |
| Geometry plus role | 0.234093 | 0.070776 | 0.8717 | 0.4183 |
| Generic role history | 0.241759 | 0.071660 | 0.8582 | 0.4036 |
| True player residual | 0.241411 | 0.071572 | 0.8561 | 0.3954 |
| Best shuffled-player residual | 0.240431 | 0.071178 | 0.8585 | 0.4075 |
| Unshrunk true player residual | 0.240380 | 0.071466 | 0.8585 | 0.3982 |

The true player residual made primary NLL 0.007318 worse than geometry plus
role, a relative deterioration of 3.13%. Brier score also worsened by
0.000797.

The match-period blocked-bootstrap interval for NLL gain was
[-0.018287, 0.000220]. Only 3.72% of bootstrap replicates had positive gain.

Per-match NLL gains were:

| Match | NLL gain |
| --- | ---: |
| Napoli, match 14 | -0.001434 |
| Lazio, match 15 | -0.026803 |
| Bayern, match 33 | +0.000562 |

Only one match had a positive effect, and that effect was extremely small.

## Shrinkage Sensitivity

| Player-history view | Primary NLL |
| --- | ---: |
| No shrinkage | 0.240380 |
| One match-equivalent | 0.240916 |
| Three match-equivalents, frozen main view | 0.241411 |
| Five match-equivalents | 0.241693 |

Every player-history sensitivity was worse than geometry plus role. The
negative result therefore is not explained by the frozen shrinkage strength.

The main true-player view slightly beat generic role history but did not beat
all player shuffles. A shuffled donor with seed 41 outperformed the true
player. This provides no evidence that the specific player's identity carried
the useful information.

## Secondary Result

For turnover within five seconds:

- Geometry-plus-role NLL: 0.267656.
- True-player-residual NLL: 0.269828.
- Relative change: -0.81%.

This missed the frozen -0.5% non-inferiority limit.

## Internal Stability Check

Every predeclared development fold was negative:

| Fold | Validation matches | Relative primary NLL change |
| --- | --- | ---: |
| A | 2, 11, 28 | -2.26% |
| B | 27, 32, 38 | -0.22% |
| C | 12, 13, 35 | -3.35% |

Pooled internal-CV primary NLL deteriorated by 1.52%. Its blocked-bootstrap
interval was [-0.007002, 0.001999].

## Gate

Passed:

- The true player residual was marginally better than generic role history.
- All integrity audits passed.

Failed:

- Primary improvement over geometry plus role.
- Positive primary bootstrap lower bound.
- Positive result in at least two V1 validation matches.
- Better performance than every player shuffle.
- Primary Brier non-inferiority.
- Secondary NLL non-inferiority.
- Positive pooled internal-CV improvement.
- Positive improvement in at least two internal folds.

The complete gate failed. The runner had no confirmatory-unseal path, and no
freeze or eligibility artifact was written.

## Interpretation

V2 resolves the ambiguity left by V1. V1's apparent gain was measured against
a rolling baseline that was itself much worse than current geometry. Once the
player signal is compacted, shrunk, compared directly with the strongest
current-state baseline, and checked against identity shuffles, it does not
help.

This does not establish that stable player tendencies do not exist. It shows
that the current FOOTPASS cohort is too shallow, or its action labels are too
limited, to demonstrate predictive player-specific memory under this
protocol. Actors have only 2.14 earlier matches on average and never more than
five.

Further tuning on these same three teams is unlikely to answer the research
question cleanly. The next useful dataset should contain substantially longer
longitudinal player histories, preferably at least 10-20 earlier matches per
frequently appearing player, with stable player identifiers and critical
event or outcome labels. Current geometry should remain the main baseline,
and role-prior plus identity-shuffle controls should be retained.

## Artifacts

- Protocol:
  `docs/FOOTPASS_COMPACT_PLAYER_RESIDUAL_PROTOCOL_V2.md`
- Config:
  `configs/footpass_compact_player_residual_v2.yaml`
- Results:
  `runs/footpass_compact_player_residual_v2/development/results.json`
- Feature audit:
  `runs/footpass_compact_player_residual_v2/development/feature_audit.json`
- Run manifest:
  `runs/footpass_compact_player_residual_v2/development/run_manifest.json`
- Frozen models:
  `runs/footpass_compact_player_residual_v2/development/models.npz`
