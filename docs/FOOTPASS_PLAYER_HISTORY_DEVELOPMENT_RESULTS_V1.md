# FOOTPASS Player-History Development Results V1

## Status

- Development result: gate failed.
- Claim status: development-only null result.
- Confirmation status: sealed and unread.
- Confirmation match IDs: 22, 40, and 43.
- Eligible to freeze/open confirmation: no.

This experiment asks whether player profiles built from strictly earlier
FOOTPASS matches improve later action-outcome prediction beyond current
geometry, role, static identity, and causal current-match statistics.

## Data and Integrity

- Development/support matches loaded: 14.
- Query matches: 12.
- Development-fit opportunities: 9,049.
- Development-validation opportunities: 2,859.
- Total development opportunities: 11,908.
- Validation primary positives: 277 of 2,859 (9.69%).
- Validation secondary positives: 273 of 2,859 (9.55%).
- Mean actor-history support: 1.93 earlier matches.
- Opportunities with unavailable actor history: 5.77%.
- Duplicate period-aware sample IDs: 0.
- Chronology violations: 0.
- Non-finite feature components: 0.
- Confirmation IDs in the extraction cache: none.

The primary target is an action-location proxy for a focal-team action in the
attacking penalty area within ten seconds and before the opponent's next
possession action. FOOTPASS does not contain ball coordinates.

## Primary Result

The frozen main comparison used a three-match prior-history cap.

| View | Validation NLL | Brier | ROC AUC | Average precision |
| --- | ---: | ---: | ---: | ---: |
| Rolling baseline | 0.294898 | 0.082123 | 0.8043 | 0.2826 |
| Prior player history | 0.275524 | 0.081617 | 0.8201 | 0.2878 |
| Geometry plus role | 0.234093 | 0.070776 | 0.8717 | 0.4183 |
| Role-mean history control | 0.271643 | 0.079701 | 0.8243 | 0.3011 |
| Best shuffled-history control | 0.271740 | 0.079511 | 0.8221 | 0.3029 |

Against the rolling baseline, true history improved NLL by 0.019374, or
6.57%, and improved Brier score by 0.000506.

The blocked bootstrap over six match-period units gave a 95% interval of
[-0.006606, 0.050583]. The interval includes zero. Bootstrap gain was positive
in 91.96% of replicates.

Per-match NLL gains were:

| Validation match | NLL gain |
| --- | ---: |
| Napoli, match 14 | +0.040621 |
| Lazio, match 15 | +0.044740 |
| Bayern, match 33 | -0.016531 |

## Controls and Sensitivity

| History view | Primary NLL |
| --- | ---: |
| One earlier match | 0.287277 |
| Three earlier matches, frozen main view | 0.275524 |
| Up to five earlier matches | 0.271926 |
| Role-mean history | 0.271643 |
| Event-only history | 0.280887 |
| Shuffled history, seed 11 | 0.271740 |

More support helped descriptively, but the role-mean history and one shuffled
identity control both outperformed the true player-specific history. This
means the apparent gain cannot be assigned to persistent individual-player
information. It is compatible with team/role context, shrinkage, or noise.

The much simpler geometry-plus-role view also substantially outperformed every
identity/history view. Static identity and current-match rolling statistics
did not add robust validation value in this cohort.

## Secondary Result

For turnover within five seconds:

- Rolling baseline NLL: 0.287288.
- Prior-history NLL: 0.298817.
- Relative NLL change: -4.01%.

Player history therefore worsened the secondary target beyond the allowed
-0.5% non-inferiority limit.

## Development Gate

Passed:

- Primary relative NLL gain was at least 0.5%.
- Primary gain was positive in at least two validation matches.
- Primary Brier score was non-inferior.
- Integrity audits passed.

Failed:

- The blocked-bootstrap lower bound was not above zero.
- True history did not beat every shuffled-history control.
- Secondary NLL non-inferiority failed.

The complete gate failed, so no confirmatory freeze was created and no
confirmatory action labels or outcomes were read.

## Interpretation

This is not evidence that player history is useless in general. It says that
with this small repeated-team cohort, a mean of fewer than two earlier matches
per actor, and this fixed logistic-probe protocol, player-specific prior
history is not distinguishable from role/team context and does not robustly
improve both outcomes.

The appropriate next scientific step is a new development protocol, not
opening confirmation. It should use more prior appearances per player and
lower-dimensional or hierarchically shrunk history features, while retaining
identity shuffles, role-mean controls, strict chronology, and untouched
confirmation matches.

## Implementation Diagnostic

An initial development attempt used the mean squared coefficient in the L2
term instead of the sum squared coefficient used by the repository's
established logistic probes. This made regularization depend incorrectly on
feature dimensionality and caused severe high-dimensional overfitting.

That attempt was archived as invalidated, no confirmation data was read, and
the corrected run changed only the L2 implementation. The configured
coefficient, split, targets, feature views, and development gate were
unchanged.

## Artifacts

- Protocol: `docs/FOOTPASS_PLAYER_HISTORY_PROTOCOL_V1.md`
- Config: `configs/footpass_player_history_v1.yaml`
- Split: `splits/footpass_player_history_chronological_v1.json`
- Development results:
  `runs/footpass_player_history_v1/development/results.json`
- Development run manifest:
  `runs/footpass_player_history_v1/development/run_manifest.json`
- Feature audit:
  `runs/footpass_player_history_v1/development/feature_audit.json`
- Invalidated implementation attempt:
  `runs/footpass_player_history_v1/development_attempt_1_underregularized/`
