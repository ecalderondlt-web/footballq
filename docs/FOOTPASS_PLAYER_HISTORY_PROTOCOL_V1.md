# FOOTPASS Player-History Prediction Protocol V1

## Frozen Question

Does a focal player's profile from strictly earlier matches improve prediction
of a later on-ball outcome beyond current tracking geometry, tactical role,
static player identity, and causal statistics from the current match?

This is a small, low-cost test of incremental player-history value. It is not a
test of full-match tactical planning, opponent scouting, or a trained
player-conditioned TD-JEPA.

## Data

- Source: official FOOTPASS tactical training HDF5.
- Tracking rate: 25 fps.
- Available fields: match-local player slot, attack direction, shirt number,
  role, normalized x/y, x/y velocity, video ROI, and action class.
- Unavailable fields: player name in the source file and ball coordinates.
- Stable focal identities are season-scoped joins of externally verified team
  identity and shirt number.
- Identity manifests:
  - `provenance/footpass_napoli_identity_v1.json`
  - `provenance/footpass_lazio_identity_v1.json`
  - `provenance/footpass_bayern_identity_v1.json`
- Immutable split:
  `splits/footpass_player_history_chronological_v1.json`

The cohort contains five Bayern appearances, seven Napoli appearances, and six
Lazio appearances. Match 10 contains both Napoli and Lazio and is loaded once.

## Chronology

For a query in match `q`, player history may contain only focal-team
appearances whose verified date is strictly earlier than the date of `q`.
Neither the query match nor any later match may enter the history profile.

- Support only: Bayern match 6; Napoli/Lazio match 10.
- Development fit: matches 2, 11, 12, 13, 27, 28, 32, 35, and 38.
- Development validation: matches 14, 15, and 33.
- Sealed confirmation: matches 22, 40, and 43.

Confirmation action labels and outcomes must not be read until development
code, configuration, feature names, fitted hyperparameters, and the
development result have been frozen.

## Opportunities

An opportunity is a focal-team action row with class:

- drive;
- pass;
- cross;
- throw-in; or
- header.

Shots, tackles, and blocks are not query events. The current action must begin
outside the attacking penalty area. Coordinates are put into the focal team's
attacking direction using the source `left_to_right` field. Sample identity is
period-aware:

`<team_id>:<match_id>:p<period>:f<frame>:slot<match_local_player_id>`

No future tracking frame is used as an input.

## Outcomes

### Primary: penalty-area action within 10 seconds

Positive when, within 10 seconds and before the opponent's next possession
action, the focal team records a possession action whose actor is inside the
attacking penalty area.

This is an action-location proxy because FOOTPASS has no ball coordinates. It
must not be described as a directly observed ball entry.

### Secondary: turnover within 5 seconds

Positive when the first subsequent possession action within five seconds is by
the opponent. If no subsequent possession action is recorded inside the
window, the label is negative and this censoring limitation is reported.

Possession-action classes are drive, pass, cross, throw-in, shot, and header.

## Feature Views

All feature views receive the current action class and identity-free current
geometry. Geometry includes actor kinematics, team/opponent aggregates, and
distance-sorted relative kinematics for nearby teammates and opponents.

1. `geometry`: current geometry and current action only.
2. `geometry_role`: geometry plus actor role.
3. `geometry_role_identity`: role view plus static season-scoped actor ID.
4. `rolling`: identity view plus causal current-match actor statistics built
   only from events before the query frame.
5. `history`: rolling view plus strictly prior-match player profiles.

History profiles contain prior event distributions, prior outcome rates, and
off-ball tracking summaries sampled every five seconds. The main support cap
is the three most recent earlier matches. Sensitivity caps are 1, 3, and 5.
Missing history uses explicit availability/support fields and a role prior
fitted on development training histories.

The main comparison is `history` versus `rolling`.

## Controls

- Same-team, same-role history permutation with seeds 7, 11, 23, 41, and 73.
- Same-team role-mean profile, which removes player identity.
- Event-only history ablation, which removes off-ball tracking summaries.
- Support-size curve at 1, 3, and 5 earlier matches.
- Reverse-chronology development diagnostic, reported as leakage and excluded
  from every gate and paper claim.

Static identity, shirt number, role, ROI, and action labels are never included
in the geometry tensor.

## Probe

- Separate binary logistic probe for each target and feature view.
- Train-only centering and scaling.
- Fixed L2 coefficient: `0.01`.
- Deterministic CPU LBFGS optimization.
- No class weighting, preserving probability calibration.
- No validation-tuned architecture or regularization.

Report sample count, prevalence, negative log likelihood, Brier score, average
precision, ROC AUC, macro F1 at 0.5, and expected calibration error. Report
per-match effects and a blocked bootstrap over match-period units.

## Development Gate

The sealed confirmation may be opened only if all checks pass for the
development validation cohort:

1. Primary relative NLL improvement of `history` over `rolling` is at least
   0.5%.
2. The 95% blocked-bootstrap lower bound for primary NLL gain is above zero.
3. Primary NLL gain is positive in at least two of the three validation
   matches.
4. `history` has lower primary NLL than every same-role shuffled-history
   control.
5. Primary Brier score is no worse than `rolling`.
6. Secondary relative NLL change is no worse than -0.5%.
7. All chronology, sample-identity, feature-lineage, and finite-value audits
   pass.

Failure leaves all three confirmation matches sealed.

## Confirmatory Success Rule

After a passing development freeze, the same checks are applied once to the
three untouched later matches. A positive result supports only this statement:

> In three predeclared later matches from the verified Bayern, Napoli, and
> Lazio FOOTPASS cohorts, strictly earlier player histories improved a frozen
> action-location prediction protocol beyond current geometry, role, static
> identity, and causal current-match statistics.

It would not establish population-wide tactical understanding, full-match
planning, opponent-specific adaptation, or superiority of a
player-conditioned TD-JEPA. A failed confirmation is a valid null result.
