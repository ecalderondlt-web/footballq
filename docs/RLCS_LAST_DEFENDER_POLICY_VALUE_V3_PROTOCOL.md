# RLCS Last-Defender Policy-Value V3 Protocol

Status: pre-outcome freeze. No V3 action label, action-success label, Split 2 validation row,
or Split 2 test row may be read before the gates below authorize it.

V1 closed as a negative result for identity-conditioned next-touch prediction. V2 closed as a
negative result for identity-conditioned ten-touch score/concede prediction despite stable player
profiles. V3 is the final RLCS test of a narrower mechanism: whether persistent actor and defender
traits affect the action selected in a mechanically comparable last-defender opportunity, and then
whether that action has different conditional value. V3 is not a continuation or retuning of V2.

## Claim boundary

The causal chain under test is:

```text
actor and defender traits
        -> action selected in one comparable opportunity family
        -> action-specific success
```

The only opportunity family is an attacking last-defender duel. No other critical-state family,
horizon, label, model class, or fallback target may be substituted after results are observed.

A Stage 1 pass supports only persistent player-style and matchup effects on action choice within
established RLCS professionals. A Stage 2 pass is required before claiming that a particular action
has greater critical value against a particular defender type. Nothing in V3 establishes broad
generalization to all Rocket League players or to association football.

## Immutable chronology and split locks

V3 reuses the frozen V2 official-series chronology and profile snapshots.

- V2 `profile_support` fits profile priors and histories only. It supplies no V3 opportunity rows.
- V2 `train` is the only threshold-calibration partition.
- V2 `train` and `internal_development` are the only Stage 0 inventory and cross-fitting rows.
- Every outer, inner, bootstrap, and permutation unit is an official series. A series never crosses
  a fold.
- Split 2 Regional 1 remains unopened until the Stage 0, label-audit, Stage 1, and Stage 2 gates all
  pass on Split 1 and the selected V3 bundle is frozen.
- Split 2 Regionals 2 and 3 remain sealed until the single Split 2 Regional 1 validation passes.
- An ordinary builder, auditor, labeler, or trainer must reject `validation` and `test` stages.

Profiles at a query use games with timestamps strictly earlier than the query replay. The actor and
last defender must both belong to the complete 48-player V2 stability cohort and must each have at
least 15 prior games at that query. No manual player selection is permitted. Player and team names
must not enter a model feature.

## Stage 0A: outcome-blind opportunity calibration

### Permitted inputs

Stage 0 may read current and past telemetry, current score and clock, replay/series/region keys,
chronology-safe profile snapshots, and the event type only to identify and de-duplicate physical
contacts and to reject goal/kickoff boundaries. It must not persist, summarize, or inspect action
classes, future contacts, shots, goals, success labels, or V2 outcome labels.

The calibration base consists of de-duplicated current contacts in V2 `train` replays that satisfy:

1. exact resolved 3v3 roster;
2. a complete 2.0-second, 10 Hz, past-only telemetry window ending at the current contact;
3. valid ball and six-car rigid-body state throughout the window;
4. one parser stint throughout the window and no goal or kickoff inside it;
5. actor and ball strictly in the actor-oriented attacking half; and
6. actor is the closest member of its team to the ball at the current frame.

Blue attacks toward actor-oriented `+y`. Orange coordinates and velocities are reflected across
the field center so it also attacks toward `+y`. Planar detector distances use Rocket League units.
The opponent goal center is `(0, 5120)`.

For a goal-side opponent, project its planar position onto the ball-to-goal segment. It is
goal-side exactly when its projection lies strictly after the ball and no farther than the goal.
The lateral distance is its perpendicular distance to that segment, and forward distance is the
distance along that segment from the ball.

The four thresholds below are calibrated once from the V2 `train` base, using NumPy's linear
quantile estimator, and then clipped to the stated physical range:

| Threshold | Frozen statistic | Physical clip |
|---|---:|---:|
| defensive corridor half-width | 60th percentile of the nearest goal-side opponent lateral distance | 700 to 1,800 |
| maximum last-defender forward distance | 80th percentile of that nearest-to-path opponent's forward distance | 1,200 to 4,200 |
| immediate intervention range | 25th percentile of the closest non-selected opponent's minimum planar distance to actor or ball | 900 to 2,200 |
| teammate overload range | 25th percentile of the closest teammate's minimum planar distance to actor or ball | 900 to 2,200 |

A calibration statistic requires at least 500 finite base observations. Any shortage is a Stage 0
failure. The quantiles, clips, base definition, or estimator may not be changed after calibration.

### Accepted opportunity

Apply the frozen thresholds to V2 `train` and `internal_development`. A base contact is accepted
exactly when:

1. exactly one opponent is goal-side, inside the defensive corridor, and no farther than the
   maximum forward distance; this player is the last defender;
2. each other opponent's minimum planar distance to the actor or ball is strictly greater than the
   immediate intervention range;
3. each teammate's minimum planar distance to the actor or ball is strictly greater than the
   teammate overload range;
4. actor and last defender are members of the frozen 48-player cohort; and
5. both stored snapshots have at least 15 strictly prior games.

The immutable sample key is
`replay_id:stint_<stint>:touch_<observed_frame_number>:last_defender_v3`. Tied distances are broken
by parser prefix. Duplicate sample keys are fatal.

### Opportunity-volume gates

The frozen inventory must contain:

- at least 1,000 accepted opportunities;
- at least 30 distinct actors and 30 distinct last defenders;
- at least 300 opportunities in EU and at least 300 in NA;
- maximum actor share at most 10%; and
- maximum last-defender share at most 10%.

Failure of any volume or concentration gate closes V3 and the RLCS substrate before action labels
or outcomes are opened.

## Stage 0B: outcome-blind common-support audit

### Frozen matchup exposure

Let each profile trait be standardized by the V2 profile-support population mean and uncertainty
scale, with each standardized value clipped to `[-5, 5]`.

- actor carry speed: `mean_speed`;
- actor boost economy: mean of `mean_boost` and negative `low_boost_fraction`;
- actor take-on/control frequency: mean of `ground_dribble_per_touch`,
  `aerial_control_per_touch`, and `flick_per_touch`;
- actor attack composite: mean of the three actor quantities above;
- defender goalside recovery: `goalside_recovery_speed`;
- defender challenge win: `challenge_win_fraction`;
- defender boost economy: the same two-feature boost composite;
- defender turnover pressure proxy: mean of `retrieval_per_touch`,
  `nearest_to_ball_fraction`, `challenge_win_fraction`, and negative
  `turnover_per_touch`;
- defender resistance composite: mean of the four defender quantities above; and
- matchup mismatch: actor attack composite minus defender resistance composite.

The favorable-matchup exposure is `1` when mismatch is greater than or equal to the median among
accepted V2 `train` opportunities and `0` otherwise. The train median is frozen before
`internal_development` exposure is assigned. No action or success label enters this definition.

### Matching-state features

Common support may use only the current, actor-oriented values below:

- ball position and velocity (three axes each);
- actor position, velocity, 3D forward orientation unit vector, and boost;
- last-defender position, velocity, 3D forward orientation unit vector, and boost;
- actor-defender planar distance;
- actor and defender intercept times, each equal to 3D ball distance divided by maximum current
  speed and 500 units/second, clipped to five seconds;
- planar ball distance to opponent goal;
- nearest-teammate support distance used by the detector;
- closest non-selected-opponent recovery distance used by the detector;
- actor-perspective score difference, regulation seconds remaining, and overtime indicator; and
- the frozen six-dimensional V2 team-form vector: team/opponent mean prior win rate, team/opponent
  mean prior goal difference, and team/opponent mean log-one-plus prior-game count.

No player ID, team ID, profile, mismatch, action, or outcome is a matching-state input. Region is an
exact matching stratum, not a fitted feature.

### Propensity and matching procedure

1. Estimate favorable-matchup propensity with five-fold `StratifiedGroupKFold`, seed 20260803,
   grouped by official series. Each training fold fits median-imputed features, a standard scaler,
   and L2 logistic regression with `C=1.0`; predictions are strictly out of fold.
2. Require at least 70% of all rows to have propensity in `[0.10, 0.90]`.
3. Compute overlap weights `w = 1-p` for favorable rows and `w = p` for unfavorable rows. Report
   effective sample size as `(sum(w)^2) / sum(w^2)` over the complete accepted inventory.
4. Within each region, form deterministic one-to-one favorable/unfavorable pairs without
   replacement. A pair is feasible only when its logit-propensity difference is no greater than
   `0.20` times the pooled logit-propensity standard deviation. Process rows with the fewest
   feasible counterparts first, breaking ties by sample key, and select the counterpart with the
   smallest Euclidean distance in globally median-imputed, standardized matching-state features.
5. A matched set is one such pair. Compute absolute standardized mean difference for every
   matching-state feature on the paired rows using the pooled matched standard deviation. A
   constant feature has SMD zero only when group means are equal; otherwise it fails.

Stage 0B passes only with:

- at least 300 matched sets;
- overlap-weight effective sample size at least 400;
- at least 70% of propensities in `[0.10, 0.90]`; and
- every absolute matched-state SMD at most 0.10.

The Stage 0 report must include every threshold, count, identity share, propensity statistic,
feature SMD, input hash, and split-open flag. Failure closes V3. A larger model cannot repair a
common-support failure.

## Identity-blinded action-label audit

Only after both Stage 0 gates pass may event action types after the accepted current contacts be
opened. Mechanical labels use the current event cluster and subsequent de-duplicated contacts up
to the earliest of two seconds, a goal/kickoff, or the second subsequent distinct contact.

The mutually exclusive labels and precedence are:

1. `finish`: an actor `shot` or `goal` annotation occurs before any subsequent different-player
   contact;
2. `release`: an actor `pass` annotation occurs and the first subsequent different-player contact
   within two seconds belongs to a teammate;
3. `take_on`: before a teammate contact, the actor has a `ground-dribble`, `air-dribble`, `flick`,
   `flip-reset`, or `double-tap` annotation, or the actor records a de-duplicated re-touch before
   any different-player contact; and
4. `ambiguous`: no class applies, signals from multiple classes conflict, a required next contact
   is missing, or a boundary intervenes.

Precedence resolves annotations only when lower-priority evidence does not imply a different
first recipient; otherwise the row is ambiguous. Ambiguous rows are excluded, never relabeled.

Using seed 20260803, sample 100 class-stratified eligible opportunities with official series as
the sampling unit. Render past/current telemetry and the short contact sequence with player names,
IDs, teams, region, series, action label, and future result hidden. A human records only
`take_on`, `release`, `finish`, or `ambiguous`. Mechanical/manual agreement among all 100 must be at
least 80%. There is no adjudication-based threshold change. Failure closes V3.

## Stage 1: action choice

Train regularized multinomial logistic regression, not a neural network. All four conditions use
identical outer five-fold official-series splits. Hyperparameter `C` is selected independently in
four-fold grouped inner cross-validation from `[0.01, 0.1, 1.0, 10.0]`; ties choose smaller `C`.

| Condition | Inputs |
|---|---|
| state | frozen matching-state features except team form |
| state + team form | state plus the six chronology-safe team-form features |
| additive profiles | state + team form plus the seven actor/defender opportunity traits |
| matchup interaction | additive profiles plus mismatch and actor-minus-defender trait interactions |

The primary metric is strictly out-of-fold three-class log loss. Use 10,000 paired
official-series bootstrap resamples, seed 20260804. For the profile-permutation control, use 2,000
seeded permutations that reassign complete player-profile histories among player IDs within region,
preserve each player's repeated-row structure, refit the frozen-hyperparameter matchup model, and
report the plus-one corrected one-sided p-value.

Stage 1 passes only when matchup interaction:

- reduces log loss by at least 1% relative to state + team form;
- reduces log loss by at least 0.5% relative to additive profiles;
- has official-series bootstrap 95% lower bound greater than zero for both reductions;
- has profile-permutation `p < 0.01`; and
- has a positive matchup-versus-team-form point estimate separately in EU and NA.

If additive profiles pass but the interaction does not, the conclusion is persistent player style,
not opponent-specific exploitation. No Stage 2 label may be opened unless every Stage 1 gate passes.

## Stage 2: action-specific success

For a Stage 1-authorized row, success is one exactly when the actor's team records an AnalyzerL
`shot` or `goal` before the earlier of five subsequent de-duplicated contacts or two consecutive
opponent contacts with no actor-team contact between them. A boundary ends the horizon. This label
is frozen before outcomes are opened.

Estimate action propensity from Stage 1. Estimate action-specific success with official-series
cross-fitted, doubly robust estimation. The primary comparison is:

- state + observed action + team form; versus
- state + observed action + additive profiles + action-by-matchup interactions.

Stage 2 passes only with at least 1% success log-loss reduction, paired official-series bootstrap
95% lower bound greater than zero, player-profile permutation `p < 0.01`, positive point estimates
in EU and NA, and leave-one-actor, defender, team, and series analyses that preserve the sign.

Only a Stage 2 pass supports the narrow claim that, against this defender type, this action by this
player has greater critical value.

## Validation, sealed test, and final stop

After a complete Split 1 Stage 2 pass, freeze the opportunity detector, profile cohort, labeler,
features, folds, hyperparameters, selected model bundle, and all hashes. Open Split 2 Regional 1
once for V3 validation. Open Split 2 Regionals 2 and 3 once only if that validation passes.

If the common-support audit, label audit, Stage 1, Stage 2, or Split 2 validation fails, close RLCS.
Do not create a fourth target, widen the opportunity definition, tune V2, extend the horizon, add a
Transformer, or aggregate to winner prediction. The final negative conclusion is that stable
player-specific information did not yield robust incremental tactical value across immediate
transitions, ten-touch outcomes, and the single mechanically defined critical opportunity family
in this observational substrate.
