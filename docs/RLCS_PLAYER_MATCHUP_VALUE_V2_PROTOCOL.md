# RLCS Player-Matchup Critical Value V2 Protocol

Status: amended before outcome construction; see
`RLCS_PLAYER_MATCHUP_VALUE_V2_AMENDMENT_01.md`.

This protocol operationalizes `RLCS_PLAYER_MATCHUP_VALUE_V2_NEXT_SCIENTIFIC_STEP.md`.
V1 remains a closed negative result for identity-conditioned next-touch prediction. V2 changes
both the target and the identity representation; it is not a reinterpretation or continuation of
the failed V1 gate.

## Claim and target

The preregistered claim is that, after complete telemetry, score, clock, overtime, and
chronology-safe team form are controlled, persistent player profiles and explicit
actor-versus-opponent interactions improve prediction of short-horizon scoring outcomes.

Each eligible de-duplicated touch receives exactly one class:

1. `no_goal`: neither team scores before the next ten touch events;
2. `score`: the actor's team scores first;
3. `concede`: the opponent scores first.

The horizon ends at the earlier of the tenth subsequent touch and the first goal or kickoff
boundary. It never crosses a reset. Model selection and comparison use three-class log loss.
Multiclass Brier score, 15-bin expected calibration error, and one-vs-rest average precision for
`score` and `concede` are secondary metrics. Accuracy is not a decision metric.

State value is `P(score) - P(concede)`. For consecutive states around an observed non-goal
action, action value is the later state value minus the earlier state value, expressed from the
same team's orientation. Scoring touches are excluded from match aggregation.

## Immutable chronology

Official series, independently within EU and NA, are ordered by earliest replay timestamp and
then series ID. Split 1 series are allocated by largest remainder to 35%, 45%, and 20%:

- earliest: profile support only;
- middle: model training;
- latest: internal development.

Split 2 Regional 1 is frozen validation. Split 2 Regionals 2 and 3 remain sealed test. No ordinary
builder, trainer, summary, or validation evaluator may open the sealed test. Test evaluation
requires a one-use unlock bound to the frozen dataset, split manifest, configuration, and selected
checkpoints.

Profiles supplied to a query use games with timestamps strictly earlier than the query game. A
game never contributes to its own profile. Empirical-Bayes population means, variances, and prior
strengths are fitted only on profile-support games. Each stored snapshot contains its prior-game
count, effective sample size, and uncertainty. Missing histories shrink to the support population.

The primary model never consumes display names, player IDs, team IDs, roster hashes, or learned
identity embeddings. Those identifiers may exist only as provenance and join keys outside model
inputs.

## Frozen 28-feature profile

The feature order is code-frozen in `footballq.data.rlcs_player_profiles.PROFILE_FEATURES`:

1. mean speed;
2. supersonic-time fraction;
3. mean boost;
4. low-boost fraction;
5. boost-active fraction;
6. aerial-time fraction;
7. goalside fraction;
8. goalside recovery speed;
9. mean distance to ball;
10. mean distance to own goal;
11. mean teammate spacing;
12. nearest-to-ball fraction;
13. touches per minute;
14. shots per touch;
15. goals per shot;
16. saves per minute;
17. attacking-half touch fraction;
18. forward ball velocity per touch;
19. ground-dribble frequency per touch;
20. aerial-control frequency per touch;
21. flick frequency per touch;
22. pass frequency per touch;
23. rebound frequency per touch;
24. retrieval frequency per touch;
25. challenge-win fraction;
26. turnover frequency per touch;
27. double-commit frequency per minute;
28. nearest-defender conceded-shot rate per minute.

Time fractions and movement summaries are estimated from one-second telemetry samples. Event
rates use the cached AnalyzerL event stream. Distance and speed features are normalized by fixed
arena and maximum-car-speed constants. These are descriptive behavioral summaries, not causal
player attributes.

## Original mandatory profile gate

Before outcome dataset construction or training, support games are split into chronological early
and late halves per player. Eligible players need at least 15 support games. The gate requires all
of:

- at least 60 eligible players;
- same-player early/late retrieval AUC at least 0.75 against different players matched by region,
  date, and prior team strength;
- median early/late Spearman correlation at least 0.35 over the frozen core continuous traits.

Failure stops V2 without outcome training and without opening Split 2 data.

Amendment 01 records that this original count gate failed with 48 eligible players, then replaces
only the arbitrary minimum-player count with a complete-cohort, bootstrap-uncertainty, and
regional-stability gate. The amendment was frozen before any V2 outcome row was constructed. Its
decision rule controls whether the downstream sections of this protocol may execute.

## Model and matched conditions

Every condition uses the same 20 x 7 x 27 telemetry Transformer (width 192, three layers, six
heads, feed-forward width 768), scalar score/clock context, optimizer, batches, and seeds 17, 23,
and 41. All models instantiate the same modules; unavailable information is masked to zero.

The five conditions are `state`, `team_form`, `actor_profile`, `additive_profiles`, and
`full_matchup`. The full model constructs three actor-opponent tokens from actor profile, opponent
profile, their difference, relative position and velocity, opponent distances to the ball and own
goal, and an intercept-time difference. Attention-weighted sum and elementwise maximum aggregate
the opponent tokens. Teammate synergy is projected separately.

Required controls are actor-profile shuffling within team-strength bands, opponent-profile
shuffling across matched series, permuting opponent profiles while retaining geometry, and
population-mean replacement.

## Stop rules and validation gate

Training cannot start unless both `score` and `concede` have at least 5,000 positive training
rows. Internal development must show a full-model gain over team form, separation from profile
shuffles, positive EU and NA point estimates, and separation from additive profiles before frozen
validation is calculated.

Frozen validation passes only when:

- full beats team form by at least 2% relative log loss in at least two of three seeds;
- the 95% official-series bootstrap lower bound is above zero;
- full beats additive and actor-only by at least 0.5% each;
- full beats each main shuffle by at least 1%;
- ECE is no more than 0.01 worse than state;
- the point estimate is positive in both EU and NA.

Only a passing frozen validation result can create the separate one-use V2 test unlock.

## V1 diagnostic boundary

`scripts/audit_rlcs_v1_outcome_heads.py` may read the 12 frozen V1 checkpoints and the already
open V1 validation split only. It reports prevalence, log loss, Brier score, average precision,
calibration, retained-possession metrics, condition/seed comparisons, and the V1 critical-state
subset. It cannot create an unlock, load test, change checkpoint selection, or alter the V1
conclusion.

## Match aggregation

Only after the action-value validation gate passes may values be accumulated at 60 and 120
seconds. A small logistic regression compares score/time/overtime/team-form alone with those
features plus cumulative values and extreme/high-value counts. At 120 seconds the full aggregate
must reduce final-winner log loss by at least 2% and have a positive official-series bootstrap
lower bound. Failure limits any claim to short-horizon critical-value prediction.
