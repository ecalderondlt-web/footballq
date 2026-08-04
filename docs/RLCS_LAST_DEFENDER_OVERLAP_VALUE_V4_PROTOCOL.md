# RLCS Last-Defender Overlap-Weighted Value V4 Protocol

Status: pre-outcome frozen. No V4 success label, Split 2 validation row,
or Split 2 test row may be read until the committed freeze and the outcome-blind support gate
authorize it.

## Why V4 exists

V3 stopped correctly under its written rule because its deterministic greedy pairs retained an
absolute standardized mean difference (SMD) of `0.2213` in acting-team prior win rate. V3 remains
closed and is not retroactively reclassified as a pass.

After that stop, an outcome-blind diagnostic used the already stored out-of-fold propensities and
found that overlap weighting reduced the largest overall matching-feature SMD to about `0.019`.
No action or success label was opened. The user explicitly authorized a separately numbered V4
using this fix. V4 therefore changes the identification procedure, not the opportunity detector,
profiles, exposure, success horizon, or chronology.

## Claim boundary

V4 asks one narrow predictive-association question:

> Among the established RLCS professionals and last-defender opportunities with measured
> covariate overlap, do chronology-safe actor/defender profiles and their frozen matchup
> interaction improve prediction of short-horizon attacking success beyond physical state and
> team form?

V4 does not estimate a randomized player effect, prove that a player caused an outcome, establish
the V3 action-choice mediation chain, generalize to all Rocket League players, or transfer to
association football. A positive result would apply to the overlap population represented by the
weighted Split 1 opportunities.

## Immutable source and chronology

V4 reuses exactly the outcome-free V3 opportunity inventory with SHA-256
`5b0d73951f04e842888eef38fbbb23c740166bfb407fc02f9300f2fc80d4f265`.

- Rows: exactly 5,256.
- Stages: V2 `train` and `internal_development` only.
- Players: the complete 48-player V2 stability cohort.
- Minimum prior games: 15 for actor and last defender at the query time.
- Opportunity detector, geometry thresholds, profile standardization, team form, exposure median,
  and sample IDs are inherited unchanged from V3.
- Split 2 Regional 1 validation and Split 2 Regionals 2-3 test remain sealed.
- Every fit, fold, bootstrap, and uncertainty unit is an official series.

The V4 design runner must reject an inventory containing action, success, outcome, future-contact,
shot, or goal columns. It must also reproduce the stored five-fold, series-out-of-fold V3
propensity values and fold assignments before computing V4 weights.

## Stage 0: outcome-blind support redesign

### Primary overlap weights

Use the frozen V3 favorable-matchup exposure and out-of-fold propensity `p`:

```text
favorable row:   weight = 1 - p
unfavorable row: weight = p
```

No trimming, winsorization, stabilization chosen from outcomes, or refitting after outcomes is
permitted. Missing matching-state features use the complete-inventory median solely for balance
diagnostics. For each exposure group, calculate its weighted mean and weighted population
variance. The absolute weighted SMD denominator is the square root of the mean of the two weighted
variances. A constant feature has SMD zero only when the weighted means are equal.

Stage 0 primary gates:

- V3 inventory hash and row count match exactly;
- no forbidden future/action/outcome column exists;
- recomputed propensities and fold assignments reproduce V3;
- at least 70% of propensities lie in `[0.10, 0.90]`;
- total overlap-weight effective sample size (ESS) is at least 400;
- favorable and unfavorable weighted ESS are each at least 400; and
- every overall weighted matching-feature absolute SMD is at most 0.10.

Regional weighted SMDs are reported transparently but are not a separate pass/fail gate. Region is
handled in the outcome model and the regional direction checks below.

### Exact-series sensitivity design

As an outcome-blind guard against cancellation across teams or regions, form a separate sensitivity
sample. Exact strata are `(official series, acting-team roster SHA-256)`. Within a stratum, pair the
maximum possible number of favorable and unfavorable rows without replacement using the minimum
total Euclidean distance in globally median-imputed, standardized matching-state features. The
assignment is solved deterministically; sample IDs break numerical ties.

Sensitivity gates:

- at least 1,000 exact-stratum pairs; and
- every unweighted paired-sample matching-feature absolute SMD at most 0.10.

Both the primary overlap-weight and exact-stratum gates must pass before any V4 outcome event is
read. Passing creates a one-use outcome unlock containing the source, protocol, code, config,
weighted-inventory, and support-audit hashes. Failure stops V4.

## Frozen success label

Only after Stage 0 passes may the parser events needed for this label be read.

For an authorized opportunity, `success = 1` exactly when the actor's team records an AnalyzerL
`shot` or `goal` from the current contact through the earliest of:

1. five subsequent de-duplicated contacts;
2. two consecutive opponent contacts with no actor-team contact between them;
3. a goal or kickoff boundary; or
4. the end of the current parser stint.

A goal at the terminating boundary counts before the horizon closes. A row that reaches the parser
stint end without one of the first three observable termination rules is censored and excluded.
No action-class label is created or inspected in V4.

Outcome-volume gates require:

- at least 1,000 uncensored rows;
- at least 100 successes and 100 failures overall; and
- at least 30 successes and 100 failures separately in EU and NA.
- at least 500 exact-stratum pairs for which both rows are uncensored;
- every overlap-weighted matching-feature SMD among uncensored rows at most 0.10; and
- every matching-feature SMD among the complete exact-stratum pairs at most 0.10.

Failure stops before fitting a predictive model.

## Models and evaluation

All conditions are regularized binary logistic regression with median imputation and standard
scaling fit only within the relevant training fold. Sample weights enter both fitting and scoring.
Outer evaluation uses five-fold `StratifiedGroupKFold`, grouped by official series, seed 20260808.
Each condition selects `C` independently in four-fold grouped inner cross-validation from
`[0.01, 0.1, 1.0, 10.0]`; ties select the smaller `C`.

| Condition | Inputs |
|---|---|
| state | all frozen matching-state features except the six team-form features |
| state + team form | all frozen matching-state features |
| additive profiles | state + team form plus the seven actor/defender component traits |
| full matchup | additive profiles plus actor/defender composites, mismatch, and five frozen actor-by-defender interactions |

The five interactions are:

- actor carry speed x defender goalside recovery;
- actor boost economy x defender boost economy;
- actor take-on/control frequency x defender challenge win;
- actor take-on/control frequency x defender turnover-pressure proxy; and
- actor attack composite x defender resistance composite.

The primary metric is strictly out-of-fold overlap-weighted binary log loss. The exact-stratum
sensitivity uses the same out-of-fold predictions but scores only complete uncensored pairs,
equally.

## Uncertainty and controls

- Compute 10,000 paired official-series bootstrap resamples, seed 20260809.
- Run 1,000 profile-row permutations, seed 20260810. Within each official series, permute the
  complete actor/defender profile row jointly, recompute all matchup interactions, refit the full
  model with its frozen outer-fold `C` values, and compare its improvement over state + team form.
  The plus-one corrected one-sided p-value is reported.
- Report overlap-weighted point estimates separately in EU and NA.
- Report the full-versus-baseline direction on the exact-series/acting-roster sensitivity pairs.

The series bootstrap is always reported. If either minimum point-estimate reduction fails, the
profile permutations are not run because they cannot rescue the failed primary gate; they are
recorded as unevaluated rather than passed.

## Split 1 decision rule

V4 passes Split 1 only when the full matchup model:

- reduces overlap-weighted log loss by at least 1% versus state + team form;
- reduces overlap-weighted log loss by at least 0.5% versus additive profiles;
- has official-series bootstrap 95% lower bounds greater than zero for both reductions;
- has profile-permutation `p < 0.01`;
- has a positive full-versus-team-form point estimate separately in EU and NA; and
- has positive full-versus-team-form and full-versus-additive point estimates in the exact-stratum
  sensitivity sample.

If any condition fails, close V4 without opening Split 2. If all conditions pass, freeze the
selected bundle and all hashes before implementing or opening one Split 2 Regional 1 validation.
Split 2 Regionals 2-3 remain sealed unless that later validation also passes.

## Final interpretation rule

A V4 failure means that correcting the measured V3 balance problem did not convert the frozen
profiles into robust incremental success prediction in this overlap population. It does not prove
that player matchups never matter. A V4 pass would be observational evidence requiring later
chronological validation and would still not establish causality.
