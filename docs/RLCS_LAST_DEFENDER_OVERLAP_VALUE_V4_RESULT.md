# RLCS Last-Defender Overlap-Weighted Value V4 Result

Status: final Split 1 stop. Split 2 validation and test remain sealed.

## Simple result

**The V4 weighting fix worked, but the player-matchup prediction still did not.**

V3 had stopped before outcomes because its greedy matched sample remained entangled with acting-team
strength. V4 kept the same 5,256 last-defender opportunities and corrected that measured imbalance
before opening success labels:

- primary overlap-weight maximum absolute SMD: `0.0193` (required at most `0.10`);
- overlap-weight ESS: `4,620.57` (required at least `400`); and
- same-series, same-acting-roster sensitivity: 1,413 pairs with maximum SMD `0.0923`.

After the support gate passed, V4 opened only the frozen Split 1 success label: whether the acting
team produced a shot or goal before the five-contact, two-consecutive-opponent-contact, or boundary
horizon. Of 5,256 opportunities, 5,181 were uncensored, with 2,134 successes and 3,047 failures.
Post-label balance also passed.

The full actor-defender matchup model was worse than every simpler condition. Lower log loss is
better.

| Condition | Overlap-weighted out-of-fold log loss |
|---|---:|
| Physical state | **0.624978** |
| State + team form | 0.625734 |
| Additive actor/defender profiles | 0.628180 |
| Full matchup interactions | **0.629135** |

Relative to state plus team form, the full matchup model was `0.5435%` worse. Its official-series
bootstrap 95% interval was `0.9267%` worse to `0.1665%` worse, entirely on the wrong side of zero.
Relative to additive profiles, full matchup was `0.1520%` worse; that interval ranged from `0.3887%`
worse to `0.0530%` better.

The direction was also negative in both regions and in the stronger exact-stratum sensitivity:

| Check | Full matchup vs state + team form |
|---|---:|
| EU overlap-weighted | 0.8403% worse |
| NA overlap-weighted | 0.2475% worse |
| Same-series/same-acting-roster pairs | 0.2617% worse |

The prespecified 1% improvement over state plus team form and 0.5% improvement over additive
profiles both failed. Profile permutations were not run because they could not rescue failed
primary point-estimate gates; they are unevaluated, not passed.

## What V4 resolves

V3 alone could not say whether its stop hid a useful player signal because its frozen greedy
matcher failed balance. V4 resolves that specific uncertainty. Once measured state and team-form
differences were balanced with overlap weights—and checked again within the same series and acting
roster—the player profiles still did not improve short-horizon success prediction.

The state-only model being best also shows that acting-team history was not secretly carrying the
V4 result. Adding team form slightly worsened prediction, and adding player profiles worsened it
further.

## Scientific conclusion

This is a stronger negative result than V3, but it remains bounded:

> Within the measured-overlap population of established 2025 RLCS professionals, chronology-safe
> actor and last-defender profiles and their frozen interactions did not add robust incremental
> prediction of short-horizon attacking success beyond the physical opportunity state.

It does not prove that player matchups never matter, identify a causal player effect, or transfer
directly to football. It does show that the V3 matching failure was not merely concealing a positive
result under the prespecified V4 formulation. There is no scientific justification for another
target or larger model on this RLCS corpus.

## Integrity record

- Pre-outcome implementation commit: `841c94f3aff721d5432a1f999df8c2473cd876c6`.
- Full repository verification before freeze: 440 tests passed; repo-wide Ruff passed.
- Support stage: approximately 4.7 seconds locally.
- Outcome labeling and grouped model analysis: approximately 17.6 seconds locally.
- Replays labeled: 198; failures: 0.
- Opened stages: Split 1 `train` and `internal_development` only.
- Action labels loaded: no.
- Split 2 Regional 1 validation loaded: no.
- Split 2 Regionals 2-3 test loaded: no.

The durable machine-readable stop record is
`provenance/rlcs_last_defender_overlap_value_v4_stop.json`. Generated Parquet and JSON evidence
remains under the ignored `data/processed/rlcs_last_defender_overlap_value_v4/` directory.
