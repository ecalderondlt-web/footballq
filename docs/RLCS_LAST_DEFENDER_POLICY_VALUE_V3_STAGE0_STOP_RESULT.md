# RLCS Last-Defender Policy-Value V3: Stage 0 Stop Result

Status: final RLCS stop at the preregistered common-support gate.

## Result in simple terms

The replay data contained plenty of last-defender situations. The detector found 5,256 eligible
opportunities involving all 48 established players, with strong EU/NA coverage and no player
dominating the sample.

The problem was not compute, data volume, or profile stability. The problem was comparison
quality. Even after the frozen state-matching procedure, favorable player-versus-defender profile
matchups remained strongly associated with the acting team's prior strength. Those situations
therefore were not balanced enough to isolate a matchup effect from team quality.

The maximum allowed absolute standardized mean difference was 0.10 for every matching feature.
Actor-team prior win rate was 0.2213 and actor-team prior goal difference was 0.2060. The gate
failed. Per the pre-outcome protocol, action labels and success outcomes were not opened, no V3
model was trained, and Split 2 remained sealed.

## Frozen gate results

| Gate | Requirement | Observed | Result |
|---|---:|---:|---|
| accepted opportunities | at least 1,000 | 5,256 | pass |
| distinct actors | at least 30 | 48 | pass |
| distinct last defenders | at least 30 | 48 | pass |
| EU opportunities | at least 300 | 2,525 | pass |
| NA opportunities | at least 300 | 2,731 | pass |
| maximum actor share | at most 10% | 4.81% | pass |
| maximum defender share | at most 10% | 4.30% | pass |
| matched opportunity sets | at least 300 | 2,010 | pass |
| overlap-weight effective sample size | at least 400 | 4,620.57 | pass |
| propensities in `[0.10, 0.90]` | at least 70% | 99.43% | pass |
| every matching-feature absolute SMD | at most 0.10 | maximum 0.2213 | **fail** |

The two features over the limit were:

- actor-team prior win rate: 0.2213;
- actor-team prior goal difference: 0.2060.

All other matching-state features were at or below 0.0663. This makes the failure specific rather
than a general lack of geometric overlap: profile-defined matchup favorability remained entangled
with the acting team's historical strength.

## What this does and does not mean

This result does not prove that player matchups never matter. It says the frozen observational RLCS
comparison cannot distinguish the intended player-versus-defender mechanism from team strength
without breaking the preregistered balance rule.

V1 found no useful identity-conditioned improvement for the next global interaction. V2 found
stable player traits but no incremental improvement for ten-touch scoring outcomes. V3 found
enough narrow critical opportunities, but the matchup comparison was not identifiable at the
outcome-blind common-support gate.

The correct final conclusion for this substrate is:

> Across immediate transitions, ten-touch scoring outcomes, and a mechanically defined
> last-defender opportunity, stable player-specific information did not yield a robust,
> identifiable increment in tactical value from these observational RLCS replays.

There will be no fourth RLCS target, no post-result threshold adjustment, and no larger RLCS model.
A future dataset must provide interventions from the same starting state or substantially deeper
repeated matchup histories.

## Integrity and compute record

- pre-outcome implementation commit: `60c73cac799cd90e1983a7f59323c60f7025205c`;
- local compute: viable, four workers, approximately 3.2 minutes, zero replay failures;
- opened partitions: Split 1 `train` and `internal_development` only;
- action labels loaded: no;
- success labels loaded: no;
- Split 2 Regional 1 validation loaded: no;
- Split 2 Regionals 2 and 3 test loaded: no;
- downstream training started: no.

The durable machine-readable record is
`provenance/rlcs_last_defender_policy_value_v3_stop.json`. It contains the frozen hashes,
calibration thresholds, all opportunity/support gates, every matching-feature balance statistic,
artifact hashes, and split-lock flags. The underlying generated Parquet and JSON artifacts remain
under the ignored `data/processed/rlcs_last_defender_policy_value_v3/` directory.
