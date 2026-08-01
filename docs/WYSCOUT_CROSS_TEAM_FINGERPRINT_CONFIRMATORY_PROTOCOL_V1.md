# Wyscout Cross-Team Player Fingerprint Confirmatory Protocol V1

## Status

This protocol is frozen before the first confirmatory retrieval run.

- Protocol: `wyscout_cross_team_player_fingerprint_v1`
- Freeze date: 2026-07-29
- Development cohort: metrics opened
- Confirmatory cohort: retrieval metrics sealed
- Confirmatory access allowed: exactly once, through the guarded runner
- Data cost: USD 0
- Expected local compute cost: USD 0

The confirmatory cohort has previously been inspected only for source
availability, chronology, match counts, pass counts, and the number of eligible
overlapping identities. No confirmatory player vectors, similarities, ranks,
AUCs, top-k scores, or MRR values have been computed or inspected.

## Research Question

Does an individual football player's on-ball behavior remain identifiable after
the player changes from a club team to a national team, or from a national team
to a club team?

The primary hypothesis is:

> An outcome-free event-history fingerprint, built from where and how a player
> passes, contains persistent player-specific signal that survives a complete
> team-context change.

This is a prerequisite test for player-conditioned football modeling. If a
player profile is only a proxy for the current team or role, conditioning a
future tracking model on that profile is unlikely to add genuine information.

## Claim Boundary

If the confirmatory gate passes, the allowed claim is:

> Outcome-free pass behavior contains reproducible player-specific information
> that persists across club and national-team contexts.

The experiment does not establish:

- improved critical-event prediction;
- off-ball player understanding;
- tracking-based tactical understanding;
- opponent-specific matchup reasoning;
- full-match planning;
- causal effects of a player's identity;
- superiority of the current TD-JEPA representation.

Those remain separate downstream questions.

## Data And License

The experiment uses the public Wyscout event dataset released by Pappalardo et
al. under CC BY 4.0.

- Collection DOI: `10.6084/m9.figshare.c.4415000`
- Dataset paper DOI: `10.1038/s41597-019-0247-7`
- Source provenance: `provenance/wyscout_public_source_v1.json`
- Processed manifest:
  `data/processed/wyscout_player_memory_v1/manifest.json`
- Processed manifest SHA-256:
  `e28f473bb63c52c07fad218fa2584df22dbb3dec9f4f0b59c5ef1c76c2c86c8b`

The compact pass table contains period-aware identities of the form:

```text
match_id:period:event_id
```

No broadcast video, computer-vision reconstruction, paid API, scraped player
attribute site, or manual event annotation is used.

## Development Cohort

Support:

- all 1,826 matches from England, France, Germany, Italy, and Spain in 2017/18;
- latest support match: 2018-05-20;
- support eligibility: at least 100 passes in at least five matches.

Query:

- all 64 matches from the 2018 World Cup;
- earliest query match: 2018-06-14;
- query eligibility: at least 20 passes in at least one match.

There is a strict chronological gap. The primary candidate pool retains every
eligible support player, including players who did not appear in the World Cup.
The query roster is not used to make the primary retrieval task easier.

Observed development population:

- 2,062 eligible support candidates;
- 269 eligible query players;
- 100% changed team ID between support and query;
- about 651 candidates per query after same-role restriction.

## Confirmatory Cohort

Support:

- all 51 matches from Euro 2016;
- latest support match: 2016-07-10;
- support eligibility: at least 20 passes in at least one match.

Query:

- all 1,826 domestic-league matches from 2017/18;
- earliest query match: 2017-08-04;
- query eligibility: at least 100 passes in at least five matches.

This reverses both the temporal direction and the football context used in
development: national team to later club football instead of club football to
later national-team football.

The eligibility audit found 194 identities with sufficient data in both
periods. This count is cohort metadata, not a retrieval result.

The frozen match manifest is:

`splits/wyscout_player_fingerprint_euro_to_club_confirmatory_v1.json`

Its file SHA-256 before unsealing is:

`eba6abc965e4cd6edc632519e266c0b0ac026c05125f7f5b357c85bb06e11c6a`

## Primary Fingerprint

The primary `behavior_72` fingerprint contains no outcome-derived feature.

It consists of:

1. Twenty-two summary behavior features:
   mean start and destination coordinates, progression, absolute lateral
   displacement, pass length, progression and lateral variability, forward,
   backward and long-pass rates, seven pass-subtype rates, and attacking-half,
   final-third, and penalty-area location rates.
2. A 30-bin destination-location histogram.
3. A 20-bin start-location histogram.

The following three features are excluded from the primary fingerprint:

- pass accuracy;
- key-pass rate;
- shot within the future event horizon.

They form the `outcome_3` negative-control view.

## Preprocessing

For each support-size condition:

1. Select the player's most recent `K` support matches.
2. Aggregate one vector per player.
3. Fit feature means and standard deviations on support players only.
4. Z-score support and query vectors with those support statistics.
5. Fit broad-role means on support players only.
6. Subtract the applicable support role mean.
7. L2-normalize each vector.
8. Compare support and query vectors with cosine similarity.

The four broad roles are goalkeeper, defender, midfielder, and forward.
Unknown roles remain explicit rather than being silently reassigned.

No query statistic is used to fit normalization, residualization, or a model.

## Candidate Sets

The primary ranking compares each query player with every eligible support
player having the same broad role.

Two stricter confound controls are also required:

- `same_support_team_and_role`: candidates shared the true player's support
  club or national team and broad role;
- `same_query_team_and_role`: candidates appear in the same query team and
  broad role.

The first control holds support-team style approximately constant. The second
holds query-team style approximately constant. A positive result on both is
needed to argue that retrieval is not merely team recognition.

## Metrics

The primary metrics are:

- same-role pairwise AUC;
- top-1, top-3, top-5, and top-10 retrieval;
- mean reciprocal rank;
- candidate-set-specific analytic chance levels;
- player-bootstrap confidence intervals.

All uncertainty uses 5,000 bootstrap resamples of player identities at the 95%
level. Passes are not treated as independent uncertainty units.

## Support-Size Test

Development support sizes are 1, 3, 5, 10, and 20 matches. The primary
development condition is 20.

Confirmatory support sizes are 1, 3, and 5 matches. The primary confirmatory
condition is 5.

The main support-size MRR must exceed the one-match MRR with a strictly positive
95% player-bootstrap lower bound. This tests whether repeated history improves
the fingerprint rather than merely providing a lucky first-match signal.

## Frozen Confirmatory Gate

The confirmatory result passes only if every condition below passes:

1. Same-role pairwise AUC is at least 0.65.
2. Same-role top-1 is at least five times its candidate-specific chance level.
3. Same-support-team-and-role top-1 is at least 1.5 times chance.
4. Same-query-team-and-role top-1 is at least 1.5 times chance.
5. The player-bootstrap lower bounds for top-1 minus chance and MRR minus
   chance are positive for all three candidate sets.
6. The five-match minus one-match MRR bootstrap lower bound is positive.

Failure of any condition means the confirmatory gate fails. Thresholds, feature
views, support sizes, and preprocessing may not be changed after unsealing.

## Development Results

The outcome-free primary view passed all six development checks.

Primary same-role result at 20 support matches:

- pairwise AUC: 0.8374;
- AUC 95% CI: [0.8123, 0.8606];
- top-1: 4.46%;
- candidate-specific top-1 chance: 0.186%;
- top-1 chance multiple: 23.98;
- top-5: 16.73%;
- top-10: 24.16%;
- MRR: 0.1136;
- chance MRR: 0.0125.

Team-confound controls:

- same support team and role: 56.70% top-1 versus 16.33% chance;
- same query team and role: 68.60% top-1 versus 26.03% chance;
- both top-1-minus-chance bootstrap lower bounds are positive.

Support size:

- one-match MRR: 0.0727;
- twenty-match MRR: 0.1136;
- paired MRR gain: 0.0409;
- gain 95% CI: [0.0154, 0.0666].

Role audit:

| Role | Players | Pairwise AUC | AUC 95% lower | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| Goalkeeper | 13 | 0.6783 | 0.4837 | 0.00% |
| Defender | 92 | 0.8462 | 0.8039 | 5.43% |
| Midfielder | 104 | 0.8523 | 0.8198 | 2.88% |
| Forward | 60 | 0.8325 | 0.7744 | 6.67% |

The outfield result is broad. The goalkeeper subgroup is small and
underpowered; no positive goalkeeper-specific claim is allowed.

Feature controls:

| View | Top-1 | MRR | Pairwise AUC |
| --- | ---: | ---: | ---: |
| Primary behavior-only 72 | 4.46% | 0.1136 | 0.8374 |
| Full 75 | 5.20% | 0.1179 | 0.8412 |
| Summary behavior 22 | 5.58% | 0.1113 | 0.8324 |
| Spatial histograms 50 | 2.97% | 0.0876 | 0.8128 |
| Outcome-only 3 | 0.37% | 0.0229 | 0.6889 |

The primary signal therefore does not depend on pass success, key-pass tags, or
future shot-chain outcomes.

## Development Artifact Identity

The final development run used:

- result file SHA-256:
  `54a7883bef38bf2714fd54ffccd554f4ef551fc5ca828e416ccb114c74fde76e`;
- result payload SHA-256:
  `b1f07058d980936066fd927fe187a85f891d49d6ff7eaa01c6c8f37051b7ae8b`;
- config SHA-256:
  `bcd4f21801647c7454b0e1b9631a3f178f996d03b422f01f75ff3ce2bce486d7`;
- analysis source SHA-256:
  `99a483bc4dc86140822dcf3f462fe8d15d0328544c9c8dc49029d6ecb9ff97f9`;
- runner source SHA-256:
  `ae1b03f615556ab87d10ec027543f498647b0784409f2229801a25a979a4188a`.

The machine-readable freeze manifest is the final authority. It is generated
after this document and verifies every file again before confirmatory access.

## Novelty Boundary

Player retrieval is not new.

Decroos and Davis introduced Player Vectors and evaluated 741 players across
consecutive seasons, but required each player to remain with the same team in
both seasons. Their reported Manhattan-distance top-1 was 38.2% and MRR was
0.469 using much denser season-level histories.

Kim et al. later used tracking heatmaps and triplet learning to identify players
from fewer matches while controlling tactical role. That study used private
wearable tracking and did not test a club-to-national-team context change.

The intended contribution here is narrower:

- an explicit cross-team identity test;
- club-to-country development and country-to-club replication;
- all eligible support players retained as distractors;
- support-team and query-team confound controls;
- outcome-free primary features;
- a support-size dose-response test;
- a fully public, zero-license-cost reproduction path.

The paper must use language such as "to our knowledge" until a final systematic
literature search is complete.

## One-Time Unseal Procedure

The only allowed command is:

```powershell
$env:PYTHONPATH='src'
python scripts/run_wyscout_player_fingerprint_v1.py --unseal-confirmatory
```

Before reading confirmatory events, the runner must:

1. verify the development gate;
2. verify the frozen config hash;
3. verify hashes for code, data manifest, both match manifests, this protocol,
   and the development artifacts;
4. create `UNSEAL_STARTED.json` using exclusive file creation;
5. refuse to run if a sentinel or confirmatory result already exists.

After a successful run it writes `UNSEAL_COMPLETED.json`. A failed run after
the start sentinel still counts as opening the cohort and may not be silently
rerun.

## Budget

The complete empirical core is designed to remain below USD 500:

| Item | Planned maximum |
| --- | ---: |
| Public Wyscout data | USD 0 |
| Local CPU development and confirmatory runs | USD 0 |
| Optional cloud reproduction | USD 25 |
| Storage and backups | USD 25 |
| Independent reproduction or research assistance | USD 200 |
| Figures, copy editing, and contingency | USD 250 |
| Total hard ceiling | USD 500 |

This excludes conference travel and optional open-access publication charges.
The scientific experiment itself requires no broadcast-video processing and no
GPU.

## Decision After Confirmation

If the gate passes:

1. Treat persistent cross-team player information as demonstrated.
2. Write this as the paper's primary low-cost empirical result.
3. Add a secondary, separately labeled experiment asking whether support-only
   profiles improve held-out action-choice prediction beyond context and role.
4. Keep the existing null critical-outcome and tracking-profile experiments in
   the paper to show where the evidence stops.

If the gate fails:

1. Report the development result as non-replicating.
2. Do not claim persistent cross-team player information.
3. Do not spend the remaining budget on broadcast CV merely to rescue the
   result.

## References

- Pappalardo, L. et al. A public data set of spatio-temporal match events in
  soccer competitions. *Scientific Data* 6, 236 (2019).
  https://doi.org/10.1038/s41597-019-0247-7
- Decroos, T. and Davis, J. Player Vectors: Characterizing Soccer Players'
  Playing Style from Match Event Streams. ECML PKDD (2019).
  https://doi.org/10.1007/978-3-030-46133-1_34
- Kim, H. et al. 6MapNet: Representing Soccer Players from Tracking Data by a
  Triplet Network. MLSA (2021).
  https://dtai.cs.kuleuven.be/events/MLSA21/papers/MLSA21_paper_kim.pdf
