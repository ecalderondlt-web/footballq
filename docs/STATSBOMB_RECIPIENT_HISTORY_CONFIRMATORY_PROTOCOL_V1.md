# StatsBomb Recipient-History Confirmatory Protocol V1

Status: metric sealed
Protocol: `frozen_online_recipient_history_tournament_replication_v1`
Freeze date: 2026-07-29

## 1. Narrow Research Question

Does a player's strictly prior, origin-zone-specific receiving history improve
the probability ranking of the actual pass recipient beyond:

1. the current pass origin zone;
2. broad role;
3. current fine-grained lineup position;
4. static player identity learned on training teams; and
5. the player's rolling reception frequency?

The primary endpoint is recipient negative log likelihood (NLL), a proper
probability-ranking score. Top-1, top-3, and mean reciprocal rank are secondary
and are not confirmatory gates.

This is an event-choice experiment. It does not test continuous tracking,
five-second critical-event prediction, opponent matchup understanding, causal
tactical intervention, or full-match planning.

## 2. Why This Protocol Exists

The original development gate required a two-percentage-point top-3 gain. That
gate remained blocked. After rebuilding the cache with the complete shuffled
profile control, the Leverkusen development result still showed a positive NLL
effect, but only a 0.55 percentage-point top-3 gain.

Before opening any tournament model score, the claim was therefore narrowed to
probability ranking. This change and the failed original gate must remain in all
paper reporting.

## 3. Data and Provenance

Source: StatsBomb Open Data, pinned source commit
`b0bc9f22dd77c206ddedc1d742893b3bbe64baec`.

The source inventory covers `competitions.json` plus every file under
`events`, `lineups`, `matches`, and `three-sixty`:

- files: 8,977
- bytes: 16,126,670,930
- inventory payload SHA-256:
  `929bb5ae690794ea38afe6d3053d3380d3a2c162a8ba4e72e0ad214472be75a8`
- inventory file SHA-256:
  `db3290f53f2b9e594f57e40a7a35cfd9811ab35b500c849a300ec73d846a7680`

Confirmatory split manifest:
`splits/statsbomb_recipient_history_confirmatory_v1.json`

- manifest SHA-256:
  `669efdc83d8425a2bcf3f893d9e4c1c002a31f978d0f6a22cd119de8ef641b77`
- train matches: 61
- validation matches: 32
- primary confirmatory matches: 51
- external replication matches: 31

StatsBomb must be credited as the data source in any publication, as required
by the source repository.

## 4. Cohorts

Model priors:

- Barcelona, La Liga 2020/2021: train
- Paris Saint-Germain, Ligue 1 2021/2022: train

Model selection:

- Paris Saint-Germain, Ligue 1 2022/2023: validation

Opened development test:

- Bayer Leverkusen, Bundesliga 2023/2024

Sealed tests:

1. UEFA Euro 2024, all 51 matches: primary confirmation
2. UEFA Women's Euro 2025, all 31 matches: external replication

All sealed matches have a local StatsBomb 360 file. The model uses 360
availability only to define the same query-event universe used in development;
it does not consume anonymous freeze-frame coordinates as input.

## 5. Sample and Chronology Contract

Each sample is a completed open-play pass with:

- a known actor;
- a known recipient;
- at least two active teammate candidates; and
- a corresponding StatsBomb 360 event identifier.

Sample identity is `match_id:period:event_uuid`.

For every query:

```text
support_match_date < query_match_date
```

Matches on the same date cannot enter one another's support. A player's
appearance enters support even if that player received zero passes, preventing
zero-event matches from silently disappearing.

## 6. Frozen Models

The baseline starts with the current origin-zone and broad-role prior, then adds:

- fine-position weight: 0.25
- static-identity weight: 0.00
- rolling reception-frequency weight: 0.25
- support size: 5 prior appearances

The profile model adds:

- origin-zone-specific receiving-profile weight: 0.25
- empirical-Bayes profile prior strength: 10.0

No hyperparameter, support size, feature, threshold, or target may be selected
on Euro 2024 or Women's Euro 2025.

## 7. Falsification Controls

The primary substitution control replaces each candidate's receiving history
with another active candidate from the same broad role while preserving:

- the query event;
- candidate set;
- current role and fine position;
- rolling candidate availability;
- train-fitted priors; and
- all labels.

A same-fine-position substitution is also reported, but it is diagnostic only
because exact StatsBomb positions are usually unique within a lineup and the
development assignment fraction was under 1%.

The complete support-size curve at `K = {1, 3, 5, 10, 20}` is reported with
frozen weights. It is not used to change the five-match confirmatory model.

## 8. Development Evidence Used to Authorize the Narrow Test

Leverkusen development test:

- eligible passes: 9,634 across 34 matches
- rolling NLL: 2.286814
- profile NLL: 2.274910
- absolute NLL improvement: 0.011904
- relative NLL improvement: 0.5206%
- match-bootstrap 95% CI: [0.008194, 0.015682]
- rolling top-3: 41.59%
- profile top-3: 42.14%
- top-3 gain: 0.55 percentage points
- top-3 bootstrap 95% CI: [-0.12, 1.20] percentage points
- same-broad-role shuffled-profile NLL: 2.337930

Frozen-weight development NLL gain by support size:

| Prior matches | NLL gain |
|---:|---:|
| 1 | 0.005908 |
| 3 | 0.010919 |
| 5 | 0.011904 |
| 10 | 0.013201 |
| 20 | 0.012339 |

Development result payload SHA-256:
`3677131bbdb311071f9271939e04b78820105e6baad470979a6871b3c4a059ce`

## 9. Confirmatory Gates

All of the following must pass:

1. Recomputed validation metrics exactly reproduce the frozen development
   artifact to absolute tolerance `1e-12`.
2. Euro 2024 relative NLL improvement is at least 0.25%.
3. Euro 2024 match-bootstrap NLL-gain lower bound is greater than zero.
4. The genuine Euro 2024 profile has lower NLL than its same-broad-role
   substituted-history control.
5. The pooled 82-match bootstrap NLL-gain lower bound is greater than zero.
6. Women's Euro 2025 has positive point NLL improvement.
7. The genuine Women's Euro 2025 profile has lower NLL than its
   same-broad-role substituted-history control.

Failure of any gate blocks the broad confirmatory claim. All metrics, including
failures and null support-size points, must still be reported.

## 10. Uncertainty

- resampling unit: match
- bootstrap replicates: 2,000
- confidence interval: percentile 95%
- frozen seed: 20260729

Event-level resampling is not used for the primary inference because passes from
the same match are dependent.

## 11. Pre-Freeze Access Audit

Before freeze, the following sealed-cohort information was accessed:

- match identifiers and dates;
- lineups;
- 360 file existence;
- aggregate match counts; and
- raw file bytes solely to compute cryptographic hashes.

No recipient probability, rank, NLL, top-k score, bootstrap effect, profile
substitution score, or support-size curve was computed for either sealed
cohort.

Euro 2024 events were used earlier in a separate Wyscout player-fingerprint
experiment only through a different provider's event data. No StatsBomb
recipient-choice metric was opened. This recipient experiment is confirmatory
for its narrow metric but is secondary evidence within the broader project.

## 12. Cost and Reproducibility

The experiment uses already downloaded public JSON and CPU evaluation. Expected
incremental cloud cost is zero; a conservative complete-paper compute allowance
is under USD 50. This leaves the project safely below the USD 500 ceiling
without broadcast-video reconstruction.

The one permitted command after the freeze manifest exists is:

```powershell
$env:PYTHONPATH='src'
python scripts/run_statsbomb_recipient_history_confirmatory_v1.py --unseal-confirmatory
```

The runner writes `UNSEAL_STARTED.json` before loading tournament recipient
metrics and refuses every rerun, including after a failed or interrupted first
attempt.
