# RLCS Player-Matchup Critical Value V2: Reduced-Cohort Pilot Result

## Simple result

**The amended 48-player profile gate passed, but the V2 outcome experiment stopped on Split 1
internal development because the full matchup model was worse than the simpler baselines in both
completed seeds.**

In plain terms, the telemetry contains stable, recognizable player-style profiles. However, this
experiment did not show that combining a specific actor profile with specific opponent profiles
improves ten-touch scoring/conceding prediction. The failure is scientific, not computational:
the current laptop was fast enough and had ample GPU memory.

Split 2 Regional 1 validation was not opened. Split 2 Regionals 2 and 3 remain sealed. No
architecture was frozen and no test unlock was created.

## Pre-outcome amendment and profile gate

Amendment 01 was frozen before any V2 outcome dataset or outcome-model result existed. It replaced
the failed arbitrary 60-player count with the user-authorized complete-cohort uncertainty gate.
All 48 eligible players were retained without manual selection.

| Amended requirement | Observed | Decision |
|---|---:|---|
| Complete eligible cohort | exactly 48 | pass |
| Prior games per eligible player | minimum observed 18; required at least 15 | pass |
| EU eligible players | 24; required at least 20 | pass |
| NA eligible players | 24; required at least 20 | pass |
| Same-player retrieval AUC | 0.9258; required at least 0.75 | pass |
| AUC player-bootstrap 95% lower bound | 0.8707; required at least 0.65 | pass |
| EU retrieval AUC | 0.8750; required positive effect | pass |
| NA retrieval AUC | 0.9688; required positive effect | pass |
| Median split-half Spearman | 0.6362; required at least 0.35 | pass |
| Spearman player-bootstrap 95% lower bound | 0.5217; required above 0.20 | pass |

This passes the amended stability gate only. It shows that continuous, past-only profiles are
persistent enough to test; it does not by itself show matchup prediction value.

## Outcome-data gate

Only the authorized Split 1 training and internal-development stages were constructed. Every
accepted replay completed without a construction failure.

| Stage | Accepted replays | Rows | No goal | Score | Concede |
|---|---:|---:|---:|---:|---:|
| Training | 349 | 59,086 | 46,815 | 7,118 | 5,153 |
| Internal development | 168 | 28,028 | 22,247 | 3,335 | 2,446 |

The unchanged training gate required at least 5,000 `score` rows and 5,000 `concede` rows. It
passed, so matched outcome training was authorized.

## Internal-development stop

The five frozen conditions were trained with identical data, capacity, optimizer schedule, and
seed. Lower three-class log loss is better.

| Seed | State | Team form | Actor profile | Additive profiles | Full matchup | Full vs team form |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.637813 | 0.637665 | 0.639588 | 0.644050 | **0.648368** | **-1.68%** |
| 23 | 0.637704 | 0.639873 | 0.640860 | 0.641655 | **0.643775** | **-0.61%** |

The required gate was a full-model improvement of at least 2% over team form in at least two of
three seeds. Seeds 17 and 23 both failed, leaving only seed 41. Even a seed-41 success could have
produced at most one passing seed, so the required two passing seeds became mathematically
impossible.

Training was therefore stopped immediately under the unchanged rule. A provisional seed-41 state
checkpoint had been written just before process termination, but the run never produced a run
manifest and is explicitly excluded from the result. It was not used for model selection or any
comparison.

The full model also had the worst point-estimate log loss of all five conditions in both completed
seeds. Profile-shuffle, official-series confidence-bound, calibration, and regional controls were
not run because the primary seed gate had already failed irreversibly. They are recorded as
unevaluated, not passed. Continuing those controls could not rescue the failed two-of-three-seeds
requirement.

## Scientific conclusion

This reduced-cohort pilot supports only the narrow claim that past-only behavioral profiles are
stable among the 48 established RLCS professionals in the available cohort. It provides no
positive evidence that the implemented actor-opponent profile interaction improves short-horizon
critical-value prediction over state and team form.

The result does not establish that player-specific matchup value never exists. It rejects this
particular V2 formulation, target, data window, and frozen evaluation path strongly enough that
Split 2 should remain untouched. Any follow-up should be a newly documented experiment rather
than tuning V2 against its internal result.

## Local-compute result

The real V2 benchmark ran on the NVIDIA GeForce RTX 5070 Ti Laptop GPU at 13.6 training steps per
second, about 3,484 samples per second, with approximately 1.0 GB peak allocated GPU memory. Data
construction used four CPU workers and completed cleanly. Local compute was viable; cloud compute
would not change the failed scientific gate.

## Evidence

Tracked protocol and implementation evidence:

- `docs/RLCS_PLAYER_MATCHUP_VALUE_V2_AMENDMENT_01.md`
- `provenance/rlcs_player_matchup_value_v2_amendment_01.json`
- `provenance/rlcs_player_matchup_value_v2_stop.json`
- `configs/rlcs_player_matchup_value_v2.yaml`
- `scripts/build_rlcs_value_dataset.py`
- `scripts/train_rlcs_value.py`
- `scripts/summarize_rlcs_value.py`

The tracked stop ledger freezes the ignored machine evidence needed to reproduce the decision,
including all ten completed checkpoint and run-manifest hashes. The underlying generated evidence
remains intentionally ignored by Git:

- `data/processed/rlcs_player_matchup_value_v2/profile_stability_audit.json`
- `data/processed/rlcs_player_matchup_value_v2/dataset_manifest.json`
- `runs/rlcs_player_matchup_value_v2/v2_stop_summary.json`
- the ten completed run manifests and hashed checkpoints under
  `runs/rlcs_player_matchup_value_v2/`

The stop summary records `opened_stages = [train, internal_development]`,
`validation_loaded = false`, and `test_loaded = false`.
