# RLCS Player-Matchup Critical Value V2: Profile-Gate Result

Historical status: this documents the required stop under the original protocol. The user then
authorized the separately documented pre-outcome
`RLCS_PLAYER_MATCHUP_VALUE_V2_AMENDMENT_01.md`. No V2 outcome row had been constructed when that
amendment was frozen. This file has not been rewritten as though the original 60-player gate
passed. The post-amendment result is documented in
`RLCS_PLAYER_MATCHUP_VALUE_V2_REDUCED_COHORT_PILOT_RESULT.md`.

## Decision

**Stop before outcome-dataset construction and model training.**

The V2 player profiles are measurably persistent, but the frozen support period contains too few
players with the required history. The failure is the preregistered sample-size gate, not local
hardware and not profile instability. Split 2 Regional 1 validation and Split 2 Regionals 2/3 test
were not loaded.

## Real-data profile gate

The chronology manifest assigned the earliest 35 Split 1 series per region to profile support.
After applying the existing V1 replay and identity quality decisions, this produced:

| Quantity | Result | Required | Gate |
|---|---:|---:|---|
| Accepted support replays | 279 | - | descriptive |
| Support players | 102 | - | descriptive |
| Players with at least 15 support games | **48** | **60** | **fail** |
| Eligible EU players | 24 | - | descriptive |
| Eligible NA players | 24 | - | descriptive |
| Same-player early/late retrieval AUC | **0.9258** | 0.75 | pass |
| Median core-trait split-half Spearman | **0.6362** | 0.35 | pass |

Twelve of the thirteen reported core traits had positive early/late correlations; the median was
well above threshold. Forward ball velocity per touch was the weakest individual trait at 0.286,
but the protocol gates the median, not every feature separately.

The scientific interpretation is narrow: interpretable prior-game profiles can identify stable
player behavior in this corpus, but the fixed 35% support window does not provide the 60-player
effective sample required to proceed with a matchup outcome model.

Machine-readable local evidence:

- `data/processed/rlcs_player_matchup_value_v2/profile_stability_audit.json`
- `data/processed/rlcs_player_matchup_value_v2/profile_build_manifest.json`
- `runs/rlcs_player_matchup_value_v2/v2_stop_summary.json`

These paths are intentionally ignored by Git because they contain generated data/run artifacts.

## V1 validation-only outcome-head diagnostic

The prescribed post hoc diagnostic evaluated all 12 frozen V1 checkpoints on the already-open
validation split only:

- 18,879 validation decisions;
- 6,302 critical-state decisions;
- 5,595 critical decisions with all identities known;
- goal-within-eight-seconds prevalence: 3.80% overall and 3.76% in critical states.

Across seeds, the full V1 model averaged goal log loss 0.2976, Brier score 0.0711, and average
precision 0.0581. Anonymous averaged 0.2979, 0.0712, and 0.0573 respectively. Those tiny,
inconsistent differences are not evidence for identity conditioning. A validation-prevalence
constant predictor has post hoc log loss 0.1615 and Brier score 0.0365, so the underweighted V1
goal heads were badly calibrated even though their ranking average precision was above prevalence.

This diagnostic does not alter the V1 conclusion: the checkpoints were selected on next-touch
NLL, not goal prediction, and the goal head used a small focal-loss weight. It supports replacing
the target/training objective rather than treating V1's auxiliary goal head as a result.

Machine-readable local evidence:

- `runs/rlcs_identity_matchup_v1/v1_outcome_head_audit.json`

## Implementation delivered

The repository now contains the frozen V2 protocol and configuration; chronological split and
empirical-Bayes profile code; ten-touch labels with reset/censoring protection; explicit
actor-versus-each-opponent modeling; matched ablations and profile shuffles; training and
evaluation code; series-bootstrap gates; and a one-use sealed-test lock. Synthetic tests cover
chronology, stability, target semantics, future leakage, matchup ablations, and test access.

The downstream builder and trainer enforce the failed profile gate and therefore refuse to spend
compute unless the protocol is explicitly revised. No outcome rows, V2 checkpoints, validation
result, test unlock, or test result were created.

## Local viability

The support profile build completed on the current device in approximately 42 seconds, and the
12-checkpoint V1 validation audit completed on the installed CUDA environment in approximately
41 seconds. Local execution is viable. Work stopped because the frozen scientific gate failed,
not because the device lacked compute.

## Reproduction commands

```powershell
$env:PYTHONPATH = "src"

python scripts/audit_rlcs_v1_outcome_heads.py `
  --config configs/rlcs_player_matchup_value_v2.yaml

.\.venv\Scripts\python.exe scripts/build_rlcs_player_profiles.py `
  --config configs/rlcs_player_matchup_value_v2.yaml `
  --stage profile_support

.\.venv\Scripts\python.exe scripts/audit_rlcs_player_profiles.py `
  --config configs/rlcs_player_matchup_value_v2.yaml

.\.venv\Scripts\python.exe scripts/summarize_rlcs_value.py `
  --config configs/rlcs_player_matchup_value_v2.yaml
```

The first command needs the CUDA-enabled Python environment for the observed runtime. Profile
construction and auditing are CPU/data-processing workloads.
