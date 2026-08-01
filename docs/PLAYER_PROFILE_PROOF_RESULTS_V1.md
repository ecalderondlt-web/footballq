# Frozen Player-Profile Proof V1

## Decision

**No-go for full identity-aware TD-JEPA pretraining from this experiment.**

History-derived TD-JEPA player profiles did not provide reliable incremental
value over current geometry, rolling event statistics, static identity, and a
same-role shuffled-history control. This is a null result for the tested
profile construction and data regime, not evidence that player history can
never help.

## Question

Does a frozen TD-JEPA embedding of a player's strictly earlier World Cup
matches improve five-second turnover and penalty-area-entry prediction at the
start of a possession?

## Frozen Protocol

- Data: all 64 PFF FC World Cup 2022 tracking matches.
- Labels and stable player IDs: corresponding StatsBomb Open Data matches.
- Support: 16 first group-stage matches.
- Probe training: 16 second group-stage matches.
- Validation: 16 third group-stage matches.
- Test: 16 knockout matches.
- Chronology rule: `support_match_datetime < query_match_datetime`.
- Encoder: frozen position-only, non-overlapping TD-JEPA checkpoint at step
  2,000.
- Encoder inputs: 1 second at 10 fps, 23 entities, and five channels:
  normalized x/y, ball flag, home flag, and away flag.
- Player profile: mean per-player entity token over clips from the most recent
  `K` earlier matches, aligned to the player's current tracking slot.
- Support sizes tested: `K = 1, 2, 3, 5`.
- Main comparison: `K = 3`.
- Probe: class-balanced linear logistic regression.
- Unseen train-constant history fields use unit scaling. This prevents legal
  larger support counts in later chronological splits from numerical
  saturation.

The originally proposed `K = 10` and `K = 20` conditions are not available in
one World Cup because a player can play at most seven matches. This bounded
experiment therefore cannot test the large-history regime.

## Data Audit

| Split | Matches | Aligned opportunities | Turnover positives | Penalty-entry positives |
|---|---:|---:|---:|---:|
| Train | 16 | 737 | 26 | 37 |
| Validation | 16 | 702 | 11 | 29 |
| Test | 16 | 826 | 13 | 51 |

The profile builder covered 45,576 of 49,830 query player slots (91.46%).
Every query context ended before its possession start. The query match itself
and all later matches were excluded from each player profile.

## Main Test Results

### Turnover within five seconds

| Model | Macro-F1 | Average precision | Brier | Log loss |
|---|---:|---:|---:|---:|
| A. Raw geometry | 0.5199 | 0.0342 | 0.0787 | 0.4130 |
| B. Raw + role | 0.5286 | 0.0415 | 0.0572 | 0.3582 |
| C. Raw + current TD-JEPA latent | 0.5225 | 0.0400 | 0.0762 | 0.4063 |
| D. Raw + static identity | **0.5342** | **0.0443** | 0.0564 | 0.3664 |
| E. Raw + rolling event statistics, K=3 | 0.4914 | 0.0258 | **0.0288** | 0.2497 |
| F. Raw + player profiles, K=3 | 0.5036 | 0.0238 | 0.0515 | 0.3623 |
| Same-role shuffled profiles, K=3 | 0.4904 | 0.0241 | 0.0329 | **0.2122** |

For F versus E, the macro-F1 gain was `+0.0122`, below the `+0.02` gate.
Average precision, Brier score, and log loss were worse. The match-bootstrap
macro-F1 gain was `+0.0134`, with a 95% interval of `[-0.0151, +0.0704]`.
The interval crosses zero.

### Penalty-area entry within five seconds

| Model | Macro-F1 | Average precision | Brier | Log loss |
|---|---:|---:|---:|---:|
| A. Raw geometry | 0.5121 | 0.0892 | 0.1849 | 1.4997 |
| B. Raw + role | 0.5037 | 0.0899 | 0.1806 | 1.5372 |
| C. Raw + current TD-JEPA latent | 0.5251 | 0.0956 | 0.1726 | 1.3846 |
| D. Raw + static identity | **0.5434** | 0.0945 | 0.1286 | 1.0441 |
| E. Raw + rolling event statistics, K=3 | 0.5384 | 0.0914 | **0.0923** | 0.9511 |
| F. Raw + player profiles, K=3 | 0.5191 | 0.0871 | 0.1394 | 1.5560 |
| Same-role shuffled profiles, K=3 | 0.5289 | **0.1018** | 0.1330 | **0.8371** |

F was worse than E on every reported metric. Its macro-F1 difference was
`-0.0193`. The match-bootstrap mean was `-0.0203`, with a 95% interval of
`[-0.0561, +0.0154]`.

## Support-Size Result

More support did not produce a broadly improving curve.

| Target | F K=1 | F K=2 | F K=3 | F K=5 |
|---|---:|---:|---:|---:|
| Turnover macro-F1 | 0.4812 | 0.4979 | 0.5036 | 0.5016 |
| Penalty entry macro-F1 | 0.4748 | 0.5039 | 0.5191 | 0.5263 |

Penalty-entry macro-F1 rose with support, but it still remained below the
rolling-statistics and static-identity baselines and did not beat shuffled
history. Turnover flattened after K=3.

## Interpretation

In simple terms, the frozen per-player embeddings mostly act like extra noisy
lineup features in this setup. The test does not show that they encode a
stable, useful player style beyond easier information such as who is playing
and their recent event counts.

The result does **not** establish that player-conditioned modeling is a dead
end. The strongest limitations are:

1. Only one competition is available, with one prior match for training
   queries and at most five useful support matches for most test players.
2. The TD-JEPA encoder was trained for scene prediction, not identity
   consistency, whole-player masking, or same-role contrastive separation.
3. Profiles use simple means and slot-aligned linear probes; no learned
   attention profile pooler or lineup interaction encoder is present.
4. Turnover is extremely imbalanced in the aligned sample: only 13 positives
   in the 826-example test split.
5. The current test has now been opened and must be treated as exploratory for
   any follow-up tuning.

The next justified experiment is not full identity-aware pretraining. It is a
smaller representation diagnostic on train/validation only: test whether
same-player clips are more similar than carefully matched same-role players,
and whether a learned profile pooler can beat shuffled histories without
opening a new test set. If that diagnostic remains negative, the current
encoder should not be used as the basis for player profiles.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/run_player_profile_proof_v1.py --device auto --rebuild-cache
```

Key artifacts:

- Config: `configs/player_profile_proof_v1.yaml`
- Chronology manifest:
  `splits/pff_wc2022_player_profile_chronological_v1.json`
- Feature audit: `runs/player_profile_proof_v1/feature_audit.json`
- Results: `runs/player_profile_proof_v1/results.json`
- Feature cache: `runs/player_profile_proof_v1/feature_cache.pt`

Provenance hashes:

- Frozen checkpoint SHA-256:
  `f7aaefae797f39f5ffad57f66427712e384bf7bb4027a1295b05aa2ec1455a33`
- Experiment code SHA-256:
  `1c1b677cf0cf2aaa09055e2422ed50607088cc8586b8cc5582bcb8f1c4a11ecb`
- Config SHA-256:
  `8a13ad219960487f3f0dfc94c77d4d1769d5117c719232b0c77b0bf45f677e0f`
- Feature cache SHA-256:
  `8ee9d6acb8dd2390bee31dcae7f52bc321466741777bb3638072e26031977c0c`
- Results SHA-256:
  `2920b64efe73585da31b4da93d3ee203c2bde6a056d18992125a11e7d46ca4eb`
