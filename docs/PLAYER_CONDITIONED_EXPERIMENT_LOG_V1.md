# Player-Conditioned Experiment Log V1

## Purpose

This document closes the human-readable reporting gap for the player-conditioned
experiments run after the original PFF architecture studies. The exact local
result payloads live under `runs/`, which is intentionally Git-ignored. This
ledger records the research questions, cohorts, headline metrics, frozen gates,
interpretation limits, and reproducible source artifacts that must survive in
Git.

The ledger does not turn a development result into a confirmation. In
particular, player retrieval is evidence of persistent identity signal, not
evidence of tactical understanding or improved outcome prediction.

## Result Index

| Experiment | Data | Question | Status |
| --- | --- | --- | --- |
| Frozen player-identity diagnostic | PFF WC 2022 | Do TD-JEPA tokens preserve cross-match player identity? | Gate failed |
| Frozen player-profile proof | PFF plus StatsBomb | Do earlier TD-JEPA profiles improve five-second outcomes? | No-go |
| Club-to-country player memory | Wyscout | Does prior player history improve later action prediction beyond team history? | Gate failed |
| Cross-team player fingerprint | Wyscout | Is outcome-free pass behavior identifiable after a team change? | Confirmatory gate passed |
| Cross-team action prediction | Wyscout | Does that history improve destination or penalty-entry prediction? | Gate failed |
| Critical-event history signal | StatsBomb | Do player profiles improve progressive-pass or turnover prediction? | Development null |
| Recipient-history development | StatsBomb | Does earlier receiving history improve recipient ranking? | Original gate blocked |
| Recipient-history confirmation | StatsBomb | Does the narrowed NLL effect replicate in tournaments? | Confirmatory gate failed |
| Transfer diagnostics | StatsBomb | Does the history term survive alternate team/tournament cohorts? | Profile weight selected as zero |
| Player-history V1 | FOOTPASS | Does prior history improve action-location and turnover proxies? | Gate failed |
| Compact player residual V2 | FOOTPASS | Does a shrunk residual beat geometry, role, and shuffles? | Gate failed |
| Identity-matchup V1 | RLCS 2025 telemetry | Does actor and opponent identity improve next-touch prediction beyond complete geometry? | Validation gate failed; test sealed |

## 1. PFF Frozen Player Diagnostics

The PFF experiments already have full result handoffs:

- `docs/PLAYER_IDENTITY_DIAGNOSTIC_RESULTS_V1.md`
- `docs/PLAYER_PROFILE_PROOF_RESULTS_V1.md`

The identity diagnostic used the first 48 World Cup matches and same-team,
same-role retrieval. At the main two-history condition, raw kinematics reached
pairwise accuracy `0.5299`, while frozen TD-JEPA reached `0.4928`. The
player-bootstrap interval for TD-JEPA minus raw was `[-0.0722, -0.0019]`.
The gate failed.

The profile proof used all 64 matches with a chronological 16/16/16/16
support/train/validation/test design. At `K=3`, player profiles did not provide
reliable incremental value over current geometry, rolling event statistics,
static identity, or shuffled histories for turnover or penalty-area-entry
prediction. The result is a null for this frozen profile construction, not a
claim that player-conditioned modeling is impossible.

Reproducible sources:

- `configs/player_identity_diagnostic_v1.yaml`
- `configs/player_profile_proof_v1.yaml`
- `splits/pff_wc2022_player_profile_chronological_v1.json`
- `scripts/run_player_identity_diagnostic_v1.py`
- `scripts/run_player_profile_proof_v1.py`
- `src/footballq/analysis/player_identity_diagnostic.py`
- `src/footballq/analysis/player_profile_proof.py`

## 2. Wyscout Club-To-Country Player Memory

Protocol: `wyscout_player_memory_club_to_country_v1`.

Question: does a player's strictly earlier club history improve later
national-team action prediction beyond pooled context and the same club's
history?

The development cohort contained `58,551` query examples. The team-history
model achieved NLL `0.357920`; the player-history model achieved `0.360772`.
Team minus player NLL was `-0.002852`, a relative deterioration of `0.7967%`.
The 2,000-replicate match-bootstrap interval was
`[-0.003541, -0.002136]`. Every frozen gate check failed, including improvement
over team history, the bootstrap lower bound, shuffled controls, and supported
subset consistency.

Decision: player history was worse than team history. No predictive
player-memory claim is supported.

Result payload SHA-256:
`86dbbfa088e5083aa2247c89d4d22d649255402f99dda3a65378c6d87f0d6ed3`.

Reproducible sources:

- `configs/wyscout_player_memory_v1.yaml`
- `splits/wyscout_player_memory_development_v1.json`
- `scripts/build_wyscout_player_memory_dataset.py`
- `scripts/build_wyscout_player_memory_split_manifests.py`
- `scripts/run_wyscout_player_memory_v1.py`
- `src/footballq/analysis/wyscout_player_memory.py`
- `src/footballq/data/wyscout_public.py`

## 3. Wyscout Cross-Team Player Fingerprint

The frozen protocol is documented in
`docs/WYSCOUT_CROSS_TEAM_FINGERPRINT_CONFIRMATORY_PROTOCOL_V1.md`.

The primary `behavior_72` vector uses outcome-free pass behavior. It compares
players only with candidates in the same broad role and includes controls that
also hold support-team or query-team context approximately constant.

Development, club 2017/18 to World Cup 2018:

- eligible query players: `269`;
- eligible support candidates: `2,062`;
- same-role pairwise AUC: `0.8374`, 95% CI `[0.8123, 0.8606]`;
- top-1: `4.46%` versus `0.186%` analytic chance;
- MRR: `0.1136` versus `0.0125` chance;
- 20-match minus one-match MRR: `0.0409`, 95% CI
  `[0.0154, 0.0666]`;
- frozen gate: passed.

Confirmation, Euro 2016 to club 2017/18:

- eligible query players: `194`;
- eligible support candidates: `364`;
- same-role pairwise AUC: `0.8198`, 95% CI `[0.7883, 0.8485]`;
- top-1: `11.86%` versus `1.108%` analytic chance;
- MRR: `0.2305` versus `0.0540` chance;
- five-match minus one-match MRR: `0.0581`, 95% CI
  `[0.0149, 0.1010]`;
- all six frozen gate checks passed.

Decision: outcome-free on-ball pass behavior contains reproducible
player-specific information that persists across club and national-team
contexts. This does not show that the representation predicts outcomes,
off-ball behavior, tactics, or full-match plans.

Development result payload SHA-256:
`b1f07058d980936066fd927fe187a85f891d49d6ff7eaa01c6c8f37051b7ae8b`.

Confirmatory result payload SHA-256:
`fd5d2a3caa92b7455c187e006963288b0889c2966dcabfb69ea294b5a468698a`.

Reproducible sources:

- `configs/wyscout_player_fingerprint_v1.yaml`
- `splits/wyscout_player_fingerprint_club_to_world_cup_v1.json`
- `splits/wyscout_player_fingerprint_euro_to_club_confirmatory_v1.json`
- `scripts/build_wyscout_player_fingerprint_manifests.py`
- `scripts/freeze_wyscout_player_fingerprint_confirmatory_v1.py`
- `scripts/run_wyscout_player_fingerprint_v1.py`
- `src/footballq/analysis/wyscout_player_fingerprint.py`

## 4. Wyscout Cross-Team Action Prediction

Protocol: `wyscout_cross_team_pass_destination_v1`.

This development experiment tested whether persistent player history improves
later pass-destination and penalty-area-entry prediction after a team-context
change. At the primary 20-match support cap, `56,847` of `65,236` validation
passes had support.

For penalty-area entry on supported validation rows:

- rolling-player NLL: `0.296693`;
- conditional-player NLL: `0.296561`;
- reported relative NLL improvement: `0.0699%`;
- bootstrap NLL-gain lower bound: `-0.000156`.

The NLL threshold and positive-bootstrap checks failed. The action-destination
head also worsened on the development set: rolling NLL `2.469539` versus
conditional NLL `2.473478`, with bootstrap gain interval
`[-0.006106, -0.001741]`.

Decision: the strong Wyscout identity fingerprint did not translate into a
robust action-prediction improvement under this model. The frozen development
gate failed and no confirmatory predictive claim is allowed.

Result payload SHA-256:
`c78ee9990f29bd04068bb708f7d0bdd8c39032fb9631fa775f7425b74e08b3d0`.

Reproducible sources:

- `configs/wyscout_cross_team_action_v1.yaml`
- `scripts/run_wyscout_cross_team_action_v1.py`
- `src/footballq/analysis/wyscout_cross_team_action.py`

## 5. StatsBomb Critical-Event History Signal

This development-only experiment tested prior player profiles for progressive
pass and five-second turnover prediction. Against rolling involvement, the
profile changed validation average precision by `-0.00404` for progressive
passes and `+0.00062` for turnover. Validation Brier score worsened by
`0.00033` and `0.00110`, respectively. Development-test average precision
worsened by `0.01093` and `0.02756`.

Decision: there was no stable incremental critical-event signal beyond recent
involvement. These results do not justify a player-conditioned outcome claim.

Result payload SHA-256:
`acc16343c2f5da9e3aef034f6bb8f079c1df42598e4ea41024812977ece449fa`.

Reproducible sources:

- `configs/statsbomb_player_history_signal_v1.yaml`
- `scripts/run_statsbomb_player_history_signal_v1.py`
- `src/footballq/analysis/statsbomb_player_history_signal.py`

## 6. StatsBomb Recipient-History Development

At the selected five-match support size, the Leverkusen development test
contained `9,634` eligible passes across `34` matches. The recipient profile
improved NLL over rolling involvement from `2.286814` to `2.274910`, an
absolute gain of `0.011904` or `0.5206%`. Its match-bootstrap 95% interval was
`[0.008194, 0.015682]`.

Top-3 accuracy improved only `0.55` percentage points, with bootstrap interval
`[-0.12, 1.20]` percentage points. The original development gate required a
two-point gain, so it remained blocked. This motivated a pre-frozen, narrower
confirmatory question about NLL only.

Result payload SHA-256:
`3677131bbdb311071f9271939e04b78820105e6baad470979a6871b3c4a059ce`.

Reproducible sources:

- `configs/statsbomb_recipient_history_v1.yaml`
- `splits/statsbomb_recipient_history_development_v1.json`
- `scripts/build_statsbomb_recipient_split_manifest.py`
- `scripts/run_statsbomb_recipient_history_v1.py`
- `src/footballq/analysis/statsbomb_recipient_history.py`

## 7. StatsBomb Recipient-History Confirmation

The frozen protocol is documented in
`docs/STATSBOMB_RECIPIENT_HISTORY_CONFIRMATORY_PROTOCOL_V1.md`.

Validation reproduced the frozen artifact exactly. Across the pooled `82`
Euro 2024 and Women's Euro 2025 matches, with `30,677` eligible passes:

- rolling NLL: `2.241628`;
- profile NLL: `2.251002`;
- profile minus rolling NLL improvement: `-0.009373`;
- relative NLL improvement: `-0.4181%`;
- match-bootstrap interval: `[-0.013031, -0.005626]`;
- rolling top-3: `43.49%`;
- profile top-3: `42.28%`.

The true profile beat the broad-role shuffled history, but it did not beat the
rolling-involvement baseline. All support sizes from 1 through 20 had negative
NLL and top-3 effects versus rolling. The frozen confirmatory gate failed.

Decision: the small development NLL benefit did not replicate in tournaments.
The evidence supports broad role and recent involvement, but not incremental
origin-zone player history for recipient ranking in this protocol.

Result payload SHA-256:
`f53bfceb723e483f7f644c723b8d20cff12b21e17f2c460cdd30b5f20ec2cbfa`.

Reproducible sources:

- `configs/statsbomb_recipient_history_confirmatory_v1.yaml`
- `splits/statsbomb_recipient_history_confirmatory_v1.json`
- `scripts/freeze_statsbomb_recipient_history_confirmatory_v1.py`
- `scripts/run_statsbomb_recipient_history_confirmatory_v1.py`
- `src/footballq/analysis/statsbomb_recipient_history.py`

## 8. StatsBomb Transfer Diagnostics

Two development diagnostics tested alternative team and tournament transfer
cohorts. In both, validation selected a profile weight of `0.0`; consequently,
the reported profile-versus-rolling gains were exactly zero and every
development gate check failed. These are model-selection nulls, not independent
confirmatory tests of a nonzero profile.

Result payload SHA-256 values:

- team transfer:
  `f6b4f2ca8f8e6bc38ba8b3f1bba5914964755d63e7e3ae8988121a5de00b27f4`;
- tournament transfer:
  `d4dcc5a915ea27048e3f7381a2be342a29bdccafbaa328a91686e8781a5adce5`.

Reproducible sources:

- `configs/statsbomb_recipient_history_transfer_v1.yaml`
- `configs/statsbomb_recipient_tournament_transfer_v1.yaml`
- `splits/statsbomb_recipient_tournament_development_v1.json`

## 9. FOOTPASS Player-History Studies

The complete FOOTPASS handoffs are:

- `docs/FOOTPASS_PLAYER_HISTORY_PROTOCOL_V1.md`
- `docs/FOOTPASS_PLAYER_HISTORY_DEVELOPMENT_RESULTS_V1.md`
- `docs/FOOTPASS_COMPACT_PLAYER_RESIDUAL_PROTOCOL_V2.md`
- `docs/FOOTPASS_COMPACT_PLAYER_RESIDUAL_RESULTS_V2.md`

V1 used `11,908` opportunities from verified Bayern, Napoli, and Lazio
identities. The large history view improved primary NLL by `6.57%` over a weak
rolling baseline, but its bootstrap interval crossed zero, a role mean and a
shuffled history performed better, turnover NLL worsened `4.01%`, and geometry
plus role was substantially stronger. The gate failed.

V2 tested a compact 28-value residual against geometry plus role. The true
player residual worsened primary NLL `3.13%`, worsened two of three validation
matches, lost to a shuffled control, and was negative in all three internal
development folds. The gate failed. Confirmation matches 22, 40, and 43 remain
outcome-sealed.

## 10. RLCS Identity-Matchup V1

This branch-level mechanism test used native RLCS 2025 Rocket League replay
telemetry, not reconstructed football broadcast video. It asked whether player
and opponent identities improve prediction of the next toucher and next-touch
zone after complete geometry, clock, and score are already known. It does not
test transfer to football and is not evidence of football tactical
understanding.

The corpus gates passed with `1,595` hashed replays, `1,445` strict-QC and
identity-resolved replays, `117,704` clean touch decisions, and zero unresolved
identity replays. The train-only 5,000-sample overfit gate reached joint NLL
`0.066389`, and all 12 matched condition/seed runs completed locally without
loading the test split.

Validation-selected factorized joint NLL was:

| Seed | Anonymous | Actor-only | Roster-only | Full | Full vs anonymous |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | `2.712673` | `2.725091` | `2.733312` | `2.730788` | `-0.668%` |
| 23 | `2.706329` | `2.721318` | `2.726149` | `2.726284` | `-0.737%` |
| 41 | `2.727498` | `2.736550` | `2.742594` | `2.739154` | `-0.427%` |

The preregistered validation unlock required at least `+2%` full-versus-anonymous
lift in two of three seeds. Zero seeds passed. No test unlock was created, and
the critical all-known test outcomes, identity shuffles, bootstraps, and sign
flips remain unobserved. V1 therefore provides a controlled validation null for
this identity-embedding architecture and next-touch objective; it does not
prove that identity-conditioned matchup signal is absent under every model or
target.

Reproducible sources:

- `docs/RLCS_IDENTITY_MATCHUP_V1.md`
- `configs/rlcs_identity_matchup_v1.yaml`
- `splits/rlcs_2025_chronological_v1.json`
- `provenance/rlcs_identity_aliases_v1.csv`
- `provenance/rlcs_identity_matchup_v1_validation.json`

## Overall Conclusion

The experiments separate two claims that must not be conflated:

1. Wyscout pass histories contain a real, reproducible cross-team player
   fingerprint.
2. None of the tested PFF, Wyscout, StatsBomb, FOOTPASS, or RLCS mechanisms has
   yet shown robust incremental prediction of tactical or critical outcomes
   beyond strong current-context, role, geometry, team, or rolling-involvement
   controls. The RLCS result is a validation-only null in another sport and
   must not be presented as a football result.

The project therefore has evidence that stable player-specific signal exists,
but not yet that the current model uses it to improve tactical prediction. The
next justified experiment must condition an appropriate predictive model on
the confirmed fingerprint while preserving chronology, team-change controls,
and a genuinely sealed outcome cohort.

The data limitation that currently prevents that experiment from being
decisive, together with the minimum acquisition target, is documented in
`docs/CURRENT_DATA_WALL.md`.
