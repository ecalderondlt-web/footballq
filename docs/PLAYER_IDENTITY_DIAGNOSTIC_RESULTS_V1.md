# Frozen Player-Identity Diagnostic V1

## Decision

**The current frozen TD-JEPA does not contain a robust cross-match player
representation.**

With one earlier match, TD-JEPA showed weak repeatability above chance but did
not improve meaningfully over raw movement. With two earlier matches, its
retrieval performance fell to chance and became significantly worse than the
raw baseline. The prespecified gate failed.

This supports acquiring longitudinal league data, but it also shows that
simply averaging more outputs from the current scene encoder is not enough.
Player-specific training objectives are needed alongside more data.

## Question

Can a frozen player token identify the same player in a later match when the
candidate set contains only players from the same national team and broad
role?

This is intentionally narrower than event prediction. It directly tests
whether the representation preserves stable player-specific information.

## Protocol

- PFF tracking and StatsBomb player identities from the first 48 World Cup
  matches only.
- Sixteen first-round matches provide initial support.
- Sixteen second-round matches are training-diagnostic queries.
- Sixteen third-round matches are validation queries.
- No knockout profiles or outcomes were loaded.
- Every support match is strictly earlier than its query match.
- Each query competes only against players who:
  - are present in the same current match;
  - play for the same team;
  - have the same broad role;
  - have at least one earlier-match profile.
- Mean validation candidate set: 4.48 players.
- Frozen input: one second of position-only tracking at 10 fps.
- Match profile: mean of the player's contextual entity tokens sampled
  approximately every 15 seconds.
- Train-fitted centering and scaling are applied before cosine retrieval.
- Support sizes: one and two prior matches.

Controls:

1. Raw one-second kinematic summaries.
2. Random match-player vectors.
3. Identity shuffled within each match's team-role group.

The cache contains 10,464 sampled clips and 1,477 match-player profiles.

## Main Result

The main condition uses up to two strictly earlier matches.

### Validation retrieval

| Representation | Top-1 | Chance top-1 | Pairwise accuracy | Mean margin |
|---|---:|---:|---:|---:|
| Raw kinematics | **0.2679** | 0.2423 | **0.5299** | **0.03149** |
| Frozen TD-JEPA | 0.2423 | 0.2423 | 0.4928 | 0.00455 |
| Random profile | 0.2423 | 0.2423 | 0.5159 | 0.00233 |
| Team-role identity shuffle | 0.2347 | 0.2423 | 0.4829 | -0.00758 |

For frozen TD-JEPA versus raw kinematics:

- Top-1 difference: `-0.0255`.
- Pairwise-accuracy difference: `-0.0371`.
- Player-bootstrap 95% interval: `[-0.0722, -0.0019]`.

The interval is below zero. Under the two-match profile rule, the frozen
representation is measurably worse than the simple raw baseline.

## One-Match Sensitivity

With only the most recent earlier match:

| Representation | Top-1 | Chance top-1 | Pairwise accuracy |
|---|---:|---:|---:|
| Raw kinematics | 0.3265 | 0.2423 | 0.5628 |
| Frozen TD-JEPA | **0.3342** | 0.2423 | **0.5645** |
| Random profile | 0.2628 | 0.2423 | 0.5061 |
| Team-role identity shuffle | 0.2117 | 0.2423 | 0.4780 |

TD-JEPA is above chance here, but its pairwise gain over raw movement is only
`+0.0017`. The player-bootstrap mean gain is `+0.0019`, with a 95% interval
of `[-0.0391, +0.0421]`. In other words, the one-match signal is real enough
to call weak repeatability, but not enough to call an incremental learned
player representation.

## Interpretation

In simple terms, a single earlier match sometimes contains enough stable
positioning information to help recognize a player. The frozen TD-JEPA keeps
some of that information, but no more reliably than a small raw movement
summary.

When two match profiles are averaged, TD-JEPA gets worse. This suggests its
entity tokens are dominated by match-specific scene context, role, formation,
or opponent conditions rather than a stable player signature. Averaging those
contextual states does not isolate player style.

The result does not show that player-specific modeling is impossible. It shows
that the current scene-prediction objective and mean-profile construction are
not sufficient.

## Research Consequence

The next model should not be the current encoder trained longer without other
changes. The justified direction is:

1. Acquire league or multi-season tracking with stable player IDs and at least
   10 earlier appearances for many players.
2. Train same-player positives across different matches.
3. Use same-team, same-role players as hard negatives.
4. Add whole-player masking and identity-consistency losses.
5. Replace simple profile means with a learned, uncertainty-aware pooler.
6. Repeat this retrieval diagnostic before reopening event-prediction tests.

The current 48 group-stage matches are sufficient for an engineering smoke
test of that identity-aware objective, but not for a strong final claim.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/run_player_identity_diagnostic_v1.py --device auto --rebuild-cache
```

Artifacts:

- Config: `configs/player_identity_diagnostic_v1.yaml`
- Audit: `runs/player_identity_diagnostic_v1/profile_audit.json`
- Results: `runs/player_identity_diagnostic_v1/results.json`
- Profile cache: `runs/player_identity_diagnostic_v1/profile_cache.pt`

Provenance hashes:

- Frozen checkpoint SHA-256:
  `f7aaefae797f39f5ffad57f66427712e384bf7bb4027a1295b05aa2ec1455a33`
- Experiment code SHA-256:
  `ace38f77bfad4226996f48a4f458cbd91e7a82ad1917a44704cdeb80e0a95eee`
- Config file SHA-256:
  `ac0ea1e2efa4291b66aa1a69a3527225bae09f6dd54b3a5d1ac2dc96545d3cda`
- Profile cache SHA-256:
  `b62281aa878caaadc403ac4d5dcbda719e5b38a534244077afea6db25da7e830`
- Results SHA-256:
  `b1ad4c12245b854a1eaae4b1ac56dc2ab2f8d7799fa9857f1f89fae1801d50c7`
