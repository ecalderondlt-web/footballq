# Verdict

I reviewed the detailed RLCS report, the simple summary, the frozen configuration, the machine-readable validation ledger, the model, the validation evaluator, and the checkpoint-selection code.

Your diagnosis is **directionally correct, but incomplete**.

V1 was not literally forecasting the next two seconds of coordinates. It used **two seconds of past telemetry** to predict the **next distinct toucher and next-touch location occurring 0.2–4.0 seconds later**. That was a valid experiment for the narrow hypothesis “does identity help predict the immediate transition after complete geometry is known?” It produced a strong negative result: full identity was worse than anonymous in all three seeds, and actor-only and roster-only were also worse. That result should remain closed and should not be reinterpreted as a near-miss.

But V1 did **not properly test the current coach-model claim**. The trainer selected checkpoints exclusively by next-touch entity-plus-zone NLL over the complete validation set. The “critical all-identities-known” subset existed only in the unopened test protocol. Although the model had `retained_possession` and `goal_within_8s` heads, the goal head had weight `0.10` while each next-touch head had weight `1.0`; validation reported only binary goal accuracy, not goal log loss, Brier score, average precision, or calibration. None of those outcome heads affected the gate or checkpoint selection.

There is also an architecture mismatch. V1 supplied raw identity embeddings learned from the next-touch objective. Its dedicated matchup token compressed the opposition to the **mean opponent embedding** and the interaction `actor × mean opponents`. That is not an explicit representation of “this attacker’s strength against the weakest or most relevant defender.” The Transformer could theoretically recover individual interactions from the six car tokens, but the objective gave it little reason to learn persistent skill traits.

Therefore:

> **V1 rejects identity-conditioned next-touch prediction. It does not reject identity-conditioned critical-action value, but target mismatch alone cannot be assumed to explain the failure. The identity representation must also change.**

# The next scientific step

Run a fresh, preregistered experiment called:

```text
RLCS Player-Matchup Critical Value V2
```

The experiment should adapt the logic of Expected Possession Value and VAEP to RLCS telemetry.

VAEP does not try to guess the exact next movement. It estimates the probabilities that the acting team will score or concede in the next several actions, defines state value as `P(score) − P(concede)`, and values an action by how much it changes that quantity. The established implementation uses the next ten actions as the short-term outcome window. EPV research similarly uses high-resolution multi-agent tracking to estimate possession outcomes and reveal player decision tendencies rather than requiring exact trajectory prediction.

## Falsifiable V2 hypothesis

> **After controlling for complete telemetry, score state, clock, and chronology-safe team strength, persistent player profiles and explicit actor-versus-opponent profile interactions improve prediction of whether the acting team scores or concedes within the next ten distinct touches.**

The matchup part is essential. A model that improves only because it knows a player or roster is generally strong does not prove the intended claim.

# 1. Outcome and action value

Use the same de-duplicated touch sequence already built for V1, but replace the primary target.

For every eligible touch, assign one mutually exclusive class:

```text
0 = no goal before the next 10 distinct touches
1 = actor's team scores first before the next 10 distinct touches
2 = actor's team concedes first before the next 10 distinct touches
```

Stop the horizon at the first goal or kickoff boundary. Do not allow the ten-action window to cross a reset.

The model outputs:

```text
P_score(s)
P_concede(s)
P_no_goal(s)
```

Define state value:

```text
V(s) = P_score(s) - P_concede(s)
```

For consecutive states surrounding an observed action:

```text
action_value(a_t) = V(s_after) - V(s_before)
```

A positive value means the action increased the acting team’s short-term scoring advantage; a negative value means it increased danger or reduced attacking value. This is the correct intermediate quantity between raw motion and eventual match outcome.

The primary metric is three-class log loss. Secondary metrics are multiclass Brier score, calibration error, and one-vs-rest average precision for scoring and conceding. Do not use raw accuracy as the main metric.

# 2. Do not use raw identity embeddings as the primary identity mechanism

Build a chronology-safe, interpretable profile for every player using **strictly earlier games only**.

In RLCS, “speed” should mean persistent behavior rather than a categorical identity token. Useful prior-game profile families are:

| Profile family | Examples |
|---|---|
| Pace and movement | Supersonic-time fraction, ball-carry speed, recovery time, goalside recovery speed |
| Boost economy | Boost use per minute, boost use per metre of ball progression, low-boost possession rate |
| Attacking behavior | Shot rate, shot conversion, attacking-half touch rate, ball progress per touch |
| Control style | Ground-dribble, aerial-touch, flick, pass, rebound, and carry frequencies |
| Defensive behavior | Goalside fraction, save rate, clear/retrieval rate, challenge success, turnover pressure |
| Risk | Turnover rate, failed challenge rate, conceded-shot rate while nearest defender |
| Team interaction | Teammate spacing, pass-to-shot conversion, double-commit rate |

Prior Rocket League research found that shots, shots conceded, saves, time spent goalside of the ball, and time at supersonic speed were associated with match performance or player rank. Those findings justify these as candidate profile features, but not as causal attributes.

Use empirical-Bayes shrinkage fitted only on the profile-support data:

- rate statistics shrink toward the population rate;
- continuous statistics shrink toward the population mean;
- the stored profile must include its effective sample size and uncertainty;
- no match may contribute to its own player profiles.

Do not use the display name, raw player ID, team ID, roster hash, or learned ID embedding in the primary V2 model.

# 3. Explicitly model the relevant opponent, not the average opponent

Keep the existing `20 × 7 × 27` telemetry encoder, but replace the V1 identity layer.

For the actor and each opponent separately, create:

```text
pair_j = MLP(
    actor_profile,
    opponent_j_profile,
    actor_profile - opponent_j_profile,
    current actor-to-opponent relative geometry,
    opponent distance to ball,
    opponent distance to own goal,
    estimated actor/opponent intercept-time difference
)
```

Compute three actor-opponent pair tokens. Aggregate them with both:

```text
attention-weighted sum
elementwise maximum
```

The maximum path is important because it can represent a weakest-link matchup. The model should be able to express:

```text
high actor carry pace
minus poor recovery profile of opponent 2
combined with opponent 2 being the last relevant defender
```

Use a separate teammate-synergy projection, but do not average teammates and opponents into one generic roster vector.

A suitable V2 model remains small:

```text
geometry Transformer width: 192
profile dimension: 24-32
profile projection: 64
pairwise MLP output: 64
three opponent pair tokens
attention + max aggregation
three-class outcome head
approximately 1.5-2.0 million parameters
```

# 4. Required baselines

Train matched models with identical geometry backbones, batch order, optimizer, seeds, and capacity.

| Condition | Additional information |
|---|---|
| **State baseline** | Telemetry, score, time, overtime |
| **State + team form** | State plus chronology-safe team/roster strength controls |
| **Actor profile** | Team-form baseline plus actor’s prior profile |
| **Additive roster profiles** | Actor and opponent profiles, but no pairwise interactions |
| **Full matchup** | Actor-versus-each-opponent pairwise model |

The principal comparison is:

```text
Full matchup
versus
State + team form
```

The comparison against additive roster profiles determines whether there is actual **matchup interaction**, rather than merely useful player ability.

Required falsification controls:

```text
shuffle actor profiles among players with similar prior team strength
shuffle opponent profiles among matched series
permute the three opponent profiles while retaining their geometry
replace player profiles with population means
```

If team form absorbs the gain, identity was only proxying roster strength. If additive profiles equal the pairwise model, the result is individual skill, not opponent-specific matchup knowledge.

# 5. Chronological split

Keep the current sealed test completely closed. V1 never loaded it, so it remains valuable.

Within Split 1, assign complete official series chronologically and separately by EU/NA:

```text
earliest 35% of Split 1 series
    -> profile support only

next 45%
    -> V2 model training

final 20%
    -> internal development
```

Then use:

```text
Split 2 Regional 1
    -> frozen V2 validation

Split 2 Regionals 2 and 3
    -> sealed V2 test
```

Profiles for every query are recomputed from matches strictly earlier than that query. A player without enough history receives a shrunk population profile plus an uncertainty indicator, not an ID embedding.

The current V1 validation result has exposed only next-touch performance. Nevertheless, V2 architecture, target horizon, features, and gates should be frozen using Split 1 only before calculating the Split 2 Regional 1 critical-value result.

# 6. First gate: prove the player profiles are persistent

Before training the outcome model, split each eligible player’s support games into chronological early and late halves.

Compare:

```text
same player's early vs late profile
against
different players matched on region, date, and prior team strength
```

Require:

```text
at least 60 eligible players
at least 15 prior games per eligible player
same-player pairwise retrieval AUC >= 0.75
median split-half Spearman correlation >= 0.35
for the core continuous traits
```

This does not prove tactical prediction. It proves that the proposed “speed,” “recovery,” “control,” and defensive traits are persistent enough to condition a model.

If this gate fails, stop. A matchup model cannot work scientifically if its player profiles are mostly noise.

# 7. Validation gates

Train seeds:

```text
17, 23, 41
```

The frozen V2 validation gate should require all of the following:

| Gate | Requirement |
|---|---:|
| Full matchup vs state + team form | At least **2.0%** relative log-loss reduction in at least two of three seeds |
| Series uncertainty | 95% series-bootstrap lower bound above zero |
| Matchup interaction | Full beats additive roster profiles by at least **0.5%** |
| Actor contribution | Full beats actor-profile-only by at least **0.5%** |
| Profile shuffle | Full beats every main shuffled-profile control by at least **1.0%** |
| Calibration | ECE no more than `0.01` worse than the state baseline |
| Regional robustness | Positive point estimate in both EU and NA |

Official series remain the bootstrap and sign-flip unit.

Only after those gates pass should V2 create a new one-time test unlock.

# 8. Aggregating critical moments into match outcome

Do not begin by training one direct winner classifier from 1,445 games. That would sharply reduce the effective sample size, heavily confound player identity with team strength, and make action attribution difficult.

Instead, aggregate the V2 action values.

For each team at elapsed times of 60 and 120 seconds:

```text
cumulative_team_value =
    sum of predicted action_value
    over all non-goal actions observed so far
```

Train a small game-level logistic regression:

```text
Baseline:
current score
time remaining
overtime
chronology-safe team form

Full:
baseline
+ cumulative value for both teams
+ largest positive action value
+ largest negative action value
+ number of high-value actions
```

The secondary overall-outcome gate is:

```text
at 120 seconds:
at least 2% final-winner log-loss reduction
over score/time/team-form baseline
with 95% series-bootstrap lower bound above zero
```

Exclude the scoring touch itself from the aggregation so the model cannot “predict” victory by counting goals that have already happened.

A primary action-value pass plus a game-level failure supports only:

```text
identity-conditioned critical-value prediction
```

It does not support:

```text
improved overall winner prediction
```

# 9. Immediate diagnostic before V2 training

Add a validation-only script:

```text
scripts/audit_rlcs_v1_outcome_heads.py
```

Using the existing 12 V1 checkpoints and only the already-open validation split, report:

```text
goal_within_8s prevalence
goal log loss
goal Brier score
goal average precision
goal calibration curve
retained-possession log loss and Brier score
anonymous / actor-only / roster-only / full differences by seed
results on the validation critical-state subset
```

This is a **post hoc diagnostic**, not a V1 rescue. It must not create a test unlock or alter the V1 conclusion.

Its purposes are limited to:

- verifying that the current goal label is constructed sensibly;
- measuring class imbalance;
- checking whether the V1 outcome heads collapsed to majority predictions;
- confirming that the new V2 target is necessary.

Because the V1 checkpoints were selected by next-touch NLL and the goal head was underweighted, even a positive diagnostic would not constitute evidence for the hypothesis.

# 10. Repository implementation path

Add:

```text
docs/RLCS_PLAYER_MATCHUP_VALUE_V2_PROTOCOL.md
configs/rlcs_player_matchup_value_v2.yaml

scripts/audit_rlcs_v1_outcome_heads.py
scripts/build_rlcs_player_profiles.py
scripts/audit_rlcs_player_profiles.py
scripts/build_rlcs_value_dataset.py
scripts/train_rlcs_value.py
scripts/eval_rlcs_value.py
scripts/summarize_rlcs_value.py

src/footballq/data/rlcs_player_profiles.py
src/footballq/data/rlcs_value_windows.py
src/footballq/models/player_matchup_value.py
src/footballq/training/train_rlcs_value.py
src/footballq/training/eval_rlcs_value.py

tests/test_rlcs_profile_chronology.py
tests/test_rlcs_profile_stability.py
tests/test_rlcs_value_labels.py
tests/test_rlcs_value_no_future_leakage.py
tests/test_rlcs_matchup_ablations.py
tests/test_rlcs_value_test_lock.py
```

Reuse the existing replay acquisition, identity registry, parser cache, run-manifest hashing, series IDs, and sealed-test infrastructure. The data engineering succeeded and does not need replacement.

# Stop conditions

Stop V2 without opening the test when any of these occurs:

- player-profile stability gate fails;
- fewer than 60 sufficiently supported players remain;
- score and concede labels do not each have at least 5,000 positive training rows;
- pairwise matchup fails to beat state-plus-team-form on internal development;
- profile shuffling retains the apparent gain;
- the result exists only in one region;
- full matchup does not beat additive profiles;
- Split 2 Regional 1 validation fails the frozen gate.

Interpret the possible outcomes strictly:

| Outcome | Allowed conclusion |
|---|---|
| Actor profile helps; matchup does not | Persistent individual skill matters, but opponent-specific interaction is unproven |
| Additive profiles help; pairwise does not | Roster composition matters, but not “this player against this opponent” |
| Pairwise model passes action-value gate | Identity-conditioned matchup improves critical-outcome prediction |
| Action-value and game aggregation pass | Combined matchup-sensitive critical moments improve early winner prediction |
| All fail | The current identity-conditioned coach-model thesis is unsupported on this substrate |

The next step is therefore **not** to lengthen the V1 next-touch horizon or reopen its test. It is to move from immediate transition prediction to a chronology-safe, profile-conditioned **critical action value** model, and only then aggregate those values toward overall match prediction.
