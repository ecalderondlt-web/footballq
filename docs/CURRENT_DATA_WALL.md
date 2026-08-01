# Current Data Wall

## Status

The current research wall is a data-combination problem, not simply a shortage
of rows or GPU time.

The project does not currently have one real-world dataset that combines all of
the following:

1. Continuous full-match coordinates for all 22 players and the ball.
2. Stable player identities across many matches.
3. At least 10-20 strictly earlier appearances for many of those players.
4. Aligned events or tactical outcomes such as passes, turnovers, entries,
   shots, possession changes, and off-ball runs.
5. Enough teams, opponents, and dates to separate player effects from role,
   team system, opponent, and tournament context.
6. A chronological development/validation/test design with a genuinely sealed
   confirmatory cohort.

Without that combination, the project can study motion, event behavior, or
player identity separately. It cannot yet demonstrate that a model has learned
persistent individual tendencies that improve real tactical prediction.

## What The Wall Blocks

The wall blocks the broad claim that player-conditioned representations improve
critical-moment or tactical prediction beyond current geometry, role, team, and
recent-involvement controls.

It does not block:

- trajectory forecasting research;
- representation and falsification diagnostics;
- event-only semantic modeling;
- synthetic-motion pretraining;
- engineering the eventual player-conditioning interface;
- the narrow finding that Wyscout pass behavior contains a persistent player
  fingerprint.

In this repository, `blocked` means that the evidence gate for a claim did not
pass. It does not mean that the code failed to run or that no useful information
was learned.

## Why Each Current Source Is Insufficient Alone

| Source | What it provides | Why it does not resolve the wall |
| --- | --- | --- |
| SkillCorner Open Data | Ten real matches with player and ball tracking plus dynamic events | Ten matches are enough for architecture and held-out diagnostics, but not for long player histories or broad opponent/team generalization. |
| PFF World Cup 2022 | Sixty-four real matches at about 30 fps with player and ball coordinates | It is one short national-team tournament. A player can appear in at most seven matches, and the chronological support available to most queries is much smaller. Raw companion identity metadata is incomplete; the player experiments require an external StatsBomb alignment. |
| FOOTPASS tactical release | Forty-eight matches of tactical player coordinates, ROI fields, roles, and action labels | It has no ball-coordinate channel and no immediately trustworthy global player identity. The verified three-team bridge gives only 2.14 earlier matches per actor on average and never more than five. |
| Wyscout public events | Thousands of matches, stable player IDs, repeated club and national-team appearances, and rich on-ball events | It has no continuous player or ball tracking and therefore cannot teach continuous off-ball movement, pitch control, or multi-agent kinematics. |
| StatsBomb Open Data and 360 | 4,235 event/lineup matches and 426 matches with selected freeze frames | It is event data with sparse snapshots, not continuous full-pitch tracking. It is useful for event choice and semantic context, but not a continuous player-conditioned world model. |
| Google Research Football | Effectively unlimited synthetic states with player and ball kinematics | Its agents are simulated rather than persistent real players, and the synthetic-to-real domain gap remains. It cannot provide evidence about real individual tendencies by itself. |
| Broadcast-video reconstruction | Potential access to many matches and visible player identity cues | Producing research-grade coordinates, ball tracks, event labels, and cross-match identities is itself a large computer-vision data project. Off-camera coverage and identity errors must be measured before treating it as ground truth. |

The sources are complementary. Their current limitations do not imply that any
one of them is useless. The missing piece is their required information in the
same aligned, longitudinal real-match cohort.

## Workarounds Already Tested

The detailed results, hashes, controls, and reproduction paths are recorded in
`docs/PLAYER_CONDITIONED_EXPERIMENT_LOG_V1.md`. The main attempts to answer the
question without acquiring the missing dataset were:

| Attempt | Reason for trying it | Outcome |
| --- | --- | --- |
| Average frozen PFF TD-JEPA player tokens across earlier matches | Reuse the existing tracking representation as player memory | Failed cross-match identity retrieval at the main two-match condition. |
| Add frozen PFF player profiles to five-second outcome probes | Test whether contextual tracking embeddings improve turnover or penalty-entry prediction | No reliable gain over geometry, rolling event statistics, static identity, or shuffled histories. |
| Learn player history from large Wyscout event logs | Substitute repeated on-ball histories for missing longitudinal tracking | Team history beat player history in the first predictive experiment. |
| Retrieve players across club and national-team contexts with Wyscout | Test whether persistent player-specific information exists at all | Passed development and confirmation. The confirmatory same-role pairwise AUC was 0.8198. This proves a narrow on-ball fingerprint exists. |
| Use the confirmed Wyscout fingerprint for cross-team action prediction | Test whether identity signal becomes predictive value | The action-prediction gate failed; destination prediction worsened and the small penalty-entry effect was not robust. |
| Use StatsBomb histories for critical-event prediction | Exploit many identified event sequences | Progressive-pass and turnover effects were unstable or negative against rolling involvement. |
| Use StatsBomb receiving histories for recipient ranking | Test a narrower tactical choice with strict chronology | A small development NLL gain did not replicate. On the pooled 82-match confirmation, the profile was 0.4181% worse than rolling involvement. |
| Build verified FOOTPASS identity bridges | Recover repeated real players from shirt numbers and lineups | V1 and compact-residual V2 both failed. The cohort remained too shallow and controls showed no robust player-specific value. |
| Scale and redesign GRF pretraining | Use unlimited simulation to compensate for scarce real tracking | GRF produced an early head start and a persistent narrow motion benefit, but did not improve the converged complete real-tracking objective. It cannot replace real longitudinal identity data. |

These experiments cover the reasonable low-cost routes available in the current
data. They show that the wall is not likely to be solved by simply averaging
embeddings, adding more features to a linear probe, changing shrinkage, adding
same-role histories, or scaling the same GRF objective.

## Combined Scientific Conclusion

The project has a useful positive result and an equally useful limitation:

1. Persistent player-specific on-ball signal exists and can survive a complete
   club-to-country or country-to-club context change.
2. The current models and datasets have not shown that this signal improves
   tactical or critical-outcome prediction beyond strong controls.

The second result does not contradict the first. Recognizing a player's passing
fingerprint is easier than learning how that player changes a future 22-player
state under a particular opponent. The latter requires continuous, aligned,
longitudinal evidence that the current sources do not jointly provide.

Further tuning on the same PFF, FOOTPASS, Wyscout, or StatsBomb validation
cohorts would increasingly measure adaptation to those cohorts rather than
answer the original research question. More model complexity is therefore not
the clean next solution to this wall.

## Minimum Useful Acquisition

The next dataset does not need to be perfect or enormous before a pilot can
start. A useful minimum pilot should target:

- 40-60 real league matches around at least two focal teams;
- stable identities for all tracked players;
- at least 30 players with 10 or more strictly earlier tracked appearances;
- all-player and ball coordinates at 10 fps or better;
- aligned pass, shot, turnover, possession, and penalty-area-entry events;
- explicit visibility, interpolation, and substitution metadata;
- a chronological split that leaves later matches sealed.

Those numbers are design targets, not a guarantee of statistical power. They
are intended to make a genuine repeated-player experiment possible while still
being small enough for a pilot.

A stronger paper-scale cohort should target roughly 150-300 matches across
multiple teams or a full league segment, with at least 100 players having 10-20
earlier appearances. It should reserve later dates, unseen opponents, or a
second competition for confirmation.

## Data Acceptance Gate

Before training on any proposed new source, create and freeze an availability
report answering:

1. How many unique matches, teams, players, and periods are present?
2. How many players have at least 1, 3, 5, 10, and 20 earlier appearances?
3. Are player identifiers stable across matches and seasons, or only
   match-local?
4. Are all 22 players and the ball available continuously? How much is
   estimated, interpolated, or off-camera?
5. Which event labels are aligned, and at what temporal precision?
6. Can coordinates, possession direction, substitutions, and pitch dimensions
   be normalized without using future information?
7. What licenses and publication restrictions apply?
8. Can development, validation, and confirmation be split chronologically with
   no player-history leakage?

Do not begin a large training run until this report shows that the proposed
source actually supplies the missing longitudinal information.

## Recommended Next Move

The direct next move is data acquisition or construction, followed by a small
frozen pilot, not another broad model-training campaign on the current sources.

The preferred order is:

1. Locate licensed league tracking with stable cross-match player IDs and
   aligned events.
2. If it is unavailable, run a limited broadcast-reconstruction pilot and
   measure coordinate, ball, event, and identity error before scaling it.
3. Preserve the confirmed Wyscout fingerprint as an optional player-history
   input, but require it to beat role, team, rolling-involvement, and shuffled
   identity controls.
4. Keep GRF as a motion auxiliary or controlled initializer, not as evidence of
   real player knowledge.
5. Freeze the chronology, metrics, controls, and stopping rule before opening a
   later-match confirmation cohort.

Until such data is available, the defensible claim is that the project has
identified a real player-specific event fingerprint and built the evaluation
machinery for a player-conditioned model, but has not yet demonstrated
player-conditioned tactical prediction.
