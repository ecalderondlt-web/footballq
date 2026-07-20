# GRF Provider-Neutral Motion Result V2

Status: `blocked` on 2026-07-13. No model training is authorized by this result.

## Decision

Replacing GRF provider velocities with unsmoothed causal position differences does not materially
reduce the train-only player-acceleration gap and substantially worsens the physical acceleration
summary. The redesign is rejected in its present form.

The earlier V1 machine gate reported a formal pass, but that result is invalid. Its score could use
one quarter of pooled standard deviation as a denominator; rare extreme candidate accelerations
inflated that scale and made the gap appear artificially small. V2 was frozen before corrected
rescoring, removes that fallback, and adds direct mean and p99 safeguards.

## Frozen Gate Result

| Criterion | Result | Decision |
| --- | ---: | --- |
| Train tensor invariants | 33,591 examples matched | pass |
| Velocity transformation applied | 10,211,782 values changed | pass |
| Player-acceleration gap below 1.0 | 1.4487 | fail |
| Acceleration-gap reduction at least 25% | 0.1442% | fail |
| Mean acceleration no worse than 7.4006 m/s^2 | 22.9713 m/s^2 | fail |
| Acceleration p99 no worse than 30.6392 m/s^2 | 30.5757 m/s^2 | pass |
| Turn gap no worse than 110% of baseline | 1.3438 <= 1.3913 | pass |
| Speed gap no worse than 110% of baseline | 0.8229 <= 0.9123 | pass |
| Audit paths train-only | yes | pass |

Machine-readable result:
`runs/integrity/grf_provider_neutral_preflight_v2_corrected.json`.

## What The Result Means

The reconstruction is wired correctly: identities, splits, period/frame keys, coordinates, masks,
and temporal indices are unchanged, while velocity channels differ. The failure is therefore not
an accidental split or sample-membership change.

Typical acceleration remains close to the original GRF distribution, but the candidate mean rises
from `7.4006` to `22.9713 m/s^2` while its p99 is nearly unchanged and its standard deviation reaches
`226.77 m/s^2`. That pattern is consistent with a very small number of severe finite-difference
spikes, such as discontinuous simulator repositioning. This audit does not yet identify their exact
event type, so kickoff/reset attribution remains a hypothesis to test rather than a conclusion.

## Next Engineering Step

Do not run another transfer experiment yet. First freeze and run a train-only discontinuity audit
that locates extreme position jumps by episode, frame, entity, scenario, and nearby simulator game
mode. Use that evidence to choose between segmenting reset boundaries, masking invalid transitions,
or retaining provider velocity with a consistency control. Any revised transform requires its own
preflight before model training.

## Claim Boundary

This result concerns observable train-domain motion preprocessing only. It is not evidence about
validation performance, tactical concepts, tactical surprise, emergent understanding, or real-game
downstream value. No PFF validation or test tensor was read.
