# GRF-PFF Train-Only Domain-Gap Protocol V1

Status: frozen before the full audit on 2026-07-13.

## Purpose

Measure where balanced GRF V2 observable motion and geometry differ from real PFF tracking after
both sources have been converted to the same geometry-only, observed-visibility, 10 fps TD-JEPA
format. Use the result to choose one targeted simulator or objective redesign without consulting
validation or test data.

This audit does not evaluate a learned representation and cannot establish tactical concepts,
semantic understanding, or downstream usefulness.

## Frozen Data Boundary

Real source:

- PFF observed-only manifest payload SHA-256:
  `ca53ef656470aea212ec9365881ffcad996a8995615532d152e1cd5df00ebac2`
- allowed split: 48 training matches only
- forbidden: all 8 validation and 8 test matches

Synthetic source:

- balanced GRF V2 manifest payload SHA-256:
  `f8d524157196ab007c21c0662a15b18e9458b2468b5a2af349895cec4bb981db`
- allowed split: 92 training episodes only
- forbidden: all synthetic validation and test episodes

Both sources use one-second contexts, 10 fps, geometry-only features, PFF-train-derived visibility,
and the future-nonoverlap data preparation path. The audit script filters manifest entries to
`split == train` before loading tensors.

## Frozen Sampling

- deterministic seed: 20260713
- shared global context budget: 24,576 examples per source
- PFF coverage: four evenly spaced tensor shards from every training match
- PFF and GRF global allocation: proportional to selected train-shard example counts
- row selection: deterministic seeded sampling without replacement
- GRF global coverage: all ten train job shards
- scenario diagnostics: at most 5,000 contexts from each GRF train job shard

This is a distribution audit, not a training sampler. PFF match coverage is deliberately broad so
one match or one period cannot determine the result.

## Frozen Measurements

All measurements use the final context frame, with acceleration and turning computed from the last
two context frames when the entity is visible in both:

- visible player count and ball visibility rate
- player speed, stationary rate below 0.5 m/s, and high-speed rate above 7 m/s
- player acceleration and high-acceleration rate above 5 m/s^2
- player turn angle when adjacent-frame speeds are both at least 0.5 m/s
- ball speed, high-speed rate above 20 m/s, acceleration, and turn angle
- nearest visible-player distance and visible player-to-ball distance
- visible home/away x-span, y-span, and team-centroid distance

Coordinates and velocities are converted back from normalized tensor values to metres and metres
per second before measurement. Metrics never use possession, events, labels, outcomes, player
identity, or latent embeddings.

## Frozen Gap Score

For each metric, compute empirical quantiles from 1% through 99%. The primary gap score is the mean
absolute real-versus-synthetic quantile difference divided by pooled robust spread. A score near 1
means the average distribution shift is approximately one pooled interquartile scale.

The report records counts, means, standard deviations, percentiles, physical units, raw quantile
distance, normalized gap score, a global ranking, and per-GRF-scenario rankings.

## Redesign Selection Rule

Prioritize the highest-ranked global kinematic metric among player/ball speed, acceleration, and
turning only when:

1. its global gap score is at least 1.0, and
2. it appears among the five largest gaps in at least two full-match GRF scenario rankings.

If no kinematic metric satisfies both conditions, do not launch a new transfer run from this audit.
Visibility and spatial metrics remain diagnostics and cannot alone justify a motion-objective
change.

Any redesign selected by this rule must be written into a separate frozen protocol before model
training. The PFF validation and test splits remain inaccessible during redesign construction.
