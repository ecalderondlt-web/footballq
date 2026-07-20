# PFF FC World Cup 2022 Integration

## Current Local Inventory

The initially supplied directory was:

```text
wc2022data/Tracking Data
```

That directory contains 58 unique game IDs. A second local delivery then became available at:

```text
wc2022datav2
```

The second delivery contains all 64 unique game IDs: 3812 through 3859 and 10502 through 10517.
The adapter resolves one physical source per game and does not count archive/extracted pairs as
separate matches. No metadata, roster, or standalone game table was found in either delivery.

## Verified Format

Each tracking file is newline-delimited JSON, optionally compressed with bzip2. Each record carries:

- global video frame and timestamp
- period and period-relative elapsed time
- raw and smoothed home-player coordinates
- raw and smoothed away-player coordinates
- raw and smoothed ball coordinates
- optional linked game-event and possession-event records

Coordinates are continuous tracking data, not video. They come from Sportlogiq broadcast tracking
combined with PFF events. Player coordinates include `VISIBLE` and `ESTIMATED` states; off-camera
locations can therefore be inferred by the provider rather than directly observed. The ball can
also be estimated, especially in the air.

Coordinates are meters around pitch center. PFF x increases camera-left to camera-right, y
increases camera-bottom to camera-top, and ball z increases vertically. The adapter converts this
to footballq's top-left-origin 105m x 68m coordinate convention.

## Verified Provider Quirks

Sampled files run at approximately 29.9697 fps. This rate is inferred from frame timestamps, not
read from the missing metadata.

Frames associated with events can be repeated. Some repeated records also contain each jersey
twice, where the second coordinate is the following frame's position. The adapter keeps the first
record for a video frame and the first coordinate for each team/jersey, yielding one ball plus 11
home and 11 away entities.

On a 600-frame diagnostic segment from game 10502:

- 13,800 canonical rows were produced
- every frame contained exactly 23 unique entities
- no duplicate `(match, period, frame, agent)` identities remained
- 81 two-second-context/two-second-future windows were produced at 10 fps
- 55.4% of entries were directly visible and 44.6% were provider-estimated

These percentages describe one short segment only and are not a dataset-wide estimate.

## Full Canonical Conversion

Canonical version 2 is complete at:

```text
data/processed/pff_wc2022_canonical_v2
```

It contains 11,849,815 unique frames, 268,688,822 canonical rows, and 2,039 period-aware
Parquet shards across all 64 matches. Every raw source and generated Parquet shard has a SHA-256
checksum. The frozen split contains 48 training, 8 validation, and 8 test matches.

Tournament-wide quality findings are:

- 42,950 repeated event-linked records removed
- zero frame gaps and zero period-time regressions
- ball coordinates absent in 32.54% of frames
- 61.97% of supplied coordinates marked `ESTIMATED`
- 0.238% of coordinate rows outside the documented 105m x 68m bounds
- 15,643 overlap-only player rows omitted during substitution transitions
- only 0.00029% of frames have a shape other than 11 home and 11 away players, with the ball
  either present or explicitly missing

Version 2 maps changing jersey identities onto eleven deterministic roster slots per team. A
returning jersey retains its slot; a substitute inherits a slot only after another player vacates
it. This prevents period-wide accumulation of substitute jerseys from corrupting the fixed
23-entity model layout. The provider jersey identity remains available as provenance.

## TD-JEPA Dataset

The finalized all-available geometry-only dataset is:

```text
data/processed/pff_wc2022_td_jepa_v2/all_available/dataset_manifest.json
```

It contains 1,975,069 unique future-nonoverlap examples in 2,039 lazy-loadable tensor shards:

- train: 1,458,160 examples
- validation: 252,310 examples
- test: 264,599 examples

Each example uses one second of context, a one-second prediction gap, and a separate one-second
target at 10 fps. No possession-derived channels are present. Every tensor file is hashed; the
final dataset-manifest hash is
`5e8e9365afc60069423f9150537c724023ab061973ee5945a988c63e7a8ac47b`.

The training and evaluation code can consume this manifest lazily with shard-grouped sampling.
A one-batch train/validation integration diagnostic completed at `runs/td_jepa/20260712_225735`.
That run validates plumbing only and is not scientific model evidence.

The finalized observed-only control is:

```text
data/processed/pff_wc2022_td_jepa_v2/observed_only/dataset_manifest.json
```

It contains 1,135,478 unique examples in the same 2,039 period-aware shards:

- train: 844,195 examples
- validation: 141,054 examples
- test: 150,229 examples

Every tensor shard is hashed; the final manifest hash is
`ca53ef656470aea212ec9365881ffcad996a8995615532d152e1cd5df00ebac2`.

## Research Status

The adapter, diagnostic path, and immutable match split are ready. The split is frozen at
`splits/pff_wc2022_64match_inductive_v1.json` with 48 training, 8 validation, and 8 test matches.
It was generated with seed 20260712 before model training or outcome inspection.

Full PFF fine-tuning is not yet a scientific run. Before interpreting player identity or absolute
pitch geometry, obtain or reconstruct with auditable provenance:

1. companion metadata with pitch dimensions, video fps, teams, period offsets, and starting side
2. player/roster identity metadata so jersey slots are not mistaken for permanent player identity
The observed-only versus all-available control is complete. Removing estimated positions reduces
the usable window population from 1,975,069 to 1,135,478 examples.

Three-seed matched 100-update diagnostics are also complete. GRF initialization improves mean
combined validation loss by 12.2% on all-available tracking and 10.3% on observed-only tracking.
Observed-only narrow TD loss improves 23.5%; all-available narrow TD loss worsens 8.0%. Details and
run IDs are in `docs/PFF_GRF_TRANSFER_DIAGNOSTIC_V1.md`.

The next compute gate is a prespecified longer matched repeat with bounded validation selection,
followed by one frozen test application and the established falsification, incremental-probe, and
discovery controls. No tactical or semantic interpretation is authorized by the diagnostic.

That longer repeat is now complete and blocked on validation. At 2,000 updates, GRF initialization
improves mean combined validation loss by only 1.297%, below the frozen 5% requirement, and worsens
mean narrow TD loss by 10.005%. No PFF test evaluation was performed. Exact runs and the gate
decision are recorded in `docs/PFF_GRF_TRANSFER_LONGER_RESULT_V1.md`.

The first PFF model run should use the frozen split, geometry-only, future-nonoverlap objective and
must pass the
same falsification controls as the current real-data path. It should be compared against a
real-only initialization and a frozen-GRF initialization; PFF test matches must not influence GRF
selection, preprocessing choices, or checkpoint selection.
