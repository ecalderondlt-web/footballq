# GRF Position-Only Volume Scaling Protocol V1

Status: frozen before master collection or model training on 2026-07-15.

## Purpose

Test whether the small persistent GRF benefit is limited by synthetic-data volume, rather than
concluding from a roughly 30,000-example source that simulation as a whole is only a warm start.
The study separates unique simulation volume from synthetic optimizer compute.

## Frozen Collection

The master collection uses the pinned FootballQ-GRF WSL2 runtime, Python 3.10, GRF 2.10.3, and
source commit `3d9e754720a95621bba6475c4d3b0d56fe919014`. The collector explicitly seeds the GRF
environment, action space, Python, and NumPy. A duplicate two-episode smoke collection produced
byte-identical SHA-256 values:
`8fdb09274e5711e52ebe9327889a17f7eb6685d2125abf6ffb4e1425fab6f48d`.

All three plans share the `gfootball_position_scale_v1` match-identity namespace and contain train
episodes only:

| Scale | Episodes | Plan payload SHA-256 | Split payload SHA-256 |
| --- | ---: | --- | --- |
| 1x | 92 | `d729369e30ccae9349257138a1c7068a40cc6da1fad5373b6b51189f9f5e0245` | `189205b98b56f421baba0873905c62216485124ea51656a7cabb631492a263da` |
| 4x | 368 | `f8f338aa7bad4321b79e87358e105515c7961ce7d3469b887ea1390776fb21f9` | `ac3fd0ef19a25d26c4b1e66a34f5b91abdaafecdfda933f883851a4808a422f6` |
| 8x | 736 | `68e9c4e1abb12f0fb593ac97548c329eace918741f3976fa84572e1bfcbc71ac` | `6c445ca9dc3820d980e40c5b33ecfd6cdaeb331ba6039ce10cc977b16e8cb76c` |

The 8x plan is collected once. The 1x and 4x raw sources are exact byte-preserving episode-prefix
subsets from every job. Scenario proportions, policies, seeds, job order, and maximum episode
lengths are identical. The curriculum contains easy/standard/hard 11v11, perturbed standard 11v11,
and six academy scenario families.

## Frozen Tensor Preparation

- splits: train only
- visibility: profile fit only on the 48 PFF training matches, payload SHA-256
  `3bd3e96d0c449e3f6a57e69a37001af71e863b82ca7613b3ba7022738280cd40`
- timing: one-second context, one-second gap, separate one-second target at 10 fps
- objective: `future_nonoverlap_context_only`
- jump boundaries: a new segment begins at a player jump of at least 3 m or ball jump of at least
  10 m
- feature view: `x_norm`, `y_norm`, `is_ball`, `is_home`, `is_away`
- boundary-crossing examples and unsafe tensor references: zero required
- example retention relative to provider-velocity preparation: at least 75% required at each scale
- nesting: every 1x sample ID and tensor value must occur identically in 4x and 8x; every 4x sample
  must occur identically in 8x

No synthetic validation or test tensor is built or read.

## Frozen Synthetic Training

All families use the same architecture, loss, optimizer, batch size 128, sampler, and seeds 7, 11,
and 23. Fixed-budget `latest.pt` checkpoints are used; synthetic validation is disabled.

| Family | Unique source | Updates | Purpose |
| --- | --- | ---: | --- |
| 1x | 1x | 263 | nested volume baseline |
| 1x replay | 1x | 2,104 | same-compute control for 8x |
| 4x | 4x | 1,052 | intermediate volume |
| 8x | 8x | 2,104 | primary scaled candidate |

The replay family intentionally revisits the 1x examples. Comparing 8x against replay distinguishes
more unique simulation from more synthetic optimizer updates.

## Frozen PFF Comparison

- PFF manifest payload SHA-256:
  `37acb8a6a00e4842a8aef8dce2700417fd7dfa24c827c3a9f46c7dac782c24ae`
- population: 844,195 train and 141,054 validation examples; zero projected test shards
- families: scratch plus the four GRF families
- paired seeds: 7, 11, and 23
- budget: 2,000 PFF updates, batch size 128, fresh optimizer
- non-selecting validation curve: updates 100, 250, 500, 1,000, and 2,000 on the first 50
  validation batches
- final validation gate: 500 validation batches at update 2,000
- embedding export: disabled

The validation curve is descriptive sample-efficiency evidence. Checkpoint selection and the final
decision use only the fixed update-2,000 evaluation.

## Primary Gate

The 8x scaling hypothesis passes only when all conditions hold:

1. all 15 PFF runs finish with finite final validation metrics
2. 8x transfer has lower total validation loss than scratch in at least two of three seeds
3. 8x lowers mean total validation loss by at least 5% relative to scratch
4. 8x mean narrow TD loss does not exceed scratch mean narrow TD loss
5. every run has final validation `z_online_std_mean` above 0.05
6. 8x mean total loss is at least 2% lower than 1x replay at the same 2,104-update synthetic budget
7. 8x mean narrow TD loss does not exceed 1x replay mean narrow TD loss

The 1x, replay, and 4x dose ordering plus validation curves are reported descriptively and cannot
replace a failed primary gate. A block stops before falsification or PFF test access. A pass permits
only a separately frozen validation falsification study.

## Claim Boundary

This experiment can identify a volume-dependent optimization benefit. It cannot establish tactical
concepts, tactical surprise, semantic understanding, downstream value, or general superiority of
simulation without raw/PCA/random downstream controls.
