# Datasets

## Metrica Sports Sample Data

URL: https://github.com/metrica-sports/sample-data

Metrica is the first fully implemented adapter. The sample data contains synchronized tracking and
event CSVs. Coordinates are normalized from 0 to 1, with `(0, 0)` at the top-left and a 105m x 68m
pitch. `footballq` converts these coordinates into canonical meter coordinates.

## SkillCorner Open Data

URL: https://github.com/SkillCorner/opendata

Docs: https://skillcorner.github.io/opendata/

SkillCorner provides 10 matches of broadcast tracking data plus dynamic events. The documented
tracking files are JSONL, sampled at 10 fps, with frame-level player and ball data in meters around
a center-origin pitch coordinate system. Phase 1 includes an adapter interface and a minimal loader
for locally present JSONL files, but the synthetic and Metrica demos do not depend on SkillCorner.

## Google Research Football Synthetic Episodes

URL: https://github.com/google-research/football

Google Research Football is a simulator/RL environment, not real match tracking.
Its raw observations expose ball/player coordinates and movement vectors. `footballq`
can consume saved raw-observation JSON/JSONL files as synthetic coordinate tracking via
the `gfootball` adapter. Use this source for simulator pretraining or domain-randomized
controls only; do not treat it as held-out real-world evidence.

## PFF FC World Cup 2022 Tracking

The local PFF delivery contains continuous broadcast-derived player and ball coordinates in
JSONL or bzip2-compressed JSONL. Coordinates are in meters around the pitch center and the sampled
files run at approximately 29.97 fps. The adapter uses the provider's smoothed trajectories by
default, deduplicates repeated event-linked frames and repeated jersey entries, and preserves
whether a location was directly visible or estimated off-camera.

The complete `wc2022datav2` folder contains 64 unique game IDs but no companion metadata file.
Pitch dimensions therefore use the documented 105m x 68m example, while team/player identity,
starting direction, and metadata-authoritative video rate remain unavailable. The immutable
48/8/8 match split is stored at `splits/pff_wc2022_64match_inductive_v1.json`; full scientific
fine-tuning still requires metadata provenance and an estimated-coordinate dependence control.

The authoritative processed source is canonical version 2 under
`data/processed/pff_wc2022_canonical_v2`. It uses deterministic eleven-player roster slots to
handle substitutions. The finalized all-available geometry-only TD manifest contains 1,975,069
future-nonoverlap examples and is consumed lazily; raw data and processed tensors remain ignored by
Git.

The raw tracking records also contain provider game and possession events. The frozen context V1
mapping was fit and audited on the 48 training matches only. It retains passes, challenges, carries,
clearances, crosses, receptions, shots, and selected match-state events; generic `OTB` interval
wrappers are excluded. Train and validation event histories are stored separately under
`data/processed/pff_statsbomb_event_context_v1`, and no test event shard exists.

## SoccerTrack v2

URL: https://atomscott.github.io/SoccerTrack-v2/

SoccerTrack v2 provides full-pitch game-state reconstruction annotations with track IDs, player
IDs, roles, jersey numbers, team side, and 2D pitch coordinates in meters. Phase 1 documents schema
compatibility and includes an adapter stub for future work.

## StatsBomb Open Data

URL: https://github.com/statsbomb/open-data

StatsBomb Open Data contains events, lineups, and selected StatsBomb 360 freeze frames. It is not
continuous full-pitch tracking. In this project it is treated as event context that can later label
or enrich tracking windows.

The semantic-event V1 source is pinned to upstream commit
`b0bc9f22dd77c206ddedc1d742893b3bbe64baec`. It contains 4,235 event/lineup matches and 426
matches with a 360 file. The immutable match split contains 3,388 train, 425 validation, and 422
test matches. Only train and validation were tensorized: 11,890,025/1,503,962 events and
739,046/93,479 period-safe causal windows. The test split has no processed tensors. Exact source,
vocabulary, tensor, and access lineage is recorded in
`docs/STATSBOMB_SEMANTIC_PRETRAIN_PROTOCOL_V1.md`.

## Commit Policy

Do not commit raw datasets, processed parquet files, NPZ exports, or rendered clips. Keep large
files in `data/` or `artifacts/`, which are ignored by default.

