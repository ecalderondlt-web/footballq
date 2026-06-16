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

## Commit Policy

Do not commit raw datasets, processed parquet files, NPZ exports, or rendered clips. Keep large
files in `data/` or `artifacts/`, which are ignored by default.

