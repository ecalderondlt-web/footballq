# FOOTPASS Tactical Data Integration

## Status

The local FOOTPASS tactical training release has been inspected as a real-data
source. This integration is data plumbing and provenance work only. It is not a
model result and does not support a tactical-understanding claim.

Local source at the time of inspection:

```text
C:\Users\emica\Downloads\tactical_data_TRAIN\train_tactical_data.h5
```

The source remains outside the repository. Do not copy the 8.8 GB extracted
file or its archive into Git.

## Source Shape

The official training delivery contains 48 matches and 96 half-match datasets.
Each labelled row has:

```text
frame, player_id, left_to_right, shirt_number, role_id,
x, y, speed_x, speed_y,
roi_x, roi_y, roi_width, roi_height, class
```

`x`, `y`, `speed_x`, and `speed_y` are pitch-state geometry. The ROI fields are
broadcast-image coordinates and may be missing when a player is off camera.
The delivery does not contain ball coordinates.

## Leakage Boundary

The lazy reader exposes geometry as exactly:

```text
x_norm, y_norm, vx_norm, vy_norm
```

The following remain separate and must not enter a geometry-only encoder:

```text
player_id, shirt_number, role_id, ROI fields, class
```

`class` is a downstream action target only. Player IDs are match-local and are
used only to maintain temporal roster slots. Any representation experiment must
retain player-slot permutation and identity controls.

## Variable Player Count

Most source frames contain 22 players. Matches 5, 9, 42, and 44 contain coherent
21-player suffixes consistent with send-off states. They are preserved with an
active-player mask and must not be dropped or imputed back to 22 players.

The internal split at
`splits/footpass_train48_development_v1.json` is stratified only on this observed
frame-shape condition. It contains 38 training matches, five validation matches,
and five internal reserve matches. Its reserve is not the hidden SoccerNet test
set. The official validation and challenge sets remain outside this manifest.

## Audit Command

```powershell
python scripts/report_footpass_availability.py `
  --h5 "C:\Users\emica\Downloads\tactical_data_TRAIN\train_tactical_data.h5" `
  --archive "C:\Users\emica\Downloads\tactical_data_TRAIN.zip" `
  --split-manifest splits/footpass_train48_development_v1.json `
  --out runs/footpass/20260719_train_availability/availability_report.json `
  --hash-source
```

The command writes both `availability_report.json` and `run_manifest.json`.

## Verified Local Audit

The full scan completed successfully on 2026-07-19. It found:

- 48 matches and 96 half-match datasets;
- 157,163,622 player rows across 7,160,226 unique frames;
- 91,327 labelled event rows;
- zero geometry rows containing NaNs and zero frame gaps;
- ROI boxes on 38.35% of all player rows and 81.44% of event rows;
- no ball-coordinate channel.

The immutable fingerprints for this inspection are:

```text
HDF5 SHA-256: bcc02cd2f05509d1e82ba16a81ada5349895410a349aeedcb13e539339379058
ZIP SHA-256:  160ecf571b5eb8298b231dcb003973702dc413469216d13d849e2c7ef6fe44ee
split SHA-256: 88d14f48741adb75dc1c1b3c0cc0eee8faa3b974bf2c5cfb0004876365d90c48
```

The machine-readable results are in
`runs/footpass/20260719_train_availability/availability_report.json`. That
directory is intentionally ignored by Git because it is a generated run
artifact.

## Next Scientific Gate

Do not start a large FOOTPASS pretraining run yet. The next gate is a bounded
adapter smoke that constructs non-overlapping geometry windows from training
matches only and verifies:

1. period-aware sample IDs are unique;
2. action classes never appear in encoder features;
3. match splits are disjoint through preprocessing;
4. 21-player masks survive batching;
5. coordinate outliers are masked and reported, not silently clipped;
6. no output is described as tactical understanding.

PFF remains the continuous 22-player-plus-ball source. FOOTPASS adds real
player geometry and player-specific action labels, but it does not replace PFF.
