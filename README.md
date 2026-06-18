# footballq

`footballq` is a Phase 1 baseline foundation for learning football game-state dynamics from
structured tracking data. The long-term target is a temporal-difference latent world model for
tactical soccer physics, but this phase deliberately stays with deterministic trajectory
prediction baselines.

Phase 1 builds reproducible ingestion, normalization, 23-entity windowing, deterministic
baselines, metrics, and training/evaluation scripts around canonical football tracking tables.
It does not implement TD-JEPA, latent flow matching, text alignment, video processing, tactical
discovery, or a UI.

## Install

```powershell
cd footballq
python -m pip install -e ".[dev]"
```

Python 3.11 or newer is required.

## Phase 1 Synthetic Benchmark

This end-to-end path does not require real tracking data.

```powershell
python scripts/make_synthetic_data.py `
  --out data/synthetic/tracking.parquet `
  --num-matches 2 `
  --num-frames 1000 `
  --fps 10

python scripts/prepare_tracking_data.py `
  --source synthetic `
  --raw data/synthetic/tracking.parquet `
  --out data/processed/synthetic_windows.pt `
  --fps-out 10 `
  --context-seconds 2.0 `
  --horizon-seconds 2.0 `
  --stride-seconds 0.2

python scripts/train_baseline.py --config configs/baseline_st_transformer.yaml
python scripts/eval_baseline.py --checkpoint runs/st_transformer/latest.pt --split test
```

Other included configs:

- `configs/baseline_constant_velocity.yaml`
- `configs/baseline_mlp.yaml`
- `configs/baseline_st_transformer.yaml`

Each training run writes:

- `runs/<model>/<timestamp>/config.yaml`
- `runs/<model>/<timestamp>/metrics_train.jsonl`
- `runs/<model>/<timestamp>/metrics_val.jsonl`
- `runs/<model>/<timestamp>/best.pt`
- `runs/<model>/<timestamp>/latest.pt`
- `runs/<model>/<timestamp>/eval_test.json`
- `runs/<model>/<timestamp>/predictions_sample.pt`

Convenience copies are also written to `runs/<model>/latest.pt` and `runs/<model>/best.pt`.

Metrics are reported in meters, including player/ball/all-entity ADE/FDE and team-shape errors.
Models train on centered normalized coordinates, then convert predictions back to meters for
reporting.

## Real Data Adapters

Prepare local SkillCorner Open Data by pointing `--raw` at the directory containing match JSONL
tracking files:

```powershell
python scripts/prepare_tracking_data.py `
  --source skillcorner `
  --raw data/raw/skillcorner `
  --out data/processed/skillcorner_windows.pt `
  --match-id skillcorner_match_1
```

Prepare IDSSE/Sportec-style data by pointing `--raw` at a long-form CSV/parquet file or directory.
The current adapter expects frame/time/entity/x/y columns and optional team/type metadata:

```powershell
python scripts/prepare_tracking_data.py `
  --source idsse `
  --raw data/raw/idsse `
  --out data/processed/idsse_windows.pt `
  --match-id idsse_match_1
```

Metrica remains available as a debug fallback:

```powershell
python scripts/prepare_tracking_data.py `
  --source metrica `
  --raw data/raw/metrica `
  --out data/processed/metrica_windows.pt `
  --match-id sample_game_1
```

Assumptions for public data are intentionally conservative: if possession, phase, event labels, or
stable home/away metadata are missing, the pipeline leaves those features empty instead of
inventing them. Missing or invisible entities keep the fixed `[23]` entity shape and are handled by
masks.

## Synthetic Demo

The demo does not require internet or real football data.

```powershell
footballq synthetic-demo
```

It writes:

- `artifacts/synthetic_demo/tracking.parquet`
- `artifacts/synthetic_demo/features.parquet`
- `artifacts/synthetic_demo/windows/windows.npz`
- `artifacts/synthetic_demo/windows/window_meta.parquet`
- `artifacts/synthetic_demo/synthetic_clip.gif` or `.mp4`

The same workflow can be run without installing the console script:

```powershell
python scripts/run_synthetic_demo.py
```

## Metrica Sample Data

Download or clone the Metrica Sports sample data from:

https://github.com/metrica-sports/sample-data

Place the files under a local directory such as:

```text
data/raw/metrica/
```

Then run:

```powershell
footballq ingest-metrica --raw-dir data/raw/metrica --out-dir data/processed/metrica --match-id sample_game_1
footballq features --tracking data/processed/metrica/tracking.parquet --out data/processed/metrica/features.parquet
footballq render --tracking data/processed/metrica/tracking.parquet --out artifacts/metrica_clip.gif --start-time-s 60 --duration-s 10 --fps 10
footballq export-windows --tracking data/processed/metrica/tracking.parquet --features data/processed/metrica/features.parquet --out data/processed/metrica/windows
```

Raw datasets and generated media are ignored by Git by default.

## Outputs

All tracking positions are normalized to meters on a 105m x 68m pitch with origin at the top-left
touchline/goal-line corner. The main tables are documented in `docs/SCHEMA.md`.

The legacy NPZ window export is still available through `footballq export-windows`. The Phase 1
baseline scripts use Torch `.pt` windows shaped like:

- `past`: `[num_windows, L, 23, F]`
- `future_xy`: `[num_windows, H, 23, 2]`
- `past_mask`: `[num_windows, L, 23]`
- `future_mask`: `[num_windows, H, 23]`

