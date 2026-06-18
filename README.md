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

## Tests

```powershell
python -m pytest -q
```

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
- `configs/baseline_st_transformer_skillcorner.yaml`

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

## Running Phase 1.5 On SkillCorner Open Data

Download SkillCorner Open Data locally and place the tracking files under:

```text
data/raw/skillcorner/
  <match-folder>/
    *_tracking*.jsonl
```

The adapter also accepts `.json` tracking files when the filename contains `tracking`. Multiple
match folders can be placed under `data/raw/skillcorner`; when more than one tracking file is
found, the match folder/file name is used as `match_id` so train/val/test splits remain
match-based.

Prepare windows:

```powershell
python scripts/prepare_tracking_data.py `
  --source skillcorner `
  --raw data/raw/skillcorner `
  --out data/processed/skillcorner_windows.pt `
  --fps-out 10 `
  --context-seconds 2.0 `
  --horizon-seconds 2.0 `
  --stride-seconds 0.2
```

Train the SkillCorner transformer baseline:

```powershell
python scripts/train_baseline.py --config configs/baseline_st_transformer_skillcorner.yaml
```

Evaluate a checkpoint:

```powershell
python scripts/eval_baseline.py --checkpoint runs/st_transformer/<RUN_ID>/best.pt --split test
```

Known limitations:

- Real SkillCorner files are not committed to this repo.
- The adapter focuses on tracking rows, not event labels or tactical phases.
- Possession, phase, and event fields are optional and stay empty when absent.
- If a tracking file has fewer than the required context plus horizon frames, preparation fails
  clearly instead of writing an empty benchmark.
- If public file layouts differ from the supported `ball_data`/`player_data` or frame `data`
  variants, the adapter may need a small schema extension.

## Experiment 2: Soccer TD-JEPA

Experiment 2 pretrains a self-supervised temporal-difference latent state model on the same
canonical 23-entity tracking representation:

```text
z_t + delta_z_t ~= z_t_plus_delta
```

This is representation pretraining only. It does not implement latent flow matching, text
alignment, video processing, tactical discovery, counterfactual search, or a UI.

Prepare synthetic TD-JEPA examples:

```powershell
python scripts/make_synthetic_data.py `
  --out data/synthetic/tracking.parquet `
  --num-matches 3 `
  --num-frames 1000 `
  --fps 10

python scripts/prepare_td_jepa_data.py `
  --source synthetic `
  --raw data/synthetic/tracking.parquet `
  --out data/processed/synthetic_td_jepa.pt `
  --fps-out 10 `
  --context-seconds 1.0 `
  --delta-seconds 0.2 `
  --stride-seconds 0.2
```

Train, evaluate, and export embeddings:

```powershell
python scripts/train_td_jepa.py --config configs/td_jepa_synthetic.yaml
python scripts/eval_td_jepa.py --checkpoint runs/td_jepa/latest.pt --split test
python scripts/export_td_embeddings.py `
  --checkpoint runs/td_jepa/latest.pt `
  --data data/processed/synthetic_td_jepa.pt `
  --out data/processed/synthetic_td_embeddings.pt `
  --split test
```

For SkillCorner, first place downloaded Open Data tracking files under `data/raw/skillcorner/`,
then run:

```powershell
python scripts/prepare_td_jepa_data.py `
  --source skillcorner `
  --raw data/raw/skillcorner `
  --out data/processed/skillcorner_td_jepa.pt `
  --fps-out 10 `
  --context-seconds 1.0 `
  --delta-seconds 0.2 `
  --stride-seconds 0.2

python scripts/train_td_jepa.py --config configs/td_jepa_skillcorner.yaml
python scripts/eval_td_jepa.py --checkpoint runs/td_jepa/latest.pt --split test
python scripts/export_td_embeddings.py `
  --checkpoint runs/td_jepa/latest.pt `
  --data data/processed/skillcorner_td_jepa.pt `
  --out data/processed/skillcorner_td_embeddings.pt `
  --split test
```

TD-JEPA run directories are written to `runs/td_jepa/<timestamp>/` and include:

- `config.yaml`
- `metrics_train.jsonl`
- `metrics_val.jsonl`
- `best.pt`
- `latest.pt`
- `embeddings_sample.pt`

Stable convenience copies are also written to `runs/td_jepa/latest.pt`,
`runs/td_jepa/best.pt`, and `runs/td_jepa/embeddings_sample.pt`.

The embedding export payload contains:

- `z`: `[num_examples, z_dim]`
- `match_id`
- `frame_t`
- `delta_frames`
- `source_split`
- `config`

Downloaded raw data, processed `.pt` files, embeddings, and run artifacts stay local and must not
be committed. They are ignored because they are large, reproducible from public sources, and may be
governed by dataset licenses.

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

