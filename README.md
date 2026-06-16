# footballq

`footballq` is a Phase 1 data foundation for learning football game-state dynamics from
structured tracking data. The long-term target is generative trajectory modeling for tactical
patterns linked to chance creation, defensive failure, and goals.

Phase 1 does not train a neural network. It builds reproducible ingestion, normalization,
feature, windowing, visualization, and baseline rollout tools around canonical football tracking
tables.

## Install

```powershell
cd footballq
python -m pip install -e ".[dev]"
```

Python 3.11 or newer is required.

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

The fixed-length window export is designed for future tensors shaped like
`[time, agent, feature]`, with masks for missing players and ball observations.

## Phase Boundary

This repository deliberately avoids PyTorch, TensorFlow, raw broadcast video, and computer-vision
pipelines in Phase 1. The learning object is structured tracking state, not pixels.

