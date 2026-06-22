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
  --out data/processed/skillcorner_td_embeddings_all.pt `
  --split all
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

## Experiment 3: Frozen Latent Probes

Experiment 3 checks whether the frozen TD-JEPA latent space contains recoverable soccer
information. It trains small probe heads on precomputed embeddings and baseline features. The
TD-JEPA encoder is not loaded or fine-tuned by probe training.

Required local inputs:

- TD-JEPA embeddings such as `data/processed/skillcorner_td_embeddings_all.pt`
- Matching SkillCorner windows such as `data/processed/skillcorner_windows.pt`

Build a probe dataset:

```powershell
python scripts/build_probe_dataset.py `
  --embeddings data/processed/skillcorner_td_embeddings_all.pt `
  --windows data/processed/skillcorner_windows.pt `
  --out data/processed/skillcorner_probe_dataset_all.pt `
  --targets possession_team has_ball_or_possession_available phase future_ball_progression_bucket future_ball_displacement_m team_shape_change_bucket
```

The builder aligns embeddings back to windows by `(match_id, frame_t)` and falls back to index
ordering only when no key alignment is possible. Splits are by `match_id`; if only one or two
matches are present, the dataset is marked as a smoke-evaluation split because fully disjoint
train/val/test matches are impossible.

Train and evaluate one probe:

```powershell
python scripts/train_probe.py --config configs/probe_future_ball_progression.yaml
python scripts/eval_probe.py --checkpoint runs/probes/<target>/<feature_source>/<probe_type>/<RUN_ID>/best.pt --split test
```

Run the compact comparison suite:

```powershell
python scripts/run_probe_suite.py `
  --dataset data/processed/skillcorner_probe_dataset_all.pt `
  --out runs/probe_suite/experiment3_all_matches
```

Probe feature sources:

- `td_jepa`: frozen TD-JEPA embeddings `z`
- `random_same_shape`: deterministic random features with the same shape as `z`
- `raw_state_summary`: current-state geometry summaries such as ball position, velocity, team
  centroids, widths, lengths, stretch indices, and centroid distances

Supported initial targets include future ball progression buckets, team shape-change buckets,
future ball displacement regression, future ball dx/dy, team centroid shift, width/length change,
stretch-index change, possession availability, and possession team when possession bits are
present. Phase labels are not invented from window tensors; `configs/probe_phase.yaml` fails
clearly until phase is preserved in a canonical label artifact.

Each probe run writes:

- `runs/probes/<target>/<feature_source>/<probe_type>/<timestamp>/config.yaml`
- `metrics_train.jsonl`
- `metrics_val.jsonl`
- `best.pt`
- `latest.pt`
- `eval_test.json`
- `label_map.json` for classification probes
- `predictions_sample.pt`

The suite writes `results.csv` and `results.json` under the requested output directory. Compare
TD-JEPA against random features first; TD-JEPA beating random on multiple labels is the main sanity
check. Raw-state summaries may beat TD-JEPA on direct geometry labels because they expose current
positions explicitly. For imbalanced classification labels, macro F1 is usually more informative
than accuracy.

Known limitations:

- Current SkillCorner window artifacts do not preserve phase strings or event labels.
- Public SkillCorner possession fields may be absent; when present fields are empty,
  `possession_team` is labeled `unknown`.
- Future ball progression uses raw x displacement because attacking direction is not stored in the
  current window tensors.
- Team shape-change labels use all visible players when possession-team identity is unavailable.

## Experiment 4A: Latent Flow Matching

Experiment 4A tests latent-space generative rollout. It uses frozen TD-JEPA embeddings as states
and trains a conditional flow-matching model to generate future latent sequences from past latent
context. It does not fine-tune the TD-JEPA encoder and does not decode latents back to coordinate
trajectories. Coordinate decoding is reserved for Experiment 4B.

Required input:

- TD-JEPA embeddings, preferably the all-match export:
  `data/processed/skillcorner_td_embeddings_all.pt`

Build a latent rollout dataset:

```powershell
python scripts/build_latent_rollout_dataset.py `
  --embeddings data/processed/skillcorner_td_embeddings_all.pt `
  --out data/processed/skillcorner_latent_rollout_dataset.pt `
  --context-steps 5 `
  --horizon-steps 5 `
  --stride-steps 1
```

The builder creates examples shaped like:

- `past_z`: `[num_examples, context_steps, latent_dim]`
- `future_z`: `[num_examples, horizon_steps, latent_dim]`
- `future_mask`: `[num_examples, horizon_steps]`

Examples are built within each `match_id` only and never cross match boundaries. Splits are by
`match_id`; if fewer than three matches are present, the dataset records a smoke-split warning.

Train latent flow:

```powershell
python scripts/train_latent_flow.py --config configs/latent_flow_skillcorner.yaml
```

Evaluate deterministic baselines:

```powershell
python scripts/eval_latent_flow.py `
  --dataset data/processed/skillcorner_latent_rollout_dataset.pt `
  --baseline last_latent `
  --split test

python scripts/eval_latent_flow.py `
  --dataset data/processed/skillcorner_latent_rollout_dataset.pt `
  --baseline constant_latent_velocity `
  --split test
```

Evaluate a trained flow checkpoint:

```powershell
python scripts/eval_latent_flow.py `
  --checkpoint runs/latent_flow/<RUN_ID>/best.pt `
  --dataset data/processed/skillcorner_latent_rollout_dataset.pt `
  --split test
```

Sample latent futures:

```powershell
python scripts/sample_latent_flow.py `
  --checkpoint runs/latent_flow/<RUN_ID>/best.pt `
  --dataset data/processed/skillcorner_latent_rollout_dataset.pt `
  --split test `
  --num-examples 8 `
  --num-samples 8 `
  --out runs/latent_flow/<RUN_ID>/samples_test.pt
```

Run a compact comparison suite:

```powershell
python scripts/run_latent_flow_suite.py `
  --dataset data/processed/skillcorner_latent_rollout_dataset.pt `
  --checkpoint runs/latent_flow/<RUN_ID>/best.pt `
  --out runs/latent_flow_suite/experiment4a
```

The suite writes `results.csv` and `results.json` with:

- `latent_ADE`: mean latent L2 error over future steps
- `latent_FDE`: latent L2 error at the final future step
- `latent_step_mse`: per-step latent MSE
- `latent_cosine_similarity`: cosine similarity between predicted and true future latents
- `minADE_8` and `minFDE_8`: best-of-8 sample metrics for flow outputs
- `diversity_mean_pairwise_distance`: sample diversity for multi-sample flow outputs

Known limitations:

- Metrics are latent-space diagnostics, not coordinate-space trajectory quality.
- The first sampler is fixed-step Euler integration.
- No text alignment, video processing, tactical discovery UI, or counterfactual tooling is included.

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

