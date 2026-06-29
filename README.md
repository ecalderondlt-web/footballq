# footballq

`footballq` is a research codebase for learning football game-state dynamics from structured
tracking data. It now contains Phase 1 trajectory foundations plus experimental TD-JEPA,
probe, latent-flow, decoder, and discovery tooling.

Current latent clusters, probe scores, and latent residual rankings are not validated tactical
evidence. Before tactical claims or Experiment 6 work, read:

- `docs/PAPER_FINAL_PATH.md`
- `docs/RESEARCH_STATUS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/NEXT_WEEK_RUNBOOK.md`
- `docs/AGENT_REVIEW_HANDOFF.md`
- `docs/RESEARCH_INTEGRITY_HANDOFF.md`

The fastest paper path is an integrity-controlled representation study:
finish split/provenance controls, retrain the geometry-only non-overlap
representation, run falsification and incremental probe controls, then use
blinded visualization only as diagnostic evidence.

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
  --targets possession_team has_ball_or_possession_available phase future_ball_global_x_bucket future_ball_displacement_m team_shape_change_bucket
```

The builder aligns embeddings back to windows by period-aware `sample_id`. Legacy alignment must
be enabled explicitly for non-scientific inspection. Splits are by `match_id`; if only one or two
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

## Experiment 4B: Residual Latent Flow Matching

Experiment 4B keeps the latent-space scope of Experiment 4A but changes the target. Instead of
generating absolute future TD-JEPA latents from noise, it models residuals around a strong smooth
latent baseline:

```text
residual_future_z = true_future_z - baseline_future_z
future_z_hat = baseline_future_z + residual_flow(past_z)
```

Supported residual baselines:

- `last_latent`: repeats the latest context latent for the full horizon
- `constant_latent_velocity`: extrapolates the latest latent velocity

Residual targets are normalized with train-split statistics only. Evaluation always reports metrics
back in original latent units.

Build the rollout dataset with residual fields:

```powershell
python scripts/build_latent_rollout_dataset.py `
  --embeddings data/processed/skillcorner_td_embeddings_all.pt `
  --out data/processed/skillcorner_latent_rollout_dataset.pt `
  --context-steps 5 `
  --horizon-steps 5 `
  --stride-steps 1 `
  --residual-mode constant_latent_velocity
```

Train residual flow:

```powershell
python scripts/train_latent_flow.py --config configs/latent_flow_residual_last.yaml
python scripts/train_latent_flow.py --config configs/latent_flow_residual_cv.yaml
```

Evaluate a residual checkpoint:

```powershell
python scripts/eval_latent_flow.py `
  --checkpoint runs/latent_flow/<RUN_ID>/best.pt `
  --split test
```

Run the comparison suite:

```powershell
python scripts/run_latent_flow_suite.py `
  --dataset data/processed/skillcorner_latent_rollout_dataset.pt `
  --checkpoint runs/latent_flow/<MLP_RUN>/best.pt `
  --checkpoint runs/latent_flow/<ABS_FLOW_RUN>/best.pt `
  --checkpoint runs/latent_flow/<RESIDUAL_RUN>/best.pt `
  --out runs/latent_flow_suite/experiment4b
```

Run a small residual-flow ablation:

```powershell
python scripts/run_latent_flow_ablation.py `
  --base-config configs/latent_flow_residual_cv_small.yaml `
  --out runs/latent_flow_ablation/experiment4b `
  --noise-scales 0.0 0.1 0.3 `
  --num-steps 5 10 `
  --num-samples 4
```

## Experiment 4B.1: Stochastic Residual-Flow Ablation

Experiment 4B.1 keeps the residual latent-flow setup and asks whether stochastic sampling can
improve best-of-sample metrics without ruining average rollout quality. This is still latent-space
only: no coordinate decoder, raw-coordinate flow, video processing, text alignment, or TD-JEPA
fine-tuning is introduced here.

Residual flow is used instead of absolute latent flow because the TD-JEPA future latents are smooth
and the deterministic smooth baselines are already strong. The flow model therefore learns the
remaining correction around `constant_latent_velocity` or `last_latent`, not the whole future from
scratch.

Longer residual-CV training:

```powershell
python scripts/train_latent_flow.py --config configs/latent_flow_residual_cv.yaml
```

Resume after an interruption:

```powershell
python scripts/train_latent_flow.py `
  --config configs/latent_flow_residual_cv.yaml `
  --resume runs/latent_flow/<RUN_ID>/latest.pt
```

The trainer records `epoch` and global `step` in checkpoints, writes `latest.pt` during long epochs
when `training.save_every_steps` is set, and logs train loss plus validation ADE/FDE/cosine.

Evaluate a resumed checkpoint:

```powershell
python scripts/eval_latent_flow.py `
  --checkpoint runs/latent_flow/<RUN_ID>/best.pt `
  --split test
```

Run the full stochastic ablation grid:

```powershell
python scripts/run_latent_flow_ablation.py `
  --base-config configs/latent_flow_residual_cv.yaml `
  --checkpoint runs/latent_flow/<RUN_ID>/best.pt `
  --out runs/latent_flow_ablation/experiment4b1 `
  --noise-scales 0.0 0.01 0.03 0.05 0.1 `
  --num-steps 5 10 20 `
  --num-samples 4 8 16
```

The ablation writes:

- `runs/latent_flow_ablation/experiment4b1/results.csv`
- `runs/latent_flow_ablation/experiment4b1/summary.json`

Interpretation:

- `latent_ADE` and `latent_FDE` are mean rollout quality across samples; lower is better.
- `minADE` and `minFDE` are best-of-sample metrics; they can improve when sampling creates useful
  alternatives.
- `diversity_mean_pairwise_distance` and `sample_std_mean` show whether the sampler is actually
  producing distinct futures.
- A stochastic config is treated as unacceptable by default if mean `latent_ADE` is more than 2x the
  `constant_latent_velocity` baseline ADE, even if its best-of-sample metric improves.

Useful metrics:

- `latent_ADE`, `latent_FDE`, `latent_RMSE`: future latent rollout error
- `one_step_error`: first future latent error
- `multi_step_rollout_error`: average horizon error, equivalent to latent ADE
- `residual_ADE`: residual-space error for residual checkpoints
- `delta_ADE`: error after subtracting the latest context latent
- `minADE`, `minFDE`, `minADE_4`, `minADE_8`, `minFDE_4`, `minFDE_8`: best-of-k sample diagnostics
- `diversity_mean_pairwise_distance`: sample diversity; in residual configs this is controlled by
  `flow.noise_scale`
- `sample_std_mean`: mean standard deviation across sampled futures

Known limitations:

- Residual flow is still a latent-space diagnostic and does not decode coordinates.
- Smooth latent baselines remain very strong because TD-JEPA embeddings are temporally smooth.
- `deterministic_mean_eval: true` and low `noise_scale` are useful first checks; larger stochastic
  noise should be treated as an ablation, not a default quality setting.

## Experiment 4C: Coordinate Decoding From TD-JEPA Latents

Experiment 4C bridges frozen TD-JEPA latents back to football coordinates. It trains lightweight
decoders on precomputed embeddings and SkillCorner tracking windows; the TD-JEPA encoder remains
frozen and is not loaded for fine-tuning.

Required upstream artifacts:

- `data/processed/skillcorner_windows.pt`
- `data/processed/skillcorner_td_embeddings_all.pt`
- optionally `data/processed/skillcorner_latent_rollout_dataset.pt` and a residual-flow checkpoint
  for broader latent-rollout comparisons

Build the decoder dataset:

```powershell
python scripts/build_decoder_dataset.py `
  --embeddings data/processed/skillcorner_td_embeddings_all.pt `
  --windows data/processed/skillcorner_windows.pt `
  --out data/processed/skillcorner_decoder_dataset.pt `
  --horizon-steps 20
```

Train current-state reconstruction:

```powershell
python scripts/train_coordinate_decoder.py --config configs/decoder_reconstruct_current.yaml
```

Train future coordinate decoders:

```powershell
python scripts/train_coordinate_decoder.py --config configs/decoder_future_from_z.yaml
python scripts/train_coordinate_decoder.py --config configs/decoder_future_from_context.yaml
python scripts/train_coordinate_decoder.py --config configs/decoder_rollout_from_latents.yaml
```

Evaluate a decoder checkpoint:

```powershell
python scripts/eval_coordinate_decoder.py `
  --checkpoint runs/decoders/<MODE>/<DECODER>/<RUN_ID>/best.pt `
  --split test
```

Run the compact comparison suite:

```powershell
python scripts/run_decoder_suite.py `
  --dataset data/processed/skillcorner_decoder_dataset.pt `
  --out runs/decoder_suite/experiment4c
```

Decoder tasks:

- `reconstruct_current`: `z_t -> current_xy`
- `future_from_z`: `z_t -> future_xy`
- `future_from_context`: `z_context -> future_xy`
- `rollout_from_latents`: future latent sequence -> future coordinates

Metrics are reported in meters after denormalizing the model outputs:

- Future prediction: `player_ADE_m`, `player_FDE_m`, `ball_ADE_m`, `ball_FDE_m`,
  `all_entity_ADE_m`, `all_entity_FDE_m`, team shape errors
- Reconstruction: `current_player_error_m`, `current_ball_error_m`,
  `current_all_entity_error_m`, current team shape errors
- Sampled rollout hooks: `mean_ADE_m`, `mean_FDE_m`, `minADE_k_m`, `minFDE_k_m`,
  `coordinate_diversity_m`

Known limitations:

- The decoder is intentionally lightweight; it is a diagnostic bridge, not a full coordinate world
  model.
- Generated decoder datasets and run artifacts stay under `data/` and `runs/` and should not be
  committed.
- This phase does not add raw-coordinate flow matching, video, text alignment, UI, tactical
  discovery, counterfactual search, or TD-JEPA fine-tuning.

## Experiment 4C.1: Decoder Learning Curve And Residual Context Decoding

Experiment 4C.1 asks whether weak coordinate decoding is mainly caused by too little data, too weak
a decoder formulation, or both. It adds match-count learning curves, raw-context ablations, and
residual coordinate decoders around the coordinate constant-velocity baseline.

Why this matters:

- Three local SkillCorner matches are enough for smoke testing but too little for a serious
  coordinate-decoder conclusion.
- A single `z_t` may not preserve exact absolute coordinates.
- Context-conditioned decoders test whether TD-JEPA embeddings add value beyond observable past
  coordinates; they do not prove that `z_t` alone is sufficient.

Place SkillCorner Open Data locally under:

```text
data/raw/skillcorner/
```

Raw match files are ignored by Git. If internet download is unavailable, manually download the
SkillCorner Open Data matches and keep their tracking JSON/JSONL files under that directory.

Rebuild all available SkillCorner windows:

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

Rebuild TD-JEPA data, train/export embeddings with the existing Experiment 2 commands, then rebuild
the decoder dataset:

```powershell
python scripts/build_decoder_dataset.py `
  --embeddings data/processed/skillcorner_td_embeddings_all.pt `
  --windows data/processed/skillcorner_windows.pt `
  --out data/processed/skillcorner_decoder_dataset.pt `
  --horizon-steps 20
```

Run the learning curve:

```powershell
python scripts/run_decoder_learning_curve.py `
  --dataset data/processed/skillcorner_decoder_dataset.pt `
  --out runs/decoder_learning_curve/experiment4c1
```

The learning-curve CSV includes match counts, train/val/test match IDs, split disjointness,
smoke-split flags, and finite meter metrics for:

- `coordinate_constant_velocity`
- `last_coordinate_position`
- `raw_past_summary_mlp`
- `z_only_decoder`
- `context_only_decoder`
- `z_plus_context_decoder`
- `residual_context_only_decoder`
- `residual_z_plus_context_decoder`
- current-state reconstruction variants

You can also run the same learning-curve settings from the checked-in config:

```powershell
python scripts/run_decoder_learning_curve.py --config configs/decoder_learning_curve_skillcorner.yaml
```

Train stronger real configs:

```powershell
python scripts/train_coordinate_decoder.py --config configs/decoder_reconstruct_current_real.yaml
python scripts/train_coordinate_decoder.py --config configs/decoder_context_reconstruct_current_real.yaml
python scripts/train_coordinate_decoder.py --config configs/decoder_residual_context_future_real.yaml
```

Evaluate:

```powershell
python scripts/eval_coordinate_decoder.py `
  --checkpoint runs/decoders/<MODE>/<DECODER>/<RUN_ID>/best.pt `
  --split test
```

Interpretation:

- `context_only` beating `z_only` means raw observable state dominates exact coordinate decoding.
- `z_plus_context` beating `context_only` is the key sign that TD-JEPA embeddings add measurable
  value beyond raw past coordinates.
- `residual_z_plus_context_decoder` approaching `coordinate_constant_velocity` means learned
  corrections are becoming meaningful.
- If only one or two matches are available, reported results are smoke-only and not a real
  match-generalization test.

Do not claim that latent rollouts beat raw coordinate baselines unless the meter-space metrics show
it. This phase still excludes text alignment, video, UI, counterfactuals, raw-coordinate flow
matching, and TD-JEPA fine-tuning.

## Experiment 4C.2: Longer Horizons And Stress Slices

Experiment 4C.2 asks whether residual coordinate decoding becomes more useful when evaluation uses
more local SkillCorner matches, longer horizons, and high-change windows where constant velocity is
less forgiving.

Raw and processed data are still local artifacts. Keep SkillCorner Open Data under:

```text
data/raw/skillcorner/
```

The adapter discovers every tracking JSON/JSONL file below that directory whose filename contains
`tracking`. `data/` and `runs/` are ignored because they can contain raw match data, processed
windows, checkpoints, and generated reports; commit only code, configs, docs, and tests.

Prepare 2s, 4s, and 6s window files:

```powershell
python scripts/prepare_tracking_data.py --source skillcorner --raw data/raw/skillcorner --out data/processed/skillcorner_windows_h2s.pt --fps-out 10 --context-seconds 2.0 --horizon-seconds 2.0 --stride-seconds 0.2
python scripts/prepare_tracking_data.py --source skillcorner --raw data/raw/skillcorner --out data/processed/skillcorner_windows_h4s.pt --fps-out 10 --context-seconds 2.0 --horizon-seconds 4.0 --stride-seconds 0.2
python scripts/prepare_tracking_data.py --source skillcorner --raw data/raw/skillcorner --out data/processed/skillcorner_windows_h6s.pt --fps-out 10 --context-seconds 2.0 --horizon-seconds 6.0 --stride-seconds 0.2
```

Or write all three horizon files with the multi-horizon helper:

```powershell
python scripts/prepare_tracking_horizons.py --source skillcorner --raw data/raw/skillcorner --out-dir data/processed --prefix skillcorner_windows --fps-out 10 --context-seconds 2.0 --horizon-seconds 2.0 4.0 6.0 --stride-seconds 0.2
```

Each command reports discovered match IDs and windows per match. One- or two-match local runs are
allowed for debugging, but real split evaluation requires at least three matches so train, val, and
test are disjoint by `match_id`.

Build decoder datasets for each horizon:

```powershell
python scripts/build_decoder_dataset.py --embeddings data/processed/skillcorner_td_embeddings_all.pt --windows data/processed/skillcorner_windows_h2s.pt --out data/processed/skillcorner_decoder_dataset_h2s.pt --horizon-steps 20
python scripts/build_decoder_dataset.py --embeddings data/processed/skillcorner_td_embeddings_all.pt --windows data/processed/skillcorner_windows_h4s.pt --out data/processed/skillcorner_decoder_dataset_h4s.pt --horizon-steps 40
python scripts/build_decoder_dataset.py --embeddings data/processed/skillcorner_td_embeddings_all.pt --windows data/processed/skillcorner_windows_h6s.pt --out data/processed/skillcorner_decoder_dataset_h6s.pt --horizon-steps 60
```

The embeddings only need to be regenerated if their `match_id`/`frame_t` rows do not align with the
new window files. When longer horizons remove end-of-match windows, the decoder builder drops
unmatched embedding rows and reports the count.

Run the multi-horizon residual learning curve and stress-slice evaluation:

```powershell
python scripts/run_decoder_learning_curve.py `
  --datasets data/processed/skillcorner_decoder_dataset_h2s.pt data/processed/skillcorner_decoder_dataset_h4s.pt data/processed/skillcorner_decoder_dataset_h6s.pt `
  --out runs/decoder_learning_curve/experiment4c2 `
  --models coordinate_constant_velocity last_coordinate_position residual_context_only residual_z_plus_context `
  --epochs 3 `
  --batch-size 256
```

Equivalent config-driven command:

```powershell
python scripts/run_decoder_learning_curve.py --config configs/decoder_learning_curve_skillcorner_4c2.yaml
```

The CSV contains overall rows and stress-slice rows for:

- `all_windows`
- `high_future_ball_displacement`
- `high_ball_acceleration`
- `high_ball_direction_change`
- `high_team_shape_change`
- `high_team_width_change`
- `high_team_length_change`
- `high_stretch_index_change`
- `possession_change` when explicit future possession-change labels exist
- `event_near_window` when window event metadata is known

High-change slices use top-25-percent thresholds computed from future ground truth for evaluation
grouping only. These labels are not fed into decoder inputs. Thresholds and slice counts are written
to `summary.json`.

Interpretation:

- `residual_z_plus_context_decoder` beating `residual_context_only_decoder` means TD-JEPA `z`
  adds measurable value beyond raw past coordinates for that horizon or slice.
- Beating `coordinate_constant_velocity` on a stress slice is stronger evidence than improving only
  over direct latent decoders, because constant velocity is the short-horizon reference to beat.
- If all available equals three matches, treat results as proof-of-concept. It is not final science
  until more SkillCorner matches or another real dataset are included.
- This experiment still excludes text alignment, video, UI, tactical discovery, counterfactuals,
  latent stochasticity tuning, and TD-JEPA fine-tuning.

## Experiment 4C.3: All-Available-Games Decoder Scale Validation

Experiment 4C.3 asks whether residual `z_plus_context` decoding becomes consistently better than
residual raw-context decoding when more real matches, longer horizons, and stress slices are
available. It is a scale-validation report, not a new architecture phase.

Verify local SkillCorner availability:

```powershell
python scripts/report_skillcorner_availability.py `
  --raw data/raw/skillcorner `
  --processed-dir data/processed `
  --embeddings data/processed/skillcorner_td_embeddings_all.pt
```

The report prints raw match IDs, tracking/metadata/event availability, window counts per horizon,
decoder examples per horizon, examples per match, and embedding/window key alignment.

Prepare all horizons with resumable per-match caching:

```powershell
python scripts/prepare_tracking_horizons.py `
  --source skillcorner `
  --raw data/raw/skillcorner `
  --out-dir data/processed `
  --prefix skillcorner_windows `
  --fps-out 10 `
  --context-seconds 2.0 `
  --horizon-seconds 2.0 4.0 6.0 `
  --stride-seconds 0.2 `
  --resume
```

By default this processes SkillCorner match folders one at a time and caches per-match horizon
windows under `data/processed/.skillcorner_window_cache/`, which makes long 6s runs easier to
resume. Use `--combined-load` only when you explicitly want a one-shot raw load.

Build decoder datasets:

```powershell
python scripts/build_decoder_dataset.py --embeddings data/processed/skillcorner_td_embeddings_all.pt --windows data/processed/skillcorner_windows_h2s.pt --out data/processed/skillcorner_decoder_dataset_h2s.pt --horizon-steps 20
python scripts/build_decoder_dataset.py --embeddings data/processed/skillcorner_td_embeddings_all.pt --windows data/processed/skillcorner_windows_h4s.pt --out data/processed/skillcorner_decoder_dataset_h4s.pt --horizon-steps 40
python scripts/build_decoder_dataset.py --embeddings data/processed/skillcorner_td_embeddings_all.pt --windows data/processed/skillcorner_windows_h6s.pt --out data/processed/skillcorner_decoder_dataset_h6s.pt --horizon-steps 60
```

If a new raw match has windows but no TD-JEPA embeddings, the decoder builder fails clearly instead
of silently dropping that match. Rebuild all-match TD-JEPA data and embeddings with the existing
Experiment 2 scripts, then rerun the decoder build.

Run all-available scale validation:

```powershell
python scripts/run_decoder_learning_curve.py `
  --datasets `
    data/processed/skillcorner_decoder_dataset_h2s.pt `
    data/processed/skillcorner_decoder_dataset_h4s.pt `
    data/processed/skillcorner_decoder_dataset_h6s.pt `
  --out runs/decoder_learning_curve/experiment4c3_all_available `
  --models coordinate_constant_velocity last_coordinate_position residual_context_only residual_z_plus_context `
  --match-counts 1 3 all `
  --epochs 5 `
  --batch-size 256 `
  --require-real-split
```

For a faster local smoke run, add:

```powershell
--max-train-batches 20 --max-eval-batches 20
```

Outputs:

- `runs/decoder_learning_curve/experiment4c3_all_available/results.csv`
- `runs/decoder_learning_curve/experiment4c3_all_available/stress_results.csv`
- `runs/decoder_learning_curve/experiment4c3_all_available/summary.json`

Interpretation:

- Positive signal: residual `z_plus_context` consistently beats residual context-only overall,
  especially at 4s/6s or on high-change stress slices.
- Negative signal: `z_plus_context` gains remain tiny or inconsistent, while coordinate constant
  velocity dominates all horizons and stress slices.
- With only three local SkillCorner matches, this remains a limited scale check. Add the remaining
  SkillCorner Open Data matches under `data/raw/skillcorner/` before treating the result as final.

Do not commit raw SkillCorner data, processed windows, decoder datasets, checkpoints, or reports.

## Experiment 5: latent transition discovery and residual diagnostics

Experiment 5 freezes coordinate decoding as infrastructure and analyzes TD-JEPA latent transitions
directly. It builds examples of `z_t`, `z_next`, `delta_z`, transition metadata, cluster labels,
latent residual scores, and representative exemplars for later visual inspection.

Build a transition dataset from all-match TD-JEPA embeddings and prepared windows:

```bash
python scripts/build_transition_dataset.py \
  --embeddings data/processed/skillcorner_td_embeddings_all.pt \
  --windows data/processed/skillcorner_windows_h2s.pt \
  --out data/processed/skillcorner_transition_dataset.pt \
  --delta-steps 2 5 10 \
  --fps 10
```

Cluster one transition horizon:

```bash
python scripts/cluster_latent_transitions.py \
  --dataset data/processed/skillcorner_transition_dataset.pt \
  --out runs/discovery/transition_clusters_h2 \
  --delta-seconds 0.2 \
  --k 8 16 32 64 \
  --feature normalized_delta_z
```

Analyze latent residual scores:

```bash
python scripts/analyze_tactical_surprise.py \
  --dataset data/processed/skillcorner_transition_dataset.pt \
  --out runs/discovery/surprise_h2 \
  --delta-seconds 0.2
```

Run the full discovery suite:

```bash
python scripts/run_discovery_suite.py --config configs/discovery_suite_skillcorner.yaml
```

The suite writes generated artifacts under `runs/discovery/experiment5_skillcorner/`, including
per-delta cluster CSVs, `enrichment_k32.csv`, `exemplars_k32.csv`, `surprise_examples.csv`,
`summary.json`, and `report.md`. Inspect `cluster_summary.json` for cluster quality, enrichment
CSVs for label associations, and residual examples for high-change latent windows.

Interpretation is deliberately conservative. Positive diagnostic evidence would be recurring
clusters that are not dominated by one match and are enriched for future ball displacement,
team-shape change, or high-residual/stress windows. Negative or weak evidence still leaves a useful
harness for visual inspection and blinded annotation. Current limitations include sparse/unknown
phase and event labels, no human tactical labels, and fully unsupervised cluster semantics.

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
