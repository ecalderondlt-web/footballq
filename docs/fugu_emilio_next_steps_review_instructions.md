# Fugu-ultra Emilio Next Steps and Review Instructions

**Reviewer:** fugu-ultra  
**Date:** 2026-06-29  
**Repo/worktree:** `/Users/luislozano/Desktop/GBAI_LLC/footballq_integrity_sprint`  
**Branch:** `codex/research-integrity-sprint-v1`  
**Expected HEAD:** `295d9e6fda94fc540f93d642ac454dbf6f7fd373`

This document is for Emilio. It explains what to do next, what to evaluate, what
is already directionally correct, and where the project is missing scientific or
engineering direction.

The short version:

> Do not scale the model or interpret current clusters yet. First make the
> integrity branch reproducible and provenance-complete, then retrain a
> geometry-only non-overlap representation, then run falsification controls,
> incremental probe tests, discovery baselines, and blinded visualization only
> after those gates pass.

---

## 0. Current preconditions and what you can do today

Read this before running anything in later sections.

### 0.1 You currently have no local data in this worktree

At review time, this worktree has no `data/` directory and no SkillCorner
matches. Verified:

```bash
find data -maxdepth 3 -type f      # -> "No such file or directory"
python scripts/report_skillcorner_availability.py \
  --raw data/raw/skillcorner \
  --processed-dir data/processed \
  --embeddings data/processed/skillcorner_td_embeddings_all.pt \
  --json                            # -> raw_match_count: 0, embedding_match_count: 0
```

Consequence: every SkillCorner command in sections 6-10 (v2 training, probes,
discovery, visualization) will fail immediately, because there is no input. This
is the single most important precondition. Do not interpret a missing-data error
as a code bug.

Continuation note: Emilio's local continuation worktree now has the ten
SkillCorner raw match folders. The current availability report confirms
`raw_periods=1,2` for all ten matches, while the existing h2s processed window
artifact is period 1 only and reports `missing_processed_periods=2` for every
match. The media coverage gap was processed-window coverage, not missing raw
period-2 data; targeted h2s per-match caches have now filled the local
period-2 diagnostic media. The remaining gap is annotation/scientific evidence.
The horizon-preparation resume path now checks cached chunks against raw match
periods, so stale period-1-only cache files are not silently reused. Targeted
`--match-ids --skip-combine` horizon preparation and multi-file/glob `--windows`
rendering recover period-2 diagnostic media without creating a single huge
combined h2s artifact. `scripts/validate_blinded_annotation_package.py` now
passes on both current local diagnostic packages before human annotation.
`scripts/analyze_blinded_annotations.py` now provides the post-review analysis
path and currently reports the balanced package as incomplete until annotation
cells are filled. It rejects filled labels outside the controlled annotation
vocabulary so typo labels block the annotation gate instead of entering the
enrichment summary.

### 0.2 Before any real-data run

1. Place SkillCorner Open Data match folders under `data/raw/skillcorner/`
   (files with `tracking` in the filename, for example
   `*_tracking_extrapolated.jsonl`, plus the `*match*.json` metadata).
2. Confirm the ten reported match IDs are actually present:

   ```bash
   python scripts/report_skillcorner_availability.py \
     --raw data/raw/skillcorner --processed-dir data/processed --json
   ```

3. Cross-check those IDs against `splits/skillcorner_10match_inductive_v1.json`.
   The manifest is explicitly marked `progress_report_unverified`; the split is
   not "verified" until the local files match these IDs:

   ```text
   train: 1886347 1899585 1925299 1953632 1996435 2011166
   val:   2006229 2017461
   test:  2013725 2015213
   ```

4. Do not commit raw data, processed `.pt` files, embeddings, checkpoints, or
   run outputs.

### 0.3 What you should do TODAY, before SkillCorner data is in place

You do not have to wait for data to make progress. The highest-value work right
now is building and unit-testing the missing control code on the synthetic smoke
path, so that when real data arrives the gates in sections 7-10 just run.

Synthetic smoke pipeline (no internet, no real football data):

```bash
python scripts/make_synthetic_data.py \
  --out /tmp/footballq_smoke_tracking.csv --num-matches 3 --num-frames 600 --fps 10 --seed 7

python scripts/prepare_td_jepa_data.py \
  --source synthetic --raw /tmp/footballq_smoke_tracking.csv \
  --out /tmp/footballq_smoke_td_nonoverlap.pt \
  --fps-out 10 --context-seconds 1.0 --delta-seconds 0.2 --stride-seconds 0.2 \
  --objective-mode future_nonoverlap_context_only \
  --prediction-gap-seconds 0.5 --feature-view geometry_only
```

Use this synthetic path to develop and test, in order:

1. the dataset-name allow-list (Task 2);
2. run-manifest wiring (Task 3);
3. the scientific-mode fallback failures (Task 4);
4. the new TD-JEPA control evaluation (section 7);
5. the `raw_plus_z` incremental probe protocol (section 8);
6. the discovery baseline families (section 9).

Develop each as code + a unit test first on synthetic data. Only then run the
real SkillCorner protocol. This is the difference between "I wrote a control" and
"I verified the control fires correctly," which matters for a defensible paper.

---

## 1. First principles for the next sprint

### 1.1 What this project can claim today

Today the project can safely claim:

- It has a football tracking representation-learning codebase.
- It has a serious first pass of research-integrity scaffolding.
- It has an immutable split manifest scaffold.
- It has period-aware `sample_id` helpers.
- It has leakage-aware feature-view scaffolding.
- It has a TD-JEPA legacy-vs-non-overlap objective distinction.
- It has tests that verify several scientific invariants.

Today the project must **not** claim:

- latent clusters are tactical concepts;
- latent residual scores are tactical surprise;
- possession-probe performance proves semantic understanding;
- low TD-JEPA loss proves tactical abstraction;
- raw global-x motion is attacking progression;
- row-level metrics from overlapping windows are independent sample evidence;
- a ten-match split is enough for a strong positive tactical paper without
  strong controls and uncertainty.

### 1.2 The paper path is conditional

The paper is not automatically a positive tactical-representation paper. It can
be one of three valid outcomes:

1. **Positive representation paper** if geometry-only non-overlap TD-JEPA adds
   held-out value beyond raw kinematics, beats shortcut controls, produces stable
   discovery clusters beyond baselines, and survives blinded annotation.
2. **Negative scientific result** if controls show the current signal is driven
   by temporal smoothness, raw kinematics, leakage, missingness, or match identity.
3. **Methods/resource paper** if the main contribution is a reproducible football
   tracking protocol with split lineage, leakage controls, and diagnostics.

Do not decide the paper type before the controls run.

---

## 2. What Emilio is doing correctly

### 2.1 Correct branch and scope

Continue from:

```bash
cd /Users/luislozano/Desktop/GBAI_LLC/footballq_integrity_sprint
git checkout codex/research-integrity-sprint-v1
git rev-parse HEAD
```

Expected:

```text
295d9e6fda94fc540f93d642ac454dbf6f7fd373
```

This is correct. Do **not** review or continue from `main`; it may be behind the
experiment stack.

### 2.2 The scientific objective is correctly framed

The right objective is:

> Determine whether a leakage-controlled, self-supervised football tracking
> representation learns held-out structure that adds value beyond raw
> kinematics and survives falsification controls.

This is better than "make the model bigger" or "find cool clusters."

### 2.3 The docs are honest about the evidence

The existing docs correctly warn that current clusters, possession probes, and
residual scores are diagnostics, not tactical evidence. That honesty is crucial.

Read first:

1. `docs/PAPER_FINAL_PATH.md`
2. `docs/NEXT_WEEK_RUNBOOK.md`
3. `docs/AGENT_REVIEW_HANDOFF.md`
4. `docs/RESEARCH_STATUS.md`
5. `docs/EXPERIMENT_PROTOCOL.md`
6. this file: `docs/fugu_emilio_next_steps_review_instructions.md`

### 2.4 The split/sample/provenance direction is right

Good scaffolds already exist:

- `splits/skillcorner_10match_inductive_v1.json`
- `src/footballq/repro/splits.py`
- `src/footballq/repro/identity.py`
- `src/footballq/repro/feature_views.py`
- `src/footballq/repro/manifest.py`
- `src/footballq/repro/falsification.py`

The period-aware sample identity direction is correct:

```text
sample_id = "{match_id}:{period}:{frame_t}"
```

This avoids period-reset ambiguity.

### 2.5 The feature-view direction is right

The `geometry_only` view correctly excludes possession channels. That matters
because `is_possession_team` and `has_possession` are direct leakage risks for
possession/availability probes.

### 2.6 The non-overlap TD-JEPA direction is right

Keeping the old TD-JEPA objective as `legacy_shifted_overlap` and adding
`future_nonoverlap_context_only` is the right scientific framing.

The legacy objective can remain as an engineering baseline. It must not be
reported as true future prediction.

---

## 3. Where Emilio is missing direction

### 3.1 Do not confuse scaffolding with evidence

The branch contains useful scaffolding. It does **not** yet contain a defensible
scientific result.

The current missing step is not "run more plots." The missing step is:

```text
verify reproducibility → wire provenance → train v2 → falsify shortcuts →
test incremental value → compare discovery baselines → then visualize blindly
```

### 3.2 Provenance is not complete

`src/footballq/repro/manifest.py` defines run-manifest helpers, but scientific
entry points do not consistently write `run_manifest.json`.

Emilio needs to make every paper-relevant run write or explicitly document a
manifest:

- TD-JEPA data preparation;
- TD-JEPA training;
- TD-JEPA evaluation;
- embedding export;
- probe dataset build;
- probe suite;
- latent rollout dataset;
- decoder dataset/training if used;
- transition dataset;
- discovery suite;
- residual diagnostics;
- visualization/annotation outputs.

Minimum manifest fields:

- UTC timestamp;
- git remote;
- branch;
- commit SHA;
- dirty status;
- exact command;
- config path and hash;
- split manifest path and hash;
- feature view;
- objective mode;
- dataset paths;
- output paths;
- warnings;
- Python/dependency/device information where practical.

### 3.3 TD-JEPA controls are not yet enough

The non-overlap objective is necessary but not sufficient.

Because `future_nonoverlap_context_only` zeros `delta_state`, the model can still
perform well by learning an identity/temporal-smoothness solution:

```text
z_pred ≈ z_t
```

This can look good if adjacent football states are similar. Therefore Emilio
must add and report:

- correct temporal pairing;
- shuffled future within batch;
- wrong-match future;
- no-motion / identity predictor;
- reversed-time context;
- masked-ball control;
- team-swap control;
- pitch-reflection control;
- consistent player-slot permutation;
- longer prediction gap.

The required validation relationship is:

```text
correct future loss < shuffled future loss
correct future loss < wrong-match future loss
learned non-overlap predictor beats no-motion / identity
```

If those do not hold, pause and redesign the representation objective.

### 3.4 Probe protocol is missing the central incremental-value test

Current probe comparisons are not enough. The paper question is not "can a probe
read labels from z?" The paper question is:

> Does z add held-out information beyond raw kinematics?

For every valid target, report:

```text
performance(raw)
performance(z)
performance(raw + z)
performance(random)
incremental_value = performance(raw + z) - performance(raw)
```

For regression metrics where lower is better, define the signed improvement
explicitly. For example:

```text
incremental_rmse_improvement = rmse(raw) - rmse(raw + z)
```

Positive is better.

Do not call `z` useful if `raw + z` does not beat `raw` on held-out matches.

### 3.5 Discovery controls are missing

K-means clusters are not evidence by themselves. K-means always partitions data.

Before naming any cluster as tactical, compare TD-JEPA clusters against:

- raw coordinate/velocity transition features;
- PCA of raw transition features;
- handcrafted structure-change metrics;
- random-encoder delta representation;
- multiple clustering seeds.

Also report:

- cluster-size distribution;
- match concentration;
- transition-magnitude concentration;
- held-out occupancy;
- seed stability;
- missingness/visibility concentration;
- whether one match dominates a result.

If TD-JEPA clusters are not more stable or more enriched than these controls,
do not write a tactical-discovery claim.

### 3.6 Visualization is too early as evidence

Visualization is useful only after controls justify looking. It can be built now
as tooling, but it must not validate the current clusters.

Blinded visualization rules:

- annotator-facing files must not show cluster IDs;
- annotator-facing files must not show residual scores;
- annotator-facing files must not show positive/control status;
- the private key file must stay separate;
- examples need matched controls, not just high-score clips.

Current visualization scaffolding is thin. The script
`scripts/render_diagnostic_clips.py` currently writes annotation CSV scaffolds;
it does not implement the runbook's full `--windows --examples --out --blinded`
clip-rendering interface. Fix that mismatch before asking anyone to annotate.

### 3.7 README/runbook commands need alignment

Some README/runbook commands still point at legacy or non-scientific paths.

Every paper-relevant real-data command should include:

```bash
--split-manifest splits/skillcorner_10match_inductive_v1.json
--scientific-mode
```

and should use explicit scientific choices:

```bash
--objective-mode future_nonoverlap_context_only
--feature-view geometry_only
```

Fix the mismatch between:

- runbook output: `data/processed/skillcorner_td_jepa_nonoverlap_geometry.pt`
- config path: `data/processed/skillcorner_td_jepa_nonoverlap.pt`

Recommendation: update the config to use the explicit `*_nonoverlap_geometry.pt`
path.

### 3.8 The ten-match split is not enough by itself

The split is:

```text
train: 6 matches
val:   2 matches
test:  2 matches
```

This is useful for an integrity sprint, but a strong positive paper needs
uncertainty at match/possession/segment level, not just row-level metrics from
overlapping windows.

If results vary strongly by match, expand the data before making tactical claims.

---

## 4. Immediate local verification Emilio should run

Use a clean Python 3.11 or 3.12 environment.

```bash
cd /Users/luislozano/Desktop/GBAI_LLC/footballq_integrity_sprint
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Then:

```bash
python -m pytest -q
python -m ruff check . --statistics
```

Expected current state from fugu-ultra's review:

```text
pytest: 124 passed, 3 warnings
ruff:  48 errors
```

Do not say the repo is lint-clean until Ruff passes or an explicit lint-debt
exception list is committed.

Verify the split:

```bash
python - <<'PY'
from footballq.repro.splits import load_split_manifest

split = load_split_manifest("splits/skillcorner_10match_inductive_v1.json")
print("split_sha256", split.sha256)
print("train", split.train_match_ids)
print("val", split.val_match_ids)
print("test", split.test_match_ids)
PY
```

Expected current hash:

```text
0d66a904f30d38c2721b03b189057cc80f1edbf16fdf42b1a35061a874850c71
```

Check SkillCorner availability:

```bash
python scripts/report_skillcorner_availability.py \
  --raw data/raw/skillcorner \
  --processed-dir data/processed \
  --embeddings data/processed/skillcorner_td_embeddings_all.pt \
  --json
```

Do not claim the ten-match split is verified until the raw local files contain
the expected match IDs.

---

## 5. Patch order before retraining v2

Do these before starting expensive v2 training.

### Task 1: Fix documentation command drift

Files to inspect/update:

- `README.md`
- `docs/NEXT_WEEK_RUNBOOK.md`
- `docs/PAPER_FINAL_PATH.md`
- `docs/AGENT_REVIEW_HANDOFF.md`
- `configs/td_jepa_nonoverlap_skillcorner.yaml`

Required changes:

1. Update reviewed/current commit references to `295d9e6fda94fc540f93d642ac454dbf6f7fd373`
   where appropriate.
2. Make README real-data TD-JEPA commands use:

   ```bash
   --split-manifest splits/skillcorner_10match_inductive_v1.json
   --scientific-mode
   --objective-mode future_nonoverlap_context_only
   --feature-view geometry_only
   ```

3. Align the non-overlap config data path with the runbook output path.
4. Fix `scripts/render_diagnostic_clips.py` command documentation so the
   documented command matches the actual script, or update the script to match
   the desired command.

Validation:

```bash
grep -R "skillcorner_td_jepa_nonoverlap" -n configs docs README.md
grep -R "analyze_tactical_surprise\\|surprise_examples" -n README.md docs scripts src/footballq
python -m ruff check docs README.md
```

### Task 2: Add dataset-name allow-list validation

File:

- `src/footballq/repro/splits.py`

Expected behavior:

- accept known datasets such as `skillcorner`, `synthetic`, `idsse`, `metrica`;
- reject unknown dataset names;
- keep the error message explicit.

Tests:

- update `tests/test_split_manifest.py`;
- add a test that `dataset="nonsense_provider"` raises `ValueError`.

Validation:

```bash
python -m pytest -q tests/test_split_manifest.py
```

### Task 3: Wire run manifests into scientific entry points

Files likely involved:

- `src/footballq/repro/manifest.py`
- `src/footballq/training/train_td_jepa.py`
- `src/footballq/training/export_td_embeddings.py`
- `src/footballq/probes/training.py`
- `src/footballq/latent_flow/train.py`
- `src/footballq/decoding/train.py`
- `src/footballq/discovery/report.py`
- relevant scripts under `scripts/`

Required behavior:

- scientific runs write `run_manifest.json`;
- manifest records command/config/split/feature/objective/git/output metadata;
- non-scientific smoke runs either write a manifest with warnings or document why
  they are exempt.

Validation:

```bash
python -m pytest -q tests/test_run_manifest.py tests/test_scientific_invariants.py
```

Add targeted smoke tests that create a small run directory and assert
`run_manifest.json` exists.

### Task 4: Make scientific mode fail on hidden fallback paths

Scientific mode must not silently fall back to:

- row-order alignment;
- `(match_id, frame_t)` alignment;
- all-row scaler/PCA fitting;
- unknown split assignment;
- missing `sample_id`;
- missing split hash.

Specific issue to fix:

- `src/footballq/discovery/transitions.py` currently normalizes `delta_z` on all
  rows if no `source_split == train` rows are found. In scientific mode, this
  should raise.

Validation:

```bash
python -m pytest -q tests/test_scientific_invariants.py tests/test_discovery_controls.py
```

### Task 5: Rename residual outputs away from surprise language

Files:

- `src/footballq/discovery/surprise.py`
- `src/footballq/discovery/exemplars.py`
- `scripts/analyze_tactical_surprise.py`
- `README.md`
- tests involving residual/surprise outputs

Required direction:

- default outputs should be:

  ```text
  latent_residual_examples.csv
  latent_residual_summary.json
  latent_residual_score
  high_latent_residual
  ```

- old `surprise_*` names can remain only as deprecated aliases or compatibility
  fields clearly marked as legacy.

Validation:

```bash
python -m pytest -q tests/test_residual_score.py tests/test_surprise.py tests/test_exemplars.py
grep -R "tactical_surprise\\|surprise_score\\|surprise_examples" -n README.md docs scripts src/footballq
```

Any remaining `surprise` hit should be either a deprecated alias or a legacy
test, not the default paper path.

### Task 6: Either clean Ruff or commit explicit lint debt

Current Ruff status is 48 errors. Decide one:

1. Fix all Ruff errors; or
2. create a tracked lint-debt document listing exact files/counts and why cleanup
   is deferred.

Validation:

```bash
python -m ruff check . --statistics
```

Do not call the repo lint-clean unless that command passes or debt is documented.

---

## 6. Representation v2 protocol

Only start v2 after the patch-order tasks above are addressed or consciously
deferred with documented risk.

### 6.1 Build the geometry-only non-overlap dataset

Recommended output path:

```text
data/processed/skillcorner_td_jepa_nonoverlap_geometry.pt
```

Command:

```bash
python scripts/prepare_td_jepa_data.py \
  --source skillcorner \
  --raw data/raw/skillcorner \
  --out data/processed/skillcorner_td_jepa_nonoverlap_geometry.pt \
  --fps-out 10 \
  --context-seconds 1.0 \
  --delta-seconds 0.2 \
  --stride-seconds 0.2 \
  --objective-mode future_nonoverlap_context_only \
  --prediction-gap-seconds 0.5 \
  --feature-view geometry_only \
  --split-manifest splits/skillcorner_10match_inductive_v1.json \
  --scientific-mode
```

Expected properties:

- `feature_view == "geometry_only"`;
- no possession channels;
- `objective_mode == "future_nonoverlap_context_only"`;
- context and target frame arrays have zero overlap;
- `sample_id` includes period;
- split hash is recorded.

### 6.2 Train at least three seeds

Update `configs/td_jepa_nonoverlap_skillcorner.yaml` so:

```yaml
data:
  path: data/processed/skillcorner_td_jepa_nonoverlap_geometry.pt
  objective_mode: future_nonoverlap_context_only
  feature_view: geometry_only
split:
  manifest_path: splits/skillcorner_10match_inductive_v1.json
```

Train seeds:

```bash
python scripts/train_td_jepa.py --config configs/td_jepa_nonoverlap_skillcorner.yaml
```

Repeat with at least:

```text
seed 7
seed 17
seed 37
```

Record:

- train/val learning curves;
- embedding variance;
- effective rank or equivalent anti-collapse diagnostic;
- validation loss by split;
- run manifest for each seed.

### 6.3 Export v2 embeddings

```bash
python scripts/export_td_embeddings.py \
  --checkpoint runs/td_jepa/<RUN_ID>/best.pt \
  --data data/processed/skillcorner_td_jepa_nonoverlap_geometry.pt \
  --out data/processed/skillcorner_td_embeddings_nonoverlap_geometry_all.pt \
  --split all
```

Expected:

- embeddings include `sample_id`;
- embeddings include `period`;
- embeddings include `source_split`;
- embeddings include `feature_view`;
- embeddings include `objective_mode`;
- embeddings include split hash.

---

## 7. TD-JEPA falsification-control protocol

### 7.0 First extend the controls module (it is incomplete)

Important: `src/footballq/repro/falsification.py` does not yet support all the
controls this gate requires. Verified `CONTROL_CONDITIONS` currently contains
only:

```text
correct_temporal_pairing
shuffled_future_within_batch
reversed_time_context
masked_ball
team_swap
pitch_reflection
consistent_player_slot_permutation
```

Missing and required before the gate is meaningful:

- `wrong_match_future` — replace each target future with a future from a
  *different match*, not just a different row in the batch. The current
  `shuffled_future_within_batch` permutes within a batch, which can accidentally
  pair same-match rows. A true wrong-match control needs `match_id` awareness, so
  it cannot be implemented by the existing per-batch function alone; do the
  cross-match sampling in the new evaluation script (or extend the function to
  take `match_id`).
- `no_motion_identity` — score a predictor that outputs `z_pred = z_t` (no
  learned delta). This is the single most important control: because
  `future_nonoverlap_context_only` zeros `delta_state`, the model can score well
  by collapsing to identity. If the trained model cannot beat this, there is no
  learned future prediction.
- `longer_prediction_gap` — rebuild data with a larger
  `--prediction-gap-seconds` and confirm the signal degrades sensibly rather than
  staying suspiciously flat.

Add the new conditions to `CONTROL_CONDITIONS`, give each a unit test in
`tests/test_falsification_controls.py` (assert the transform changes the batch
and records `control_condition`), and only then wire them into evaluation.

### 7.1 Evaluation script

Add or extend a script such as:

```text
scripts/eval_td_jepa_controls.py
```

It should evaluate the same checkpoint/data split under:

```text
correct_temporal_pairing
shuffled_future_within_batch
wrong_match_future
no_motion_identity
reversed_time_context
masked_ball
team_swap
pitch_reflection
consistent_player_slot_permutation
longer_prediction_gap
```

Minimum output:

```text
runs/td_jepa_controls/<RUN_ID>/controls_val.json
runs/td_jepa_controls/<RUN_ID>/controls_test.json
runs/td_jepa_controls/<RUN_ID>/run_manifest.json
```

Required table:

| condition | split | loss | cosine | examples | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| correct | val/test | ... | ... | ... | baseline |
| shuffled_future | val/test | ... | ... | ... | shortcut control |
| wrong_match_future | val/test | ... | ... | ... | match-context control |
| no_motion_identity | val/test | ... | ... | ... | temporal smoothness control |

Passing gate:

```text
correct loss < shuffled future loss
correct loss < wrong-match future loss
learned predictor beats no-motion / identity
same ordering holds across seeds
```

If the gate fails, do not continue to tactical discovery. Pause and redesign the
objective or expand data.

---

## 8. Probe protocol

### 8.1 Targets

Treat these as leakage sanity checks only:

- `possession_team`;
- `has_ball_or_possession_available`.

Treat these as geometry controls:

- `future_ball_global_x_bucket`;
- `future_ball_dx_global_m`;
- `future_ball_displacement_m`.

Treat these as weak semantic candidates unless labels are sharpened:

- `team_shape_change_bucket`;
- `team_centroid_shift_m`;
- `team_width_change_m`;
- `team_length_change_m`;
- `stretch_index_change_m`.

Do not call all-visible-player shape change a clean team tactical target. It is a
diagnostic unless explicitly separated into home/away/possession/defending-team
labels.

### 8.2 Add raw-plus-z features

Add a feature source:

```text
raw_plus_z
```

Expected behavior:

```text
raw_plus_z = concat(raw_state_summary, z)
```

Then every valid probe should report:

```text
raw_state_summary
td_jepa
raw_plus_z
random_same_shape
```

### 8.3 Incremental-value report

For classification:

```text
incremental_macro_f1 = macro_f1(raw_plus_z) - macro_f1(raw_state_summary)
```

For regression:

```text
incremental_rmse_improvement = rmse(raw_state_summary) - rmse(raw_plus_z)
incremental_mae_improvement  = mae(raw_state_summary) - mae(raw_plus_z)
```

Positive is better.

Passing gate:

- geometry-only `z` adds value over raw on at least two nontrivial held-out
  targets;
- leakage sanity checks are reported separately;
- uncertainty is reported at match, possession, or segment level;
- no result is only row-level from overlapping windows.

---

## 9. Discovery-control protocol

Current latent clustering is not enough. Add explicit feature families:

1. TD-JEPA delta representation.
2. Raw coordinate/velocity transition features.
3. PCA of raw transition features.
4. Handcrafted structure-change metrics.
5. Random-encoder delta representation.

For each feature family:

- fit scaler/PCA/clusterer on train matches only;
- assign validation/test without refitting;
- run multiple clustering seeds;
- report held-out occupancy;
- report seed stability;
- report match concentration;
- report transition-magnitude concentration;
- report missingness/visibility concentration.

Passing gate:

```text
TD-JEPA clusters are more stable and more enriched than raw/PCA/random/handcrafted controls.
No key result is dominated by one match, missingness pattern, or transition magnitude.
```

If this gate fails, clusters remain exploratory partitions, not tactical motifs.

---

## 10. Blinded visualization and annotation

Visualization should come after controls, not before.

Required outputs:

- top-down frame strips;
- cluster contact sheets;
- high-latent-residual contact sheets;
- matched low-residual controls;
- blinded annotation folder;
- annotation CSV;
- private key CSV.

Annotator-facing files must hide:

- cluster ID;
- latent residual score;
- positive/control status;
- target labels;
- split if it could bias annotation.

The private key file must contain:

- blind ID;
- match ID;
- period;
- frame;
- cluster ID;
- residual score;
- positive/control status;
- split;
- any metadata needed for analysis.

Passing gate:

- blinded human annotation finds reproducible enrichment over matched controls;
- high-residual examples are not mainly tracking jumps, missing-ball frames, or
  extreme acceleration;
- multiple annotators or repeated annotation agreement is preferred if feasible.

---

## 11. Paper decision rules

Use this after v2, probes, discovery controls, and annotation.

### Proceed with a tactical representation paper only if:

- split lineage is used consistently through pretraining and downstream evaluation;
- correct temporal pairing beats shuffled and wrong-match controls;
- learned non-overlap prediction beats no-motion/identity;
- geometry-only `z` adds value over raw kinematics on at least two nontrivial
  held-out targets;
- results are stable across at least three seeds;
- discovery clusters beat raw/PCA/random/handcrafted controls;
- cluster occupancy is not dominated by one match or transition magnitude;
- high residual examples are not primarily tracking artifacts;
- blinded annotation shows enrichment over matched controls;
- uncertainty is reported above row level.

### Write a negative or methods paper if:

- non-overlap TD-JEPA cannot beat shortcut controls;
- `raw + z` does not beat raw;
- clusters do not beat baselines;
- results vary strongly by match;
- annotations do not enrich over matched controls.

### Expand data before claiming tactics if:

- the two test matches dominate conclusions;
- confidence intervals are too wide;
- results change qualitatively by match.

---

## 12. Suggested commit sequence

Use small commits:

1. `docs: align fugu review handoff and scientific command path`
2. `test: cover split dataset validation and scientific fallback failures`
3. `feat: reject unknown split manifest datasets`
4. `feat: write run manifests for scientific entry points`
5. `fix: align nonoverlap geometry dataset path`
6. `feat: add TD-JEPA falsification control evaluation`
7. `feat: add raw-plus-z incremental probe protocol`
8. `feat: add discovery baseline feature families`
9. `fix: rename residual outputs away from surprise terminology`
10. `test: expand scientific integrity checks and settle lint debt`

After each commit:

```bash
python -m pytest -q
python -m ruff check . --statistics
```

If full pytest is too slow during inner loops, run targeted tests first and full
tests before pushing.

---

## 13. Final reminder for Emilio

The fastest defensible paper path is not:

```text
more model scale → prettier clusters → tactical story
```

The fastest defensible path is:

```text
reproducibility → provenance → geometry-only non-overlap v2 →
falsification controls → raw-vs-z incremental value →
discovery baselines → blinded annotation → paper decision
```

If the controls fail, that is still a publishable and useful scientific result
if documented honestly.
