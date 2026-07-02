# Next Week Runbook

This is the execution plan Emilio should follow to move the project toward a
paper. The purpose is to create defensible evidence, not to produce impressive
numbers quickly.

Read first:

- `docs/PAPER_FINAL_PATH.md`
- `docs/RESEARCH_STATUS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/AGENT_REVIEW_HANDOFF.md`

## Global Rules

- Use `codex/research-integrity-sprint-v1` or a branch created from it.
- Do not train paper runs without `--split-manifest`.
- Do not call outputs held-out unless encoder training, preprocessing fits, PCA,
  clustering, and model selection all exclude validation/test matches.
- Treat one-match or two-match runs as `smoke_only`.
- Do not claim current clusters are tactical concepts.
- Do not call latent residual scores tactical surprise.
- Do not interpret possession/availability probes as learned semantics when the
  encoder saw possession channels.

## Day 1: Reproducibility And Split Verification

Purpose: make sure the branch can be built and the ten-match protocol is real.

Commands:

```bash
git fetch --all --prune
git checkout codex/research-integrity-sprint-v1
git pull --ff-only
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

Validate the split manifest:

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

Expected outputs:

- full tests pass in Emilio's environment
- Ruff is either clean or exact remaining lint debt is recorded
- split hash is printed and reused in all later artifacts

Common failure modes:

- missing `typer`, `pyyaml`, `pyarrow`, or torch because the editable install was
  not run
- local SkillCorner files do not match the reported ten match IDs
- README commands still omit `--split-manifest`

Do not claim:

- "the repo is reproducible" unless a fresh install and full test suite pass
- "the split is verified" unless local files contain the expected match IDs

## Day 2: Finish Integrity Wiring

Purpose: close the remaining engineering gaps before running expensive jobs.

Verify or complete:

1. Make every scientific entry point write a run manifest.
2. Update README commands to use `--split-manifest` and `--scientific-mode`.
3. Add dataset-name rejection to split validation.
4. Rename default residual output files away from `surprise_*`, keeping old names
   only as deprecated aliases if needed.
5. Either clean Ruff repo-wide or document exact lint debt.

Suggested validation:

```bash
python -m pytest -q tests/test_run_manifest.py tests/test_split_manifest.py
python -m pytest -q tests/test_scientific_invariants.py
python -m ruff check .
```

Expected outputs:

- run directories contain `run_manifest.json`
- scientific artifacts include `split_manifest_sha256`
- invalid dataset names in split manifests fail

Do not claim:

- "provenance is solved" if any scientific command can still write artifacts
  without command/config/split metadata

## Day 3: Build Representation Datasets And Train V2

Purpose: produce the representation runs that can support or falsify the paper.

Legacy baseline:

```bash
python scripts/prepare_td_jepa_data.py \
  --source skillcorner \
  --raw data/raw/skillcorner \
  --out data/processed/skillcorner_td_jepa_legacy.pt \
  --objective-mode legacy_shifted_overlap \
  --feature-view full_state_legacy \
  --split-manifest splits/skillcorner_10match_inductive_v1.json \
  --scientific-mode
```

Primary non-overlap geometry-only dataset:

```bash
python scripts/prepare_td_jepa_data.py \
  --source skillcorner \
  --raw data/raw/skillcorner \
  --out data/processed/skillcorner_td_jepa_nonoverlap_geometry.pt \
  --objective-mode future_nonoverlap_context_only \
  --prediction-gap-seconds 0.5 \
  --feature-view geometry_only \
  --split-manifest splits/skillcorner_10match_inductive_v1.json \
  --scientific-mode
```

Train the non-overlap representation using the non-overlap config. Use at least
three seeds before interpreting stability:

```bash
python scripts/train_td_jepa.py --config configs/td_jepa_nonoverlap_skillcorner.yaml
```

Expected outputs:

- TD data has period-aware `sample_id`
- context and target frame-index arrays have no overlap in non-overlap mode
- checkpoints record `feature_view`, `objective_mode`, and split hash

Common failure modes:

- config points at the wrong TD data path
- a smoke dataset is accidentally used for a paper run
- the old overlap objective is reported as future prediction

Do not claim:

- low TD loss means tactical abstraction
- the legacy overlap result is a future-prediction result

## Day 4: Falsification And Probe Controls

Purpose: test whether the representation learned anything beyond easy shortcuts.

Run TD controls:

- correct temporal pairing
- shuffled future within batch
- wrong-match future
- reversed-time context
- no-motion/no-future predictor
- masked ball
- team swap
- pitch reflection
- consistent player-slot permutation

Expected comparison:

```text
correct pairing loss < shuffled future loss
correct pairing loss < wrong-match future loss
non-overlap predictor beats identity/no-motion controls
```

Build leakage-controlled probes with the same split manifest:

```bash
python scripts/build_probe_dataset.py \
  --embeddings data/processed/skillcorner_td_embeddings_all.pt \
  --windows data/processed/skillcorner_windows_h2s.pt \
  --out data/processed/skillcorner_probe_dataset.pt \
  --targets future_ball_global_x_bucket future_ball_displacement_m team_shape_change_bucket \
  --split-manifest splits/skillcorner_10match_inductive_v1.json \
  --scientific-mode
```

Report for each valid probe:

```text
performance(raw)
performance(z)
performance(raw + z)
incremental_value = performance(raw + z) - performance(raw)
```

Expected outputs:

- possession probes are separated as leakage sanity checks
- global-x targets are not called attacking progression
- raw baselines and `z` baselines use the same split and examples

Do not claim:

- possession classification is semantic emergence
- raw global-x displacement is tactical progression
- `z` is useful if `raw + z` does not beat `raw`

## Day 5: Discovery Controls And Blinded Visualization

Purpose: test whether latent clusters survive baselines before visual
interpretation.

Build transitions with split lineage:

```bash
python scripts/build_transition_dataset.py \
  --embeddings data/processed/skillcorner_td_embeddings_all.pt \
  --windows data/processed/skillcorner_windows_h2s.pt \
  --out data/processed/skillcorner_transition_dataset.pt \
  --delta-steps 2 5 10 \
  --fps 10 \
  --split-manifest splits/skillcorner_10match_inductive_v1.json \
  --scientific-mode
```

Run discovery:

```bash
python scripts/run_discovery_suite.py \
  --config configs/discovery_suite_skillcorner.yaml \
  --split-manifest splits/skillcorner_10match_inductive_v1.json \
  --scientific-mode
```

Required discovery controls:

- TD-JEPA delta representation
- raw coordinate/velocity transition features
- PCA of raw transition features
- handcrafted structure-change metrics
- random-encoder delta representation
- multiple clustering seeds
- train-fit / val-test assignment
- match concentration and transition-magnitude concentration

Render diagnostics only after control summaries are available:

```bash
python scripts/render_diagnostic_clips.py \
  --windows data/processed/skillcorner_windows_h2s.pt \
  --examples runs/discovery/experiment5_skillcorner/latent_residual_examples.csv \
  --out runs/diagnostics/blinded_clips \
  --blinded
```

Expected outputs:

- cluster summaries by split
- stability metrics across seeds
- nuisance correlations for latent residual scores
- blinded annotation directory
- separate private key file

Do not claim:

- clusters are tactical motifs because k-means produced nonempty clusters
- low correlation with ball displacement proves tactical intelligence
- annotated clips validate clusters unless annotation is blinded and compared to
  matched controls

## Go / No-Go Criteria

Proceed toward a tactical representation paper only if most are true:

1. The inductive split is used through encoder pretraining and downstream
   evaluation.
2. Correct temporal pairing clearly outperforms shuffled and wrong-match pairing.
3. Non-overlap future prediction beats identity/no-motion controls.
4. Geometry-only `z` adds value over raw kinematics on at least two nontrivial
   held-out targets.
5. Results are stable across at least three seeds.
6. Discovery clusters beat raw/PCA/random controls.
7. Cluster occupancy is not dominated by one match, missingness, or transition
   magnitude.
8. High residual examples are not mainly tracking artifacts.
9. Blinded annotation shows enrichment over matched controls.
10. Uncertainty is reported at match, possession, or segment level.

Decision rules:

- Proceed with latent-flow or language-alignment experiments only after the
  gates above pass.
- Redesign the representation if non-overlap prediction cannot beat controls.
- Expand data if results vary strongly by match.
- Keep the current model as an engineering baseline if it remains dominated by
  temporal smoothness or metadata.
