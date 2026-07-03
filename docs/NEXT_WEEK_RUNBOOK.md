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
  --out data/processed/skillcorner_td_jepa_nonoverlap.pt \
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

If no-motion remains close, run the same geometry-only protocol with a longer
prediction gap before scaling the model:

```bash
python scripts/prepare_td_jepa_data.py \
  --source skillcorner \
  --raw data/raw/skillcorner \
  --out data/processed/skillcorner_td_jepa_nonoverlap_gap1p0.pt \
  --objective-mode future_nonoverlap_context_only \
  --prediction-gap-seconds 1.0 \
  --feature-view geometry_only \
  --split-manifest splits/skillcorner_10match_inductive_v1.json \
  --scientific-mode

python scripts/train_td_jepa.py --config configs/td_jepa_nonoverlap_gap1p0_skillcorner.yaml
```

Current diagnostic result: the gap-1.0 one-epoch seeds improve no-motion to
caution, not pass, and do not fix player-slot or team-slot invariance. Treat this
as a redesign hint, not a paper result.
Use `scripts/compare_td_falsification_gates.py` to compare gate summaries across
gap variants before choosing the next representation change.
The next non-scaling redesign scaffold is CLS-token encoder pooling:
`configs/td_jepa_nonoverlap_gap1p0_cls_skillcorner.yaml`.
Current one-seed smoke result: CLS pooling remains blocked and does not fix
slot-control failures.
The next stronger diagnostic is optional slot-aligned target reconstruction:
`configs/td_jepa_nonoverlap_gap1p0_slot_recon_skillcorner.yaml`. This adds
slot-level pressure and must still pass falsification before any downstream
interpretation.
Current diagnostic result: slot reconstruction fixes slot-control sensitivity
under `total_loss` gating, but remains blocked by no-motion and weak context
team-label sensitivity. The next redesign should combine slot-level pressure
with a stronger future/non-smoothness objective rather than moving to
visualization.
The combined diagnostic config is
`configs/td_jepa_nonoverlap_gap1p0_slot_recon_margin_skillcorner.yaml`, which
adds a margin term requiring predicted latents to beat the no-motion latent.
Current diagnostic result: the margin term clears no-motion, but the initial
combined weights weaken context-side team/slot controls. Tune or redesign the
combined objective before running downstream probes/discovery.
The higher slot-reconstruction-weight variant
`configs/td_jepa_nonoverlap_gap1p0_slot_recon_w0p25_margin_skillcorner.yaml`
improves the tradeoff but still does not pass: no-motion remains a strong pass,
player-slot and team-swap controls are only caution, and context-side
team-label swap remains fail. Treat this as a design clue, not as a downstream
green light.
Context-side reconstruction is now available through
`loss.context_reconstruction_weight`. The equal-weight context diagnostic
`configs/td_jepa_nonoverlap_gap1p0_context_slot_recon_margin_skillcorner.yaml`
clears context-side team/slot controls but overcorrects and leaves no-motion
blocked. The lower-weight candidate
`configs/td_jepa_nonoverlap_gap1p0_context_w0p05_slot_recon_margin_skillcorner.yaml`
clears the current falsification gate across seeds 7, 11, and 23 under the
`total_loss` gate at
`runs/td_jepa/v2_nonoverlap_geometry_gap1p0_context_w0p05_slot_recon_margin_falsification_gate_extended/`.
Use this as the current candidate representation for the next gate: incremental
probe tests comparing raw versus z-scored feature views. Do not run blinded
visualization until probe and discovery-baseline gates are complete.
Current linear h2s probe result for this candidate is mixed and aggregated at
`runs/probe_suite/v2_context_w0p05_slot_recon_margin_h2s_linear_incremental_summary/`.
Future ball displacement and z-scored team-shape show consistent incremental
gains, but global-x bucket and unnormalized team-shape do not. Continue to
discovery baselines as diagnostics, but do not treat the probe gate as a paper
evidence pass.
Current discovery controls for the same candidate are aggregated at
`runs/discovery/v2_context_w0p05_slot_recon_margin_control_summary/`. All
required feature families are present, but normalized latent clusters look
similar to raw/PCA/random controls, so the combined gate at
`runs/integrity/v2_context_w0p05_slot_recon_margin_gate_summary.json` remains
blocked by probe/discovery diagnostics.
The current blinded scaffold is
`runs/diagnostics/v2_context_w0p05_slot_recon_margin_blinded_seed7_h02/`; it
separates annotator rows from the private key and now includes diagnostic GIF
media for 25 of 40 seed-7 rows. The 15 remaining rows are period-2 examples;
the currently available processed SkillCorner window tensors contain period 1
only, so those `clip_path` fields remain blank. This is still diagnostic media,
not blinded annotation evidence.

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
- team-slot swap
- home/away label swap
- target home/away label swap
- pitch reflection
- consistent player-slot permutation
- target consistent player-slot permutation

Expected comparison:

```text
correct pairing loss < shuffled future loss
correct pairing loss < wrong-match future loss
non-overlap predictor beats identity/no-motion controls
```

After running the controls for all seeds, aggregate them into an explicit gate
artifact:

```bash
python scripts/summarize_td_falsification.py \
  --summary 7:runs/td_jepa/SEED7_RUN/falsification_val/td_falsification_summary.json \
  --summary 11:runs/td_jepa/SEED11_RUN/falsification_val/td_falsification_summary.json \
  --summary 23:runs/td_jepa/SEED23_RUN/falsification_val/td_falsification_summary.json \
  --out runs/td_jepa/v2_nonoverlap_geometry_falsification_gate_extended
```

Treat `scientific_claim_status: blocked` as a no-go for tactical claims.

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

After all seeds finish, aggregate probe increments and match-level uncertainty:

```bash
python scripts/summarize_probe_incremental.py \
  --suite 7:runs/probe_suite/SEED7/results.json \
  --suite 11:runs/probe_suite/SEED11/results.json \
  --suite 23:runs/probe_suite/SEED23/results.json \
  --out runs/probe_suite/v2_nonoverlap_geometry_h2s_incremental_summary
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

After all baseline cluster summaries exist, aggregate the comparison with:

```bash
python scripts/summarize_discovery_controls.py \
  --cluster-summary normalized_delta_z:7:runs/discovery/SEED7_LATENT/delta_0p2s/cluster_summary.json \
  --cluster-summary raw_delta_z:7:runs/discovery/SEED7_BASELINES/raw_delta_z_h02/cluster_summary.json \
  --cluster-summary pca_delta_z:7:runs/discovery/SEED7_BASELINES/pca_delta_z_h02/cluster_summary.json \
  --cluster-summary random_encoder_delta_z:7:runs/discovery/SEED7_BASELINES/random_encoder_delta_z_h02/cluster_summary.json \
  --out runs/discovery/v2_nonoverlap_geometry_control_summary
```

Repeat the `--cluster-summary` entries for all seeds and feature families before
using the summary as a gate.

Render diagnostics only after control summaries are available:

```bash
python scripts/render_diagnostic_clips.py \
  --examples runs/discovery/experiment5_skillcorner/latent_residual_examples.csv \
  --out runs/diagnostics/blinded_clips \
  --max-rows 40 \
  --blinded \
  --windows data/processed/skillcorner_windows_h2s.pt \
  --clip-fps 5
```

Current-candidate example:

```bash
python scripts/render_diagnostic_clips.py \
  --examples runs/discovery/v2_context_w0p05_slot_recon_margin_seed7/normalized_delta_z_h02/latent_residual_examples.csv \
  --out runs/diagnostics/v2_context_w0p05_slot_recon_margin_blinded_seed7_h02 \
  --max-rows 40 \
  --blinded \
  --windows data/processed/skillcorner_windows_h2s.pt \
  --clip-fps 5
```

Expected outputs:

- cluster summaries by split
- stability metrics across seeds
- nuisance correlations for latent residual scores
- blinded annotation directory
- annotator CSV with only blind IDs, match/period/frame, clip paths, and blank
  annotation cells
- separate private key file
- media coverage count; blank `clip_path` values must be explained by missing
  matched tracking windows, not silently ignored

Do not claim:

- clusters are tactical motifs because k-means produced nonempty clusters
- low correlation with ball displacement proves tactical intelligence
- annotated clips validate clusters unless annotation is blinded and compared to
  matched controls

## Go / No-Go Criteria

After falsification, probe, and discovery summaries exist, write the combined
gate status:

```bash
python scripts/summarize_integrity_gates.py \
  --falsification runs/td_jepa/v2_nonoverlap_geometry_falsification_gate_extended/td_falsification_gate_summary.json \
  --probe runs/probe_suite/v2_nonoverlap_geometry_h2s_incremental_summary/probe_incremental_summary.json \
  --discovery runs/discovery/v2_nonoverlap_geometry_control_summary/discovery_control_summary.json \
  --out runs/integrity/v2_nonoverlap_geometry_gate_summary_extended.json
```

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
