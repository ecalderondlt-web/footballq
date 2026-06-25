# Emilio Integrity Sprint Implementation Plan

## Purpose

This plan turns the audit conclusions into an executable engineering sprint.
The goal is to make future footballq experiments reproducible, leakage-aware,
and scientifically defensible before larger model work.

## Ownership

Emilio should implement this sprint directly, using Codex or GPT Pro as a
reviewer at checkpoints. The work changes scientific protocol and experiment
semantics, so every design choice should be understood before results are used.

## Commit Strategy

Use small reviewable commits:

1. `docs: establish audited research status and next-week runbook`
2. `feat: add immutable split manifests and period-aware sample identities`
3. `feat: add experiment provenance and leakage-controlled feature views`
4. `feat: add TD-JEPA falsification and discovery controls`
5. `test: expand scientific-integrity checks and clean lint`

## Day 1 - Documentation And Split Foundation

### Goal

Make the repository honest about what is established, what is unproven, and how
future experiments must be split.

### Add Or Update

- `AGENTS.md`
- `README.md`
- `docs/RESEARCH_STATUS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/NEXT_WEEK_RUNBOOK.md`
- `docs/EMILIO_RESEARCH_NOTES.md`
- `splits/skillcorner_10match_inductive_v1.json`
- `src/footballq/repro/splits.py`
- `tests/test_split_manifest.py`

### Split Manifest

Use the reported ten-match split, clearly marked as unverified until local files
are checked:

```json
{
  "name": "skillcorner_10match_inductive_v1",
  "version": 1,
  "dataset": "skillcorner",
  "protocol": "inductive",
  "train_match_ids": ["1886347", "1899585", "1925299", "1953632", "1996435", "2011166"],
  "val_match_ids": ["2006229", "2017461"],
  "test_match_ids": ["2013725", "2015213"],
  "all_match_ids": ["1886347", "1899585", "1925299", "1953632", "1996435", "2006229", "2011166", "2013725", "2015213", "2017461"],
  "source": "progress_report_unverified",
  "seed": null,
  "creation_timestamp_utc": "2026-06-25T00:00:00Z",
  "expected_count": 10,
  "notes": "Match IDs came from the progress report and must be verified against local SkillCorner files before scientific claims."
}
```

### Required Tests

- no train/val/test overlap
- duplicate match IDs fail validation
- `all_match_ids` equals the union of split sets
- scientific runs with fewer than three distinct matches fail
- split manifest SHA-256 is stable

### Commands

```bash
python -m pytest tests/test_split_manifest.py -q
python -m pytest -q
```

## Day 2 - Period-Aware Sample Identity And Provenance

### Goal

Make every artifact traceable and every sample alignment period-aware.

### Add Or Update

- `src/footballq/repro/identity.py`
- `src/footballq/repro/manifest.py`
- `src/footballq/data/windows.py`
- `src/footballq/data/td_jepa_dataset.py`
- `src/footballq/training/export_td_embeddings.py`
- `src/footballq/probes/dataset.py`
- `src/footballq/latent_flow/dataset.py`
- `src/footballq/decoding/dataset.py`
- `src/footballq/discovery/transitions.py`
- `tests/test_sample_identity.py`
- `tests/test_run_manifest.py`

### Required Behavior

Canonical key:

```text
(match_id, period, frame_t)
sample_id = "{match_id}:{period}:{frame_t}"
```

Scientific alignment should:

- assert uniqueness
- fail on duplicates
- reject cross-period transition pairs
- avoid row-order fallback unless `allow_legacy_alignment=True`
- record legacy alignment in metadata when used

Run manifests should record:

- UTC timestamp
- remote URL
- branch and commit SHA
- dirty status
- exact command
- config path and hash
- split manifest path and hash
- evaluation protocol
- feature view
- dataset paths, sizes, and hashes where practical
- match IDs and counts by split
- Python and dependency versions
- device
- output paths and hashes
- warnings

### Required Tests

- period 1 frame 10 and period 2 frame 10 are different sample IDs
- duplicate `(match_id, period, frame_t)` fails
- period-preserving joins work
- cross-period transitions fail
- run manifest validates required fields
- split-manifest hash propagates into downstream metadata

## Day 3 - Feature Views, Probe Validity, And Label Semantics

### Goal

Separate leakage sanity checks from semantic-candidate probes and stop calling
global x displacement tactical progression.

### Add Or Update

- `src/footballq/repro/feature_views.py`
- `src/footballq/probes/labels.py`
- `src/footballq/probes/features.py`
- `src/footballq/probes/dataset.py`
- `src/footballq/probes/training.py`
- `scripts/build_probe_dataset.py`
- `scripts/run_probe_suite.py`
- `tests/test_feature_views.py`
- `tests/test_probe_validity.py`
- `tests/test_label_semantics.py`

### Feature Views

Support:

- `full_state_legacy`
- `geometry_only`
- `missingness_only_control`
- `raw_kinematics_control`

`geometry_only` should include:

- `x_norm`
- `y_norm`
- `vx_norm`
- `vy_norm`
- `is_ball`
- `is_home`
- `is_away`

It should exclude possession channels and should not use `visible_mask` as an
ordinary content feature.

### Probe Validity Classes

Every probe should expose:

```text
validity_class = semantic_candidate
validity_class = geometry_control
validity_class = leakage_sanity_check
validity_class = unavailable
```

Possession-team and availability probes are leakage sanity checks for full-state
embeddings.

### Label Renames

Rename raw global-x targets:

- `future_ball_dx_global_m`
- `future_ball_global_x_bucket`

Only add attack-relative progression when attacking direction is known from
reliable causal metadata:

- `future_ball_progression_attacking_m`
- `future_ball_progression_attacking_bucket`

Separate all-player shape metrics from home, away, possession-team, and
defending-team shape metrics.

### Required Tests

- geometry-only excludes possession channels
- feature-name mismatches fail clearly
- missingness-only controls run
- leaked probes are classified as leakage sanity checks
- attack-relative progression is unavailable when direction is unknown
- global x displacement keeps geometric naming

## Day 4 - TD-JEPA Legacy Labeling, Non-Overlap V2, And Controls

### Goal

Preserve the old overlapping objective as legacy while adding a true
future-prediction formulation.

### Add Or Update

- `src/footballq/data/td_jepa_dataset.py`
- `src/footballq/models/td_jepa.py`
- `src/footballq/training/train_td_jepa.py`
- `scripts/prepare_td_jepa_data.py`
- `configs/td_jepa_nonoverlap_synthetic.yaml`
- `configs/td_jepa_nonoverlap_skillcorner.yaml`
- `src/footballq/repro/falsification.py`
- `tests/test_td_jepa_nonoverlap.py`
- `tests/test_falsification_controls.py`
- `tests/test_player_slot_sensitivity.py`

### Objective Modes

1. `legacy_shifted_overlap`
   - reproduces current behavior
   - explicitly marked legacy

2. `future_nonoverlap_context_only`
   - context contains only information available at or before time `t`
   - target starts after `t + gap`
   - context and target share no frames
   - predictor receives context only
   - no target future chunk is passed to the predictor
   - target encoder remains stop-gradient / EMA
   - temporal indices are saved

### Falsification Controls

Add controls for:

- correct temporal pairing
- shuffled future within batch
- future from another match
- reversed-time context
- no-motion/no-future predictor
- masked ball
- team swap
- pitch reflection
- consistent player-slot permutation
- longer prediction gap

### Player-Slot Diagnostic

Consistently permute player slots within each team across all context frames and
report:

- cosine similarity between original/permuted latent
- L2 difference
- downstream prediction change if practical

Do not independently shuffle each frame.

### Required Tests

- non-overlap has zero shared time indices
- context-only predictor receives no target frames
- wrong-match controls use another match
- reversal, reflection, and permutation preserve tensor validity
- metadata records control condition
- player-slot permutation preserves temporal continuity

## Day 5 - Discovery Controls, Residual Score Naming, Visualization Scaffold

### Goal

Make discovery train-fit/held-out-assignment aware and prevent tactical-surprise
overclaiming.

### Add Or Update

- `src/footballq/discovery/transitions.py`
- `src/footballq/discovery/clustering.py`
- `src/footballq/discovery/surprise.py`
- `src/footballq/discovery/report.py`
- `src/footballq/discovery/controls.py`
- `scripts/run_discovery_suite.py`
- `scripts/render_diagnostic_clips.py`
- `tests/test_discovery_controls.py`
- `tests/test_residual_score.py`
- `tests/test_blinded_rendering.py`

### Discovery Requirements

Scientific runs must:

- fit normalization/PCA/clusterers on training matches only
- assign validation/test without refitting
- report results separately by split
- run multiple seeds
- report stability
- report match concentration per cluster
- report transition-magnitude concentration per cluster

Feature families:

- TD-JEPA delta representation
- raw coordinate/velocity transitions
- PCA of raw transition features
- handcrafted structure-change metrics
- random-encoder delta representation

Separate:

```text
magnitude = ||delta_z||
direction = delta_z / (||delta_z|| + epsilon)
```

Rename `silhouette_proxy` to `centroid_margin_proxy` unless true silhouette is
implemented.

### Residual Score

Rename default “tactical surprise” output:

- `latent_residual_score`
- or `latent_prediction_residual`

Keep `tactical_surprise` only as a deprecated alias or future hypothesis.

Report nuisance correlations with:

- ball speed
- player speed
- acceleration
- missing-player count
- ball visibility
- pitch region
- match identity
- possession change
- raw transition magnitude

### Visualization Scaffold

Visualization is diagnostic, not validation. Add commands for:

- top-down frame strips
- cluster contact sheets
- high-residual contact sheets
- matched low-residual controls
- annotation CSV templates
- blinded annotation mode

Blinded outputs must hide cluster IDs, residual scores, and positive/control
status. Keep a separate key file.

### Required Tests

- train-fit/test-assign works
- seed stability is measurable
- magnitude-only synthetic clusters are detected
- match-specific artifacts are visible
- residual-score metadata uses the new name
- blinded renderer separates annotator files from key files

## Final Verification Before PR

Run:

```bash
python -m pytest -q
python -m ruff check .
python -c "import footballq; from footballq.data.td_jepa_dataset import build_td_jepa_examples; from footballq.discovery.surprise import compute_surprise; print('imports-ok')"
```

Synthetic smoke:

```bash
python scripts/make_synthetic_data.py --out /tmp/footballq_sprint_tracking.csv --num-matches 3 --num-frames 120 --fps 10 --seed 7
python scripts/prepare_tracking_data.py --source synthetic --raw /tmp/footballq_sprint_tracking.csv --out /tmp/footballq_sprint_windows.pt --fps-out 10 --context-seconds 1.0 --horizon-seconds 1.0 --stride-seconds 0.2
python scripts/prepare_td_jepa_data.py --source synthetic --raw /tmp/footballq_sprint_tracking.csv --out /tmp/footballq_sprint_td_legacy.pt --fps-out 10 --context-seconds 1.0 --delta-seconds 0.2 --stride-seconds 0.2 --objective-mode legacy_shifted_overlap
python scripts/prepare_td_jepa_data.py --source synthetic --raw /tmp/footballq_sprint_tracking.csv --out /tmp/footballq_sprint_td_nonoverlap.pt --fps-out 10 --context-seconds 1.0 --delta-seconds 0.2 --stride-seconds 0.2 --objective-mode future_nonoverlap_context_only --prediction-gap-seconds 0.5
```

If SkillCorner data is available locally, run but do not commit outputs:

```bash
python scripts/report_skillcorner_availability.py --raw data/raw/skillcorner --processed-dir data/processed --embeddings data/processed/skillcorner_td_embeddings_all.pt
python scripts/prepare_tracking_data.py --source skillcorner --raw data/raw/skillcorner --out data/processed/skillcorner_windows.pt --fps-out 10 --context-seconds 2.0 --horizon-seconds 2.0 --stride-seconds 0.2
python scripts/prepare_td_jepa_data.py --source skillcorner --raw data/raw/skillcorner --out data/processed/skillcorner_td_jepa_nonoverlap.pt --fps-out 10 --context-seconds 1.0 --delta-seconds 0.2 --stride-seconds 0.2 --objective-mode future_nonoverlap_context_only --prediction-gap-seconds 0.5 --split-manifest splits/skillcorner_10match_inductive_v1.json
```

## Go/No-Go Criteria

Do not advance to larger flow-matching or language-alignment experiments until
most are satisfied:

1. Inductive split is used consistently through encoder pretraining and
   downstream evaluation.
2. Correct temporal pairing clearly outperforms shuffled and wrong-match pairing.
3. Non-overlapping future prediction learns beyond identity/no-motion controls.
4. Geometry-only `z` adds value over raw kinematics on at least two nontrivial
   held-out targets.
5. Results are stable across at least three seeds.
6. Discovery clusters are more stable and tactically enriched than raw/PCA/random
   controls.
7. Cluster occupancy is not dominated by one match, missingness pattern, or
   transition magnitude.
8. High latent-residual examples are not primarily tracking jumps, missing-ball
   frames, or extreme acceleration.
9. Blinded annotation finds reproducible tactical enrichment over matched
   controls.
10. Results include match-level or possession-level uncertainty estimates.

## Recommendation

Retrain representation v2 first, using the non-overlap future-prediction
formulation and leakage-controlled feature views. Diagnostic visualization can
proceed in parallel only as exploratory tooling.
