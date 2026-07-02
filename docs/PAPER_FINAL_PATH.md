# Paper Final Path

This document defines the shortest defensible path from the current footballq
research stack to a paper-quality result. The goal is not to maximize model
complexity. The goal is to make a claim that survives leakage checks, held-out
evaluation, and independent review.

## Current Position

- Branch under review: `codex/research-integrity-sprint-v1`
- Current reviewed commit: `f4d250714fce942703c4342eda8cc39be973d759`
- Base experiment stack: `origin/codex-phase1-baseline-pipeline`
- Core warning: current latent clusters, possession probes, and latent residual
  scores are not yet validated tactical evidence.

The existing codebase is useful as research infrastructure. It is not yet a
scientific result that can support tactical-representation claims.

## Paper Objective

The paper should answer one narrow question:

> Can a self-supervised representation of football tracking windows learn
> held-out, non-leaked structure that adds value beyond raw kinematics and
> produces stable, interpretable transition groups under controls?

This question can support three possible paper outcomes:

1. Positive representation result: geometry-only non-overlap TD-JEPA adds
   held-out value beyond raw kinematics, passes falsification controls, and
   discovery clusters are stable and tactically enriched under blinded review.
2. Negative scientific result with reusable infrastructure: overlapping TD-JEPA
   and metadata-rich probes look strong, but controls show the result is driven
   by temporal smoothness, leakage, or raw kinematics.
3. Methods/resource paper: the main contribution is a reproducible football
   tracking research protocol with split lineage, leakage controls, and
   diagnostic tooling.

Do not decide which paper it is before running the controls.

## What Is Done

The current branch contains a first pass of research-integrity infrastructure:

- Immutable ten-match split manifest:
  `splits/skillcorner_10match_inductive_v1.json`
- Split validation and split hash helpers:
  `src/footballq/repro/splits.py`
- Period-aware sample identity helpers:
  `src/footballq/repro/identity.py`
- Feature views:
  `src/footballq/repro/feature_views.py`
- Run-manifest utility:
  `src/footballq/repro/manifest.py`
- Run-manifest writing for scientific TD-JEPA preparation/training/export,
  probe, latent-rollout, decoder, transition, and discovery entry points
- TD-JEPA objective modes:
  `legacy_shifted_overlap`
  and `future_nonoverlap_context_only`
- TD falsification transforms:
  `src/footballq/repro/falsification.py`
- Probe validity metadata and leaked-probe classification
- Global-x progression target rename scaffolding
- Period/sample-id propagation into windows, embeddings, probes, decoder data,
  latent rollout data, and transition datasets
- Train-fit / held-out assignment scaffolding for discovery clustering
- Latent residual naming and nuisance-correlation diagnostics
- Default residual diagnostic files:
  `latent_residual_examples.csv` and `latent_residual_summary.json`
- Blinded diagnostic rendering scaffold
- Focused integrity tests

## What Is Not Done

These items block paper-quality claims:

- Full repo lint is not clean.
- Full test collection depends on a complete local dev environment.
- Discovery controls now have a first raw/PCA/random/handcrafted multi-seed
  summary, but latent-delta discovery has not separated itself from those
  controls or shown blinded enrichment.
- Documentation still needs to be rechecked against a fresh install and real
  SkillCorner availability.
- Team-shape labels remain all-visible-player stretch diagnostics, not a clean
  team tactical target.
- The real SkillCorner ten-match split has been located locally, but processed
  artifact coverage and alignment still need final audit.
- Representation v2 has only been retrained as one-epoch diagnostic seeds under
  the non-overlap, geometry-only protocol.
- Falsification controls have a three-seed gate summary, and the gate remains
  blocked by near-invariance to team swap and consistent player-slot
  permutation plus a close no-motion control.
- A combined integrity gate summary exists for the current v2 diagnostics, and
  its overall claim status is blocked.
- No blinded annotation evidence exists yet.

## Final Path

### Phase 0: Make The Branch Reproducible

Goal: a new collaborator can install, run tests, and understand which claims are
allowed.

Required actions:

1. Install the project in a fresh Python 3.11 or 3.12 environment:

   ```bash
   python -m pip install -e ".[dev]"
   ```

2. Run the documented checks:

   ```bash
   python -m pytest -q
   python -m ruff check .
   ```

3. If repo-wide Ruff cleanup is deferred, document the exact remaining count and
   files. Do not call the repo lint-clean.

4. Update README commands so scientific examples include:

   ```bash
   --split-manifest splits/skillcorner_10match_inductive_v1.json
   --scientific-mode
   ```

5. Make every scientific script write a run manifest or explicitly document why
   it does not.

Acceptance criteria:

- Full test suite passes in Emilio's environment.
- Either `python -m ruff check .` passes or a tracked lint-debt exception list is
  present.
- README, `docs/NEXT_WEEK_RUNBOOK.md`, and this document agree on commands.

### Phase 1: Verify Data And Split Lineage

Goal: every experimental stage uses the same match split and no test match enters
pretraining or preprocessing fit.

Required actions:

1. Verify the ten reported SkillCorner match IDs exist locally.
2. Validate the split:

   ```bash
   python - <<'PY'
   from footballq.repro.splits import load_split_manifest
   split = load_split_manifest("splits/skillcorner_10match_inductive_v1.json")
   print(split.sha256)
   print(split.train_match_ids)
   print(split.val_match_ids)
   print(split.test_match_ids)
   PY
   ```

3. Add a dataset-name allow-list to split validation.
4. Record the split hash in every generated artifact.
5. Reject scientific runs that lack period-aware `sample_id` fields.

Acceptance criteria:

- The same split hash appears in TD data, checkpoints, embeddings, probes,
  decoder data, latent rollout data, transition data, and reports.
- No scientific command falls back to random or index-order alignment.

### Phase 2: Train Representation V2

Goal: separate legacy transition consistency from real future prediction.

Run at least two representation protocols:

1. Legacy baseline:

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

2. Primary paper candidate:

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

3. Train at least three seeds for the primary candidate.

Required diagnostics:

- train and validation learning curves
- embedding variance and effective rank
- correct vs shuffled future loss
- correct vs wrong-match future loss
- reversed-time control
- masked-ball control
- team-swap and pitch-reflection controls
- player-slot permutation sensitivity

Acceptance criteria:

- Correct temporal pairing must beat shuffled and wrong-match pairings on
  validation matches.
- Non-overlap future prediction must beat identity/no-motion controls.
- Results must be stable across at least three seeds.

### Phase 3: Probes With Incremental Value

Goal: show whether `z` adds held-out information beyond raw kinematics.

For every valid probe, report:

```text
performance(raw)
performance(z)
performance(raw + z)
incremental_value = performance(raw + z) - performance(raw)
```

For lower-is-better regression metrics, define signed improvement explicitly.

Probe policy:

- Possession and availability probes on `full_state_legacy` embeddings are
  leakage sanity checks only.
- `future_ball_global_x_bucket` is a geometry target, not tactical progression.
- Attack-relative progression is unavailable unless causal attacking direction is
  stored per sample.
- Team-shape labels must state whether they are home, away, possession-team,
  defending-team, or all-player diagnostics.

Paper-relevant probe candidates:

- turnover within horizon
- line-breaking progression
- penalty-area entry
- switch of play
- transition vs settled possession
- future threat increase
- defensive block compression/expansion
- event macro-category

Acceptance criteria:

- Geometry-only `z` adds value over raw kinematics on at least two nontrivial
  held-out targets.
- Report match-level or possession-level uncertainty, not only row-level scores.

### Phase 4: Discovery Controls

Goal: determine whether clusters are tactical motifs or just partitions of
smooth motion, magnitude, provider artifacts, or match identity.

Required feature families:

- TD-JEPA delta representation
- raw coordinate/velocity transition features
- PCA of raw transition features
- handcrafted structure-change metrics
- random-encoder delta representation

Required protocol:

- Fit scaler/PCA/clusterers on train matches only.
- Assign validation and test transitions without refitting.
- Run multiple clustering seeds.
- Report cluster-size distribution, match concentration, transition-magnitude
  concentration, held-out occupancy, and seed stability.

Acceptance criteria:

- TD-JEPA clusters are more stable and more enriched than raw/PCA/random
  controls.
- No key result is dominated by one match, missingness pattern, or transition
  magnitude.

### Phase 5: Diagnostic Visualization And Annotation

Goal: use visualization as evidence only after controls justify looking.

Generate:

- top-down frame strips
- cluster contact sheets
- high-residual contact sheets
- matched low-residual controls
- blinded annotation folders
- annotation CSV templates
- private key files linking anonymous clip IDs to metadata

Blinding rules:

- Annotator-facing files must not show cluster IDs.
- Annotator-facing files must not show residual scores.
- Annotator-facing files must not reveal positive/control status.
- The key file stays outside the annotator directory.

Acceptance criteria:

- Blinded human annotation finds reproducible tactical enrichment over matched
  controls.
- High-residual examples are not mainly tracking jumps, missing-ball frames, or
  extreme acceleration.

### Phase 6: Paper Decision

Use this decision table:

| Evidence | Decision |
| --- | --- |
| Passes falsification, incremental probes, discovery stability, and blinded annotation | Proceed with tactical representation paper |
| Non-overlap model cannot beat controls or adds no value over raw geometry | Write negative result / redesign representation |
| Results vary strongly by match | Expand data before making tactical claims |
| Current model remains dominated by temporal smoothness or leaked metadata | Keep as engineering baseline only |

## Claims Allowed Now

- The repository contains infrastructure for football tracking representation
  experiments.
- The old overlapping TD-JEPA formulation is an engineering baseline.
- Possession and availability probes can serve as pipeline sanity checks.
- Latent residual scores are diagnostics for prediction residuals.
- Current clusters are exploratory partitions requiring controls and annotation.

## Claims Not Allowed Yet

- The latent clusters are tactical concepts.
- The latent residual score is tactical surprise.
- Possession probe performance proves semantic understanding.
- Low TD-JEPA loss proves tactical abstraction.
- Ten-match overlapping windows provide independent sample sizes.
- Raw global x displacement is attacking progression.

## Immediate Message To Emilio

Emilio should first make the integrity branch reproducible, then retrain the
geometry-only non-overlap representation, then run falsification and incremental
probe controls. Visualization is useful after those gates, but it should be
blinded diagnostic evidence, not the next scientific claim.
