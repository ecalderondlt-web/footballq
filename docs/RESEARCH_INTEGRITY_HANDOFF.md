# Research Integrity Handoff

## Start Here

This branch is a docs-only handoff for Emilio. It organizes the audit conclusions
and the next implementation sprint. It does not implement the integrity layer yet.

Use this as the source branch for the sprint:

```text
origin/codex-phase1-baseline-pipeline
SHA: fdb04547884b1b9b9d85a38bd8fe464598d34089
```

The default `main` branch was still Phase 1 only at:

```text
fef577ecdf559d1c848fa732176f388139f1a923
```

## Scientific Decision

Do **not** proceed directly to Experiment 6 or treat current latent discovery as
validated tactical evidence.

Recommended next step:

```text
Retrain and validate representation v2 first, with split lineage, leakage
controls, non-overlapping TD-JEPA future-prediction controls, and held-out
discovery baselines.
```

Diagnostic visualization may proceed in parallel, but only as exploratory
tooling. It must not be used to validate current clusters as tactical motifs
unless the controls in the sprint plan pass.

## What Is Established

- The experiment branch implements canonical 23-entity football tracking tensors.
- The experiment branch implements TD-JEPA-like training, probes, latent flow,
  coordinate decoders, and latent transition discovery tooling.
- The experiment branch test suite passed during audit and before this handoff.
- Synthetic smoke paths can build canonical windows and TD-JEPA examples.

## What Is Not Established

- Reported real-data numerical claims are not reproducible from the checkout
  without raw data, processed windows, embeddings, checkpoints, and run outputs.
- The current TD-JEPA latent space has not demonstrated tactical understanding.
- Current k-means latent clusters have not been shown to be tactical concepts.
- Current latent residual scores have not been calibrated as tactical surprise.
- Possession and availability probe performance is leakage-affected.

## Core Issues To Fix Before Scientific Claims

### 1. Possession Probe Leakage

The current input feature list includes:

- `is_possession_team`
- `has_possession`

Possession-team and availability labels may be constructed from those same
features or from related metadata. Strong performance on those probes is a
pipeline sanity check, not evidence of learned tactical semantics.

Removing those channels only at probe time is insufficient if the encoder was
pretrained with them. A legitimate geometry-only ablation requires retraining
the encoder with a geometry-only feature view.

### 2. Current TD-JEPA Objective Is Too Easy

The legacy TD-JEPA construction is approximately:

```text
state_t:            frames 0 ... 9
state_t_plus_delta: frames 2 ... 11
delta_state:        frames 10 ... 11
```

Most target frames overlap with context, and the motion encoder directly observes
the newly introduced future frames. Low TD loss or high cosine similarity can
therefore reflect temporal smoothness and overlap rather than tactical
abstraction.

Preserve this as `legacy_shifted_overlap`, then add a separate
`future_nonoverlap_context_only` formulation.

### 3. Split Lineage Is Required

Every stage must use the same immutable match split:

- representation pretraining
- embedding export
- probes
- latent flow
- coordinate decoders
- discovery
- scaler/PCA/cluster fitting
- model selection

If an encoder is pretrained on a match that later appears in downstream test,
the downstream result is transductive. Transductive runs may exist, but they must
be labeled as such.

### 4. Sample Identity Must Include Period

Current joins can use `(match_id, frame_t)`, which is ambiguous when frame
counters reset between periods.

Use:

```text
(match_id, period, frame_t)
sample_id = "{match_id}:{period}:{frame_t}"
```

Scientific runs should reject duplicate keys and avoid silent row-order fallback.

### 5. Raw Global X Is Not Tactical Progression

Global x displacement is not consistently attacking progression across teams and
halves. Rename current raw-x targets to geometric names such as:

- `future_ball_dx_global_m`
- `future_ball_global_x_bucket`

Only create attack-relative targets when attacking direction is explicitly known
from reliable, causal metadata.

### 6. Overlapping Windows Are Not Independent

Overlapping windows can share nearly all frames. Metrics may be computed per
window, but uncertainty should be reported at a higher level such as match,
period, possession, or non-overlapping segment.

### 7. K-Means Clusters Are Not Automatically Tactical

K-means partitions continuous representations by construction. Before tactical
interpretation, compare against raw, PCA/raw, handcrafted structure metrics,
random-encoder transitions, and held-out cluster assignment.

Rename `silhouette_proxy` to `centroid_margin_proxy` unless a true silhouette
implementation is added.

### 8. Tactical Surprise Is Currently A Residual Norm

The current score is essentially:

```text
||z_next - z_predicted||
```

Call it `latent_residual_score` or `latent_prediction_residual` until it is
calibrated and controlled for confounds.

## Emilio's First Commands

```bash
git fetch --all --prune
git checkout -b codex/research-integrity-sprint-v1 origin/codex-phase1-baseline-pipeline
python -m pytest -q
python -m ruff check .
```

Expected baseline:

- pytest should pass.
- Ruff may fail with pre-existing findings until cleaned during the sprint.

## Recommended First PR

Emilio's first implementation PR should be scoped to documentation plus the split
manifest and validators:

```text
docs: establish audited research status and next-week runbook
feat: add immutable split manifests and period-aware sample identities
```

Do not start by optimizing ADE, latent flow, or architecture scale.

## Where The Detailed Plan Lives

Read:

```text
docs/EMILIO_INTEGRITY_SPRINT_PLAN.md
```

That file contains the five-day implementation sequence, tests, commands, and
go/no-go criteria.
