# Agent Review Handoff

This file is for the next agent or reviewer who inspects the repository after
the integrity sprint changes.

## Repository State Reviewed

- Repository: `ecalderondlt-web/footballq`
- Branch: `codex/research-integrity-sprint-v1`
- Commit reviewed: `f4d250714fce942703c4342eda8cc39be973d759`
- Local worktree used for review: `/private/tmp/footballq_integrity_sprint`

## Follow-Up Status

After the original review, the branch was advanced to `295d9e6` and then updated
with reproducibility cleanup: scientific paper-path entry points now write run
manifests, README commands use split/scientific flags, split validation rejects
unknown dataset names, and default residual diagnostic files use
`latent_residual_*` names. The scientific gates remain unchanged: verify local
SkillCorner data, retrain geometry-only non-overlap representation v2, run
falsification/probe/discovery controls, and use blinded visualization only after
those controls.

Subsequent local diagnostics verified the ten SkillCorner split folders, trained
three one-epoch geometry-only non-overlap v2 seeds, exported embeddings, ran
validation falsification controls, ran h2s incremental probes, ran normalized
latent-delta discovery for three seeds, ran 0.2s discovery baselines, and
generated a blinded annotation scaffold. These outputs are diagnostic only:
no-motion remains close to learned prediction, home/away label-swap controls
separate from correct pairing, and team-slot/player-slot controls remain
near-invariant.

Continuation status on the local `codex/research-integrity-sprint-v1` branch:
the v2 geometry-only non-overlap diagnostic was extended with slot-aligned target
reconstruction, a no-motion margin, and optional context-side reconstruction.
The first context-side reconstruction diagnostic overcorrected and remained
blocked by no-motion, but the lower-weight context diagnostic at
`configs/td_jepa_nonoverlap_gap1p0_context_w0p05_slot_recon_margin_skillcorner.yaml`
cleared the current falsification gate across seeds 7, 11, and 23 under the
`total_loss` gate. The aggregate is stored at
`runs/td_jepa/v2_nonoverlap_geometry_gap1p0_context_w0p05_slot_recon_margin_falsification_gate_extended/`
with `scientific_claim_status: controls_passed`. This is not a paper result:
linear h2s incremental probes were run next and are mixed. The aggregate at
`runs/probe_suite/v2_context_w0p05_slot_recon_margin_h2s_linear_incremental_summary/`
shows consistent gains for future ball displacement and z-scored team-shape, but
negative raw-plus-`z` deltas for global-x bucket and unnormalized team-shape.
Discovery baselines against raw/PCA/random and handcrafted controls were then
run for the same candidate and aggregated at
`runs/discovery/v2_context_w0p05_slot_recon_margin_control_summary/`. The
combined gate at `runs/integrity/v2_context_w0p05_slot_recon_margin_gate_summary.json`
remains blocked by `probe_incremental` and `discovery_controls`, while
falsification is `controls_passed`. A blinded annotation scaffold was generated
at `runs/diagnostics/v2_context_w0p05_slot_recon_margin_blinded_seed7_h02/`;
it separates annotator rows from the private key. A follow-up diagnostic render
initially filled `clip_path` values for the 25 rows whose identities match
`data/processed/skillcorner_windows_h2s.pt`. Targeted full-period h2s per-match
caches then filled the remaining period-2 rows. The renderer now writes
`render_manifest.json` with source paths, rendered/reused/missing counts,
`missing_windows=0`, and `claim_status: diagnostic_only`. This is complete
diagnostic media only; no completed blinded annotation evidence exists yet.
`scripts/validate_blinded_annotation_package.py` now checks annotator/key/manifest
consistency, hidden-field separation, clip existence, and blank annotation cells;
both current local diagnostic packages pass with 40 rows and 40 clip paths.
`scripts/analyze_blinded_annotations.py` now provides the post-review summary
path; the current balanced package correctly reports
`annotation_status: incomplete` because no annotation cells are filled. The
combined integrity gate can optionally include this annotation summary, and the
current with-annotation gate remains blocked by probe, discovery, and blinded
annotation.
`docs/BLINDED_ANNOTATION_GUIDE.md` now gives annotators exact allowed labels
without exposing private fields. The analyzer rejects filled rows outside that
controlled vocabulary with `annotation_status: invalid_labels`, and the
combined gate treats that as blocked.
The SkillCorner availability report now exposes raw-vs-processed period
coverage. Current local raw tracking reports `raw_periods=1,2` for all ten
matches, while current h2s output reports `periods=1`, `window_count=315400`,
and `missing_processed_periods=2` for every match. The period-2 media gap is
therefore a processed-artifact coverage/provenance issue, not missing raw
period-2 SkillCorner data. `scripts/prepare_tracking_horizons.py --resume` now
checks cached per-match chunks against raw match periods, so the current
period-1-only cache files are treated as stale and regenerated rather than
silently reused. The same script now supports targeted `--match-ids` rebuilds
with `--skip-combine`, and `scripts/render_diagnostic_clips.py` accepts multiple
window files or globs; the local period-2 diagnostic media gap has been filled
from per-match h2s caches without writing a giant combined tensor.
A balanced diagnostic scaffold was also generated at
`runs/diagnostics/v2_context_w0p05_slot_recon_margin_blinded_balanced_seed7_h02/`
with 20 high-residual rows and 20 hidden low-residual controls. The private key
stores row status and control-match metadata; the annotator CSV does not.

## Review Purpose

The project is being prepared for a paper. The reviewer should evaluate whether
the code can support defensible scientific claims, not only whether scripts run.

The central question is whether TD-JEPA-style tracking representations learn
held-out tactical or strategic structure beyond:

- direct input metadata leakage
- temporal smoothness
- raw kinematics
- missingness/provider artifacts
- match identity
- transition magnitude

## What Changed In Emilio's Latest Commit

The latest commit added or modified 58 files with roughly 2,485 insertions and
240 deletions. The change is a broad integrity scaffold, not a finished paper
pipeline.

Major additions:

- `splits/skillcorner_10match_inductive_v1.json`
- `src/footballq/repro/splits.py`
- `src/footballq/repro/identity.py`
- `src/footballq/repro/feature_views.py`
- `src/footballq/repro/manifest.py`
- `src/footballq/repro/falsification.py`
- `configs/td_jepa_nonoverlap_synthetic.yaml`
- `configs/td_jepa_nonoverlap_skillcorner.yaml`
- `scripts/render_diagnostic_clips.py`
- integrity tests under `tests/test_*`

Major modified areas:

- tracking windows now carry period-aware metadata
- TD-JEPA data construction supports legacy overlap and future non-overlap modes
- embedding export carries split and feature-view metadata
- probe datasets use period-aware alignment and classify target validity
- decoder, latent-flow, and transition datasets accept split manifests
- discovery clustering has train-fit / held-out assignment scaffolding
- latent residual diagnostics partially replace tactical-surprise language

## Verification Already Run

Commands run during review:

```bash
git fetch --all --prune
git status --short --branch
git rev-parse HEAD origin/codex/research-integrity-sprint-v1
PYTHONPATH=src python3.12 -m pytest -q --ignore=tests/test_synthetic_demo.py --maxfail=3
PYTHONPATH=src python3.12 -m pytest -q tests/test_split_manifest.py tests/test_sample_identity.py tests/test_feature_views.py tests/test_label_semantics.py tests/test_probe_validity.py tests/test_td_jepa_nonoverlap.py tests/test_falsification_controls.py tests/test_discovery_controls.py tests/test_residual_score.py tests/test_run_manifest.py tests/test_blinded_rendering.py tests/test_scientific_invariants.py
PYTHONPATH=src python3 -m ruff check . --statistics
```

Observed results:

- Latest remote branch and local branch both pointed to
  `f4d250714fce942703c4342eda8cc39be973d759`.
- `PYTHONPATH=src python3.12 -m pytest -q --ignore=tests/test_synthetic_demo.py --maxfail=3`
  reported `123 passed, 3 warnings`.
- Focused integrity tests reported `23 passed`.
- Full `python3.12 -m pytest -q` did not collect in this local environment
  because `typer` was missing.
- `python3 -m ruff check . --statistics` reported 48 errors.

Environment note:

- Python 3.14 in the local environment had dependency issues around `pyarrow`.
- Python 3.12 had pandas, yaml, and torch, but lacked typer and ruff.
- Python 3 had ruff, but lacked typer.
- Emilio should verify in a clean environment with `python -m pip install -e ".[dev]"`.

## High-Priority Review Findings

### 1. Provenance utility exists but is not fully integrated

Evidence:

- `src/footballq/repro/manifest.py` defines `build_run_manifest`.
- `src/footballq/training/train_td_jepa.py` writes config/checkpoints/metrics
  but does not write a run manifest.

Scientific impact:

- Later artifacts can still lose command/config/device/split lineage.

Reviewer task:

- Check every experimental entry point for a `run_manifest.json` or explicit
  documented exception.

### 2. Documentation is honest but incomplete

Evidence:

- `docs/NEXT_WEEK_RUNBOOK.md` was only a short checklist before this handoff.
- `docs/EMILIO_RESEARCH_NOTES.md` was only a short warning note.

Scientific impact:

- Emilio or another reviewer could run legacy/non-scientific commands by mistake.

Reviewer task:

- Confirm `docs/PAPER_FINAL_PATH.md`, `docs/NEXT_WEEK_RUNBOOK.md`, and README
  agree on the scientific command path.

### 3. Discovery controls remain incomplete

Evidence:

- `src/footballq/discovery/clustering.py` supports only `raw_delta_z`,
  `normalized_delta_z`, and `z_t_delta_z`.
- Cluster summary records `single_seed_result`.

Scientific impact:

- Current discovery cannot yet establish tactical motifs over raw/PCA/random
  baselines.

Reviewer task:

- Require raw transition, PCA raw transition, handcrafted, and random encoder
  controls before any tactical cluster interpretation.

### 4. README had stale scientific commands in the original review

Original evidence:

- The Experiment 5 README commands build transitions and run discovery without
  `--split-manifest` or `--scientific-mode`.
- The README still references `analyze_tactical_surprise.py`.

Scientific impact:

- A collaborator could generate outputs that look scientific but lack split
  lineage.

Follow-up status:

- README paper-path commands now include split/scientific flags for TD-JEPA,
  probe, latent rollout, decoder, transition, and discovery builders.

Remaining reviewer task:

- Update or flag every README command that can produce paper-relevant artifacts.

### 5. Split validation misses dataset-name rejection

Evidence:

- `src/footballq/repro/splits.py` validates overlap, duplicates, union, count,
  and minimum match count.
- It does not reject unknown dataset names.

Scientific impact:

- Bad or mislabeled split manifests can be accepted.

Reviewer task:

- Add a dataset allow-list or schema validation.

### 6. Residual-score rename was partial in the original review

Original evidence:

- `src/footballq/discovery/surprise.py` now computes `latent_residual_*`.
- Output files and compatibility fields used `surprise_examples.csv`,
  `surprise_summary.json`, and `surprise_score`.

Scientific impact:

- Old tactical-surprise language can leak back into reports.

Follow-up status:

- Default residual outputs now use `latent_residual_examples.csv` and
  `latent_residual_summary.json`; deprecated aliases remain only through
  compatibility APIs.

Remaining reviewer task:

- Rename default artifact filenames and report fields, keeping old aliases only
  behind explicit compatibility labels.

## Files To Inspect First

- `docs/PAPER_FINAL_PATH.md`
- `docs/NEXT_WEEK_RUNBOOK.md`
- `docs/RESEARCH_STATUS.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `splits/skillcorner_10match_inductive_v1.json`
- `src/footballq/repro/splits.py`
- `src/footballq/repro/identity.py`
- `src/footballq/repro/feature_views.py`
- `src/footballq/repro/manifest.py`
- `src/footballq/repro/falsification.py`
- `src/footballq/data/td_jepa_dataset.py`
- `src/footballq/training/train_td_jepa.py`
- `src/footballq/training/export_td_embeddings.py`
- `src/footballq/probes/dataset.py`
- `src/footballq/probes/labels.py`
- `src/footballq/discovery/transitions.py`
- `src/footballq/discovery/clustering.py`
- `src/footballq/discovery/surprise.py`

## Reviewer Checklist

- [ ] Fresh install succeeds with `python -m pip install -e ".[dev]"`.
- [ ] Full `python -m pytest -q` passes.
- [x] Repo-wide Ruff is clean or a tracked lint-debt exception is explicit.
- [ ] README scientific commands include split manifests.
- [ ] Every real-data scientific command requires or records split lineage.
- [ ] Run manifests are written from every experimental entry point.
- [ ] Period-aware sample IDs are required in scientific mode.
- [ ] Legacy alignment is impossible in scientific mode.
- [ ] Feature-view metadata is recorded in checkpoints and embeddings.
- [ ] Geometry-only embeddings cannot access possession channels.
- [ ] Possession probes are marked leakage sanity checks.
- [ ] Raw global-x targets are not called attacking progression.
- [ ] TD non-overlap context and target have zero shared frames.
- [ ] Falsification controls are actually evaluated, not only transformable.
- [ ] Discovery fits scalers/PCA/clusterers on train matches only.
- [ ] Discovery includes raw/PCA/random/handcrafted controls.
- [ ] Residual diagnostics report nuisance correlations by match.
- [ ] Blinded visualization keeps annotation files separate from key files and
      records media coverage/missing-window limitations.
- [ ] Paper claims are limited to the evidence actually produced.

## Recommended Next Decision

Do not proceed directly to a tactical-discovery paper claim. Proceed with the
paper path only after:

1. The integrity documentation and README command paths are aligned.
2. Run manifests are integrated.
3. Representation v2 is retrained using geometry-only non-overlap prediction.
4. Falsification, incremental probe, discovery-control, and blinded annotation
   gates are run.

The fastest defensible path is an integrity-controlled representation study,
not a larger model sweep.
