# Integrity Sprint README

This document summarizes what the research-integrity sprint accomplished and
what is still needed before footballq can support paper-quality claims.

As of this handoff, the branch is:

- `codex/research-integrity-sprint-v1`
- latest pushed commit before this document: `349fc08`
- current claim status: diagnostic only

Do not treat current clusters as tactical concepts. Do not describe latent
residual scores as tactical surprise. Do not use possession or availability
probe performance as evidence of emergent tactical understanding while
possession-derived channels are available to the encoder.

## What Was Accomplished

### Provenance And Reproducibility

- Added immutable split-manifest infrastructure with dataset-name validation.
- Propagated period-aware `sample_id` fields through the scientific pipeline.
- Added run-manifest writing for the main paper-path entry points.
- Updated scientific commands to use `--split-manifest` and `--scientific-mode`.
- Renamed residual outputs toward `latent_residual_*` terminology.
- Retired repo-wide Ruff debt; `python -m ruff check .` currently passes.

### SkillCorner Data And Window Coverage

- Verified the local ten-match SkillCorner split folders.
- Extended the availability report to compare raw frame period coverage against
  processed window period coverage.
- Confirmed local raw tracking has `raw_periods=1,2` for all ten matches.
- Identified the current h2s processed artifact as period-1-only, with
  `missing_processed_periods=2`.
- Added stale-cache detection to horizon preparation so period-1-only cached
  chunks are not silently reused.
- Added targeted `--match-ids` and `--skip-combine` horizon preparation.
- Added multi-file/glob `--windows` rendering for diagnostic clips.
- Recovered local period-2 diagnostic media from targeted per-match h2s caches
  without creating one giant combined tensor.

### Representation Diagnostics

- Retrained geometry-only non-overlap v2 diagnostics across three seeds.
- Explored longer prediction gap, CLS pooling, slot reconstruction, no-motion
  margin, and context-side reconstruction variants.
- Identified the current candidate config:
  `configs/td_jepa_nonoverlap_gap1p0_context_w0p05_slot_recon_margin_skillcorner.yaml`.
- This candidate clears the current falsification gate across seeds 7, 11, and
  23 under `total_loss`, but only as a diagnostic pass. It does not validate
  downstream probes, discovery, or visualization.

### Falsification, Probe, And Discovery Gates

- Added and aggregated extended falsification controls.
- Ran h2s incremental probe suites comparing raw, z, raw+z, and z-scored views.
- Ran discovery baselines against normalized latent deltas, raw/PCA/random
  controls, and handcrafted structure metrics.
- Added `scripts/summarize_integrity_gates.py` to combine falsification, probe,
  discovery, and optional annotation summaries.
- Added explicit machine-readable gate blockers. Current blockers include:
  - negative seed- and match-level probe increments for global-x bucket
  - negative seed- and match-level probe increments for unnormalized team-shape
  - latent discovery entropy not separated from controls
  - latent match concentration not separated from controls
  - sparse held-out clusters
  - transition-magnitude concentration
- The gate CLI now prints `next_scientific_action` and one
  `blocking_condition[...]` line per blocker.

### Blinded Visualization And Annotation Scaffolding

- Generated diagnostic blinded clip scaffolds with hidden private keys.
- Generated a balanced diagnostic package with 20 high-residual rows and 20
  hidden low-residual controls.
- Rendered diagnostic GIF media for all 40 balanced rows.
- Added `scripts/validate_blinded_annotation_package.py` to verify:
  - annotator/key consistency
  - hidden-field separation
  - clip-path existence
  - blank annotation cells before human review
- Added `scripts/analyze_blinded_annotations.py` to summarize completion,
  labels, and positive/control enrichment after review.
- Added `docs/BLINDED_ANNOTATION_GUIDE.md` with controlled labels:
  - `tactical_pattern`
  - `routine_motion`
  - `tracking_artifact`
  - `ambiguous`
- The analyzer rejects labels outside the controlled vocabulary with
  `annotation_status: invalid_labels`.

### Verification

Current verification status:

- `python -m ruff check .`: passed
- `python -m pytest -q`: `176 passed`
- current blank annotation package: `annotation_status: incomplete`
- current combined gate: `overall_claim_status: blocked`

## Current Gate Status

The current with-annotation gate command is:

```bash
python scripts/summarize_integrity_gates.py \
  --falsification runs/td_jepa/v2_nonoverlap_geometry_gap1p0_context_w0p05_slot_recon_margin_falsification_gate_extended/td_falsification_gate_summary.json \
  --probe runs/probe_suite/v2_context_w0p05_slot_recon_margin_h2s_linear_incremental_summary/probe_incremental_summary.json \
  --discovery runs/discovery/v2_context_w0p05_slot_recon_margin_control_summary/discovery_control_summary.json \
  --annotation runs/diagnostics/v2_context_w0p05_slot_recon_margin_blinded_balanced_seed7_h02/annotation_summary.json \
  --out runs/integrity/v2_context_w0p05_slot_recon_margin_gate_summary_with_annotation.json
```

Expected current status:

```text
overall_claim_status: blocked
blocking_gates: probe_incremental, discovery_controls, blinded_annotation
next_scientific_action: Resolve mixed incremental probe results or redesign the representation before treating discovery or annotation as evidence.
```

## What Is Still Needed

### Human Or Scientific Work

These items cannot be completed honestly by more code scaffolding alone:

- Complete blinded annotation using only `docs/BLINDED_ANNOTATION_GUIDE.md`,
  `annotator/annotations.csv`, and the referenced GIFs.
- Keep private key files hidden from annotators.
- Analyze annotation enrichment against matched controls after annotation.
- Decide whether to redesign the representation in response to the mixed probe
  and discovery blockers.
- Run an independent clean-room rerun before claiming reproducibility.
- Report uncertainty at match, possession, or segment level before any paper
  claim.

### Engineering Work That May Still Help

- Add stronger paper-relevant probe targets if causally available labels can be
  built without leakage.
- Improve representation objectives only if the redesigned candidate is rerun
  through falsification, probe, discovery, and annotation gates.
- Add convenience scripts around clean-room reproduction, provided they do not
  weaken the scientific gates.
- Add richer annotation QA after real human annotations exist, such as
  inter-annotator agreement, adjudication tracking, or artifact-rate summaries.

## Recommended Next Step

Do not run more unblinded visualization as evidence. The next defensible step is
to resolve the mixed probe/discovery gate outcome: either redesign the
representation and rerun the gates, or complete blinded annotation only as a
diagnostic exercise while preserving the current blocked claim status.

For step-by-step commands, use `docs/NEXT_WEEK_RUNBOOK.md`. For the paper
decision path, use `docs/PAPER_FINAL_PATH.md`.
