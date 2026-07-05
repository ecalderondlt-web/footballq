# Clean-Room Reproduction Report

This report documents the independent clean-room rerun requested in
`docs/INTEGRITY_SPRINT_README.md` ("Run an independent clean-room rerun before
claiming reproducibility"). It was executed on a second machine, by a different
operator than the sprint implementation, from a fresh checkout of
`codex/research-integrity-sprint-v1` at `acb2ba3`, using only public data and
the repository's documented commands. Work branch: `luis/clean-room-paper-v1`.

## Environment

- Hardware: Apple M3 Max, 128 GB RAM (CPU-only; `device: auto` resolves to CPU)
- OS: macOS (Darwin 25.5.0)
- Python: 3.13.9 (fresh venv), editable install via `pip install -e ".[dev]"`
- PyTorch: 2.12.1
- Original run (for comparison): Emilio's environment, Python/torch versions as
  recorded in his run manifests

## Stage results

### Environment verification

- `python -m pytest -q`: **176 passed** (matches the reported count exactly)
- `python -m ruff check .`: **passes** (matches)

### Data and split verification

- All 10 SkillCorner Open Data matches downloaded fresh from the public
  repository (Git LFS media endpoint); SHA-256 of every file recorded in
  `data/raw/skillcorner/download_sha256.log`.
- Local match directories exactly equal the split manifest's 10 IDs
  (train 6 / val 2 / test 2); manifest hash `0d66a904f30d38c2721b03b189057cc80f1
  edfb16fdf42b1a35061a874850c71` reproduced by `load_split_manifest`.
  <!-- hash split across lines only for markdown width; see manifest -->
- Availability report: 10/10 matches, `raw_periods=1,2` for every match —
  matching the sprint README's claim.

### Window preparation (fresh, post-fix code)

- `prepare_tracking_horizons.py` (h2s, cache per match, both periods):
  **315,400 windows, `periods=1,2`, `missing_processed_periods=none`.**
- Notable: the sprint README described the original h2s artifact as
  period-1-only with the same total count. The fresh build shows the underlying
  window population was complete; the period-1-only flag was a metadata/stale-
  cache defect (fixed by the sprint's period-aware identity + cache validation),
  not missing data. The runbook's period-2 media recovery workaround is not
  needed on a fresh build.

### TD-JEPA dataset (scientific view)

- `prepare_td_jepa_data.py` (non-overlap, gap 1.0 s, geometry_only,
  scientific mode): **178,344 examples**, `state_t = (178344, 10, 23, 7)`.

### Training (candidate config, seeds 7 / 11 / 23)

- Config: `configs/td_jepa_nonoverlap_gap1p0_context_w0p05_slot_recon_margin_skillcorner.yaml`
  (1 epoch per seed, per the diagnostic protocol; ~14 min/seed on this CPU).
- Seed 7 (val): `td_loss 0.00018` vs `no_motion_td_loss 0.00668` (learned
  predictor beats the identity baseline by ~37x on TD loss);
  `anti_collapse_loss ~ 0.0006`; `cosine_similarity 0.989`.
- <!-- TODO: seeds 11/23 one-line metrics -->

### Falsification gate (total_loss, seeds 7/11/23)

<!-- TODO: fill from runs/td_jepa/v2_..._falsification_gate_extended -->

### Incremental probes

<!-- TODO: fill from runs/probe_suite/v2_..._incremental_summary -->

### Discovery controls

<!-- TODO: fill from runs/discovery/v2_..._control_summary -->

### Combined gate

<!-- TODO: fill from runs/integrity/v2_..._gate_summary.json; compare blocker
list line-by-line against docs/INTEGRITY_SPRINT_README.md expectations -->

### Blinded package + model-annotator panel

<!-- TODO: fill after panel run; human annotation package left blank for the
human gate -->

## Verdict

<!-- TODO: reproduced / not reproduced per gate, with any deviations -->
