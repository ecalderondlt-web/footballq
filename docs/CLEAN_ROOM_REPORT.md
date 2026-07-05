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
  (train 6 / val 2 / test 2); manifest hash
  `0d66a904f30d38c2721b03b189057cc80f1edbf16fdf42b1a35061a874850c71`
  reproduced by `load_split_manifest`.
- Note: the manifest's own `source` field still reads
  `progress_report_unverified`. We deliberately do NOT edit the manifest —
  changing any byte would change its SHA-256 and orphan the lineage hash
  recorded in every downstream artifact. Verification is instead recorded
  here and in the run manifests.
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
- Seed 11 (val): `td_loss 0.00015` vs `no_motion_td_loss 0.00490`;
  `anti_collapse_loss 0.00048`; `cosine_similarity 0.990`.
- Seed 23 (val): `td_loss 0.00023` vs `no_motion_td_loss 0.00996`;
  `anti_collapse_loss 0.00017`; `cosine_similarity 0.985`.

### Falsification gate (total_loss, seeds 7/11/23)

**`scientific_claim_status: controls_passed`, no blocking conditions —
reproduces the original run's outcome.** Ratio of corrupted-condition total
loss over correct pairing (mean across seeds; blockable conditions all pass
with min ratio >= 1.29):

| condition | mean ratio | min ratio | status |
| --- | --- | --- | --- |
| target_team_label_swap | 11.28 | 10.62 | pass |
| pitch_reflection | 5.26 | 4.97 | pass |
| future_from_another_match | 5.00 | 4.73 | pass |
| shuffled_future_within_batch | 4.96 | 4.70 | pass |
| team_swap | 3.31 | 3.16 | pass |
| team_label_swap | 3.17 | 3.04 | pass |
| no_motion_predictor | 2.27 | 1.81 | pass |
| consistent_player_slot_permutation | 1.41 | 1.35 | pass |
| target_consistent_player_slot_permutation | 1.35 | 1.29 | pass |
| masked_ball (excluded from blocking) | 1.09 | 1.06 | caution |
| reversed_time_context (excluded from blocking) | 1.03 | 1.03 | fail |

### Incremental probes

**Reproduces the original run's blockers exactly (same two blocking targets).**
Signed incremental value of raw+z over raw (linear probes, h2s, positive is
better; seed stats over 7/11/23, match stats over held-out matches):

| target | view | seed mean | seed min | all positive |
| --- | --- | --- | --- | --- |
| future_ball_displacement_m | raw | +0.0699 | +0.0614 | yes |
| future_ball_displacement_m | z-scored | +0.2412 | +0.1951 | yes |
| team_shape_change_bucket | z-scored | +0.0486 | +0.0409 | yes |
| future_ball_global_x_bucket | raw | +0.0018 | -0.0060 | **no** |
| future_ball_global_x_bucket | z-scored | -0.0163 | -0.0221 | **no** |
| team_shape_change_bucket | raw | -0.0310 | -0.0427 | **no** |

Same match-level pattern. Note: on seed 7 alone, unnormalized global-x was
positive (+0.013); the three-seed minimum flips it negative — the multi-seed
gate is what makes this conclusion stable.

### Discovery controls

**Reproduces the original run: the latent family does not separate from
controls.** At k=32, delta=0.2 s, averaged over clustering seeds 7/11/23:

- cluster-size entropy: latent 0.8468 vs best control 0.8663
  (margin **-0.0195**, requirement > +0.02)
- top-match fraction: latent 0.5497 vs best control 0.4904
  (margin **-0.0593**, requirement > +0.02; lower is better, latent is worse)
- min held-out examples per cluster: **0** (sparse held-out clusters)
- max share of top-magnitude transitions in one cluster: **1.0**
  (transition-magnitude concentration)

All four of the original run's discovery blockers fire here too.

### Combined gate

**`overall_claim_status: blocked`** with blocking gates
`probe_incremental` and `discovery_controls` (annotation gate not yet
attached at this stage). Blocker-by-blocker comparison with
`docs/INTEGRITY_SPRINT_README.md`:

| blocker (original run) | clean-room |
| --- | --- |
| negative probe increments, global-x bucket | reproduced (both views) |
| negative probe increments, unnormalized team-shape | reproduced |
| latent entropy not separated from controls | reproduced |
| latent match concentration not separated | reproduced |
| sparse held-out clusters | reproduced |
| transition-magnitude concentration | reproduced |
| falsification controls_passed | reproduced |

`next_scientific_action` (verbatim): "Resolve mixed incremental probe results
or redesign the representation before treating discovery or annotation as
evidence."

### Blinded package + model-annotator panel

<!-- TODO: fill after panel run; human annotation package left blank for the
human gate -->

## Verdict

<!-- TODO: reproduced / not reproduced per gate, with any deviations -->
