# Research Status

footballq currently has a useful Phase 1/experiment stack, but latent clusters,
probe scores, and residual rankings are not validated tactical evidence.

Before any tactical claims or Experiment 6 work, scientific runs must use:

- immutable match split manifests with recorded SHA-256 hashes
- period-aware `sample_id` values
- provenance metadata
- leakage-controlled feature views
- non-overlapping TD-JEPA controls
- train-fit and held-out-assignment discovery controls

Legacy artifacts may be inspected only when explicitly marked legacy.

## Sprint Infrastructure Status

The research integrity sprint infrastructure is implemented and smoke-tested for
review as engineering scaffolding. This does not validate tactical claims.

Current reproducibility cleanup adds dataset-name validation for split manifests,
scientific run-manifest writing on the main paper-path entry points, README
command alignment for split/scientific modes, and default
`latent_residual_*` diagnostic artifact names.

Local SkillCorner split folders have been verified for the ten-match manifest,
and one-epoch geometry-only non-overlap v2 diagnostics have been run for three
seeds. These runs are still diagnostics, not paper-quality evidence.
Probe evaluation now records match-level grouped metrics in `eval_test.json`
for uncertainty diagnostics, and the current h2s v2 probe runs have been
re-evaluated with those summaries.

The three-seed geometry-only non-overlap falsification runs are now aggregated
in `runs/td_jepa/v2_nonoverlap_geometry_falsification_gate_extended/`. The gate
status is `blocked`: shuffled future, wrong-match future, pitch reflection,
reversed-time, masked-ball, and home/away label-swap controls pass the current
ratio threshold, while team-slot swap plus context/target player-slot
permutation fail and no-motion remains a caution.

The discovery control summary in
`runs/discovery/v2_nonoverlap_geometry_control_summary/` now includes nuisance
fields for match concentration, transition-magnitude concentration, and minimum
held-out examples per cluster. The current k=32 summaries are not obviously
one-match dominated, but latent-delta quality/concentration remains similar to
raw, PCA, and random-encoder controls, so the discovery outputs remain
diagnostic partitions only.

Earlier h2s incremental probe suites for the prior v2 representation are
aggregated in `runs/probe_suite/v2_nonoverlap_geometry_h2s_incremental_summary/`.
For the lower-weight context-reconstruction candidate, three linear h2s probe
suites are aggregated at
`runs/probe_suite/v2_context_w0p05_slot_recon_margin_h2s_linear_incremental_summary/`.
The current probe result is mixed: future ball displacement has consistent
raw-plus-`z` gains over raw (mean RMSE improvement 0.109; z-scored mean
improvement 0.498), and z-scored raw-plus-`z` improves the all-player
team-shape diagnostic (mean macro-F1 improvement 0.074). But raw-plus-`z` is
consistently worse than raw for global-x bucket (mean macro-F1 delta -0.014;
z-scored delta -0.009) and unnormalized team-shape (mean delta -0.026). Treat
this as a mixed diagnostic probe result, not a downstream evidence pass.

The older combined gate artifact at
`runs/integrity/v2_nonoverlap_geometry_gate_summary_extended.json` reports
`overall_claim_status: blocked` for the earlier representation. It should not be
used as the current lower-weight context candidate's combined gate until
discovery controls are rerun or explicitly marked as carried-forward
diagnostics. Any visualization remains blinded diagnostic material only.

A geometry-only gap-1.0 diagnostic config is available at
`configs/td_jepa_nonoverlap_gap1p0_skillcorner.yaml`, and three one-epoch
diagnostic seeds were run against
`data/processed/skillcorner_td_jepa_nonoverlap_gap1p0.pt`. The extended
falsification summary at
`runs/td_jepa/v2_nonoverlap_geometry_gap1p0_falsification_gate_extended/` remains
`blocked`: no-motion improves to caution rather than fail, but team-slot and
context/target player-slot controls still fail, and home/away label-swap
controls are only caution across seeds.
The comparison artifact at
`runs/td_jepa/v2_nonoverlap_geometry_gap_comparison/` shows the longer gap
improves the worst no-motion ratio but does not materially change slot
sensitivity, and it weakens label-swap separation.

CLS-token encoder pooling is implemented behind `model.pooling: cls` and has a
gap-1.0 diagnostic config at
`configs/td_jepa_nonoverlap_gap1p0_cls_skillcorner.yaml`. One seed was trained
as `runs/td_jepa/20260702_171902`; its extended falsification gate remains
`blocked`, with stronger label-swap separation but no improvement on team-slot
or player-slot controls.

Optional slot-aligned target reconstruction is implemented via
`model.state_decoder_hidden_dim` and `loss.slot_reconstruction_weight`, with a
diagnostic config at
`configs/td_jepa_nonoverlap_gap1p0_slot_recon_skillcorner.yaml`. This is intended
to test whether explicit slot-level pressure can address slot-control failures;
it is not evidence until falsification gates pass. Three one-epoch diagnostic
seeds were run and summarized at
`runs/td_jepa/v2_nonoverlap_geometry_gap1p0_slot_recon_falsification_gate_extended/`
using `total_loss` as the gate metric. Slot reconstruction makes team-slot and
context/target player-slot controls pass, but the run remains `blocked` because
no-motion fails and context-side team-label swap is only caution.
The comparison artifact
`runs/td_jepa/v2_nonoverlap_geometry_gap1p0_slot_recon_comparison/` records this
as a partial redesign result, not a paper-quality pass.

An optional no-motion margin loss is implemented via
`loss.no_motion_margin_weight` and `loss.no_motion_margin`, with a combined
diagnostic config at
`configs/td_jepa_nonoverlap_gap1p0_slot_recon_margin_skillcorner.yaml`. This is
intended to test whether slot-level pressure plus explicit no-motion separation
can clear the remaining falsification blockers. Three one-epoch diagnostic seeds
were run and summarized at
`runs/td_jepa/v2_nonoverlap_geometry_gap1p0_slot_recon_margin_falsification_gate_extended/`.
The margin term clears no-motion decisively under `total_loss` gating, but the
gate remains `blocked` because context-side team/slot controls fall back to
caution or fail. The comparison artifact
`runs/td_jepa/v2_nonoverlap_geometry_gap1p0_slot_recon_margin_comparison/`
shows the current tradeoff: slot reconstruction fixes slot controls, margin
fixes no-motion, and the next redesign/tuning step must preserve both.

A higher slot-reconstruction-weight combined diagnostic is recorded at
`configs/td_jepa_nonoverlap_gap1p0_slot_recon_w0p25_margin_skillcorner.yaml`.
Three one-epoch seeds were summarized at
`runs/td_jepa/v2_nonoverlap_geometry_gap1p0_slot_recon_w0p25_margin_falsification_gate_extended/`.
This preserves the no-motion pass (`ratio_min` 36.08) and improves player-slot
and team-swap sensitivity relative to the first combined diagnostic, but the
gate remains `blocked`: player-slot and team-swap controls are only caution, and
context-side team-label swap remains fail (`ratio_min` 1.04). Do not proceed to
downstream probes, discovery, or visualization from this diagnostic.

Context-side reconstruction is implemented via
`loss.context_reconstruction_weight`, reusing the state decoder on `z_t` to add
`context_reconstruction_loss`. The first context-reconstruction diagnostic at
`configs/td_jepa_nonoverlap_gap1p0_context_slot_recon_margin_skillcorner.yaml`
clears context-side team/slot controls but is still `blocked` by the
no-motion predictor. The lower-weight context diagnostic at
`configs/td_jepa_nonoverlap_gap1p0_context_w0p05_slot_recon_margin_skillcorner.yaml`
is the first current diagnostic to clear the falsification gate under
`total_loss`: three one-epoch seeds are summarized at
`runs/td_jepa/v2_nonoverlap_geometry_gap1p0_context_w0p05_slot_recon_margin_falsification_gate_extended/`,
with `scientific_claim_status: controls_passed`. The minimum passing ratios
include no-motion 1.81, context player-slot 1.34, target player-slot 1.34,
context team-label swap 3.03, and team swap 3.12. This is a falsification-gate
diagnostic pass only; it does not validate downstream probes, discovery
baselines, or visualization.

Scientific validation still requires discovery enrichment beyond raw/PCA/random
controls for the current candidate, stronger or confirmatory probe evidence,
possession- or segment-level uncertainty, and blinded annotation against matched
controls.

Current focused verification:

- import smoke: passed
- synthetic data/window/TD-JEPA preparation smoke: passed for legacy overlap and
  future non-overlap modes
- focused invariant tests and touched-file Ruff: passed
- full test suite: `151 passed`

Known lint exception: repo-wide `python -m ruff check . --statistics` reports 45
pre-existing issues in older decoder, latent-flow, probe, script, and legacy
test files outside the integrity-sprint edits. Touched sprint files are expected
to remain Ruff-clean unless a final repo-wide lint cleanup is explicitly chosen.
