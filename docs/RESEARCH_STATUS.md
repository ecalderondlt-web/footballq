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

The h2s incremental probe suites are aggregated in
`runs/probe_suite/v2_nonoverlap_geometry_h2s_incremental_summary/` with signed
raw-plus-`z` improvements and match-level deltas. Current diagnostics show
consistent raw-plus-`z` gains for global-x bucket and future ball displacement,
and consistent z-scored raw-plus-`z` gains for the all-player team-shape
diagnostic, but these are geometry/control targets with only two held-out test
matches.

The combined gate artifact at
`runs/integrity/v2_nonoverlap_geometry_gate_summary_extended.json` reports
`overall_claim_status: blocked`. The current next scientific action is to
redesign or retrain the representation until falsification controls pass; any
visualization remains blinded diagnostic material only.

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

Scientific validation still requires longer/stronger representation runs,
stronger separation from no-motion and team/slot-invariance controls, discovery
enrichment beyond raw/PCA/random controls, possession- or segment-level
uncertainty, and blinded annotation against matched controls.

Current focused verification:

- import smoke: passed
- synthetic data/window/TD-JEPA preparation smoke: passed for legacy overlap and
  future non-overlap modes
- focused invariant tests and touched-file Ruff: passed

Known lint exception: repo-wide `python -m ruff check . --statistics` reports 45
pre-existing issues in older decoder, latent-flow, probe, script, and legacy
test files outside the integrity-sprint edits. Touched sprint files are expected
to remain Ruff-clean unless a final repo-wide lint cleanup is explicitly chosen.
