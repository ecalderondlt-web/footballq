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

A Google Research Football synthetic-pretraining engineering pilot is now available
on `codex/fifa-like-training`. GRF 2.10.3 was built under Ubuntu 22.04 WSL 2 and
20,000 simulator frames from 20 episode IDs produced 9,720 geometry-only,
non-overlapping examples. Three matched one-epoch SkillCorner fine-tunes improved
mean test total loss from 0.006952 for real-only initialization to 0.006533 for
GRF initialization (6.0%), but the seed-level changes were -3.0%, -15.7%, and
+2.0%. Mean TD loss improved 25.7%. The transfer falsification aggregate reports
`controls_passed`, while reversed-time remains a failed detailed control and
masked-ball remains caution. This is a heterogeneous engineering diagnostic, not
evidence of tactical understanding. Matched three-seed h2s linear incremental probes
are now aggregated at
`runs/probe_suite/gfootball_transfer_pilot_v1_h2s_linear_incremental_summary/`.
The transferred representation improves mean future-ball-displacement RMSE over raw
by 0.112 unnormalized and 0.596 train-z-scored, but global-x macro-F1 remains negative
(-0.034 and -0.012) and unnormalized all-player shape remains negative (-0.025).
Z-scored shape remains positive (+0.075), essentially matching the real-only result.
All raw/random baseline rows reproduce exactly across the old and transferred suites.
The transfer probe gate is therefore still mixed and `diagnostic_only`; matched
discovery controls are now aggregated at
`runs/discovery/gfootball_transfer_pilot_v1_control_summary/`. Transferred normalized
latent-delta cluster entropy is 0.821, nearly identical to the random-projection control
at 0.821 and similar to raw/PCA latent-delta controls at 0.827/0.843. Maximum top-match
concentration worsens to 0.335 from 0.241 in the real-only normalized candidate and is
not separated from the controls. The combined transfer gate at
`runs/integrity/gfootball_transfer_pilot_v1_gate_summary.json` is `blocked` by mixed
incremental probes and discovery controls. This pilot does not justify scaling GRF
pretraining or proceeding to annotation without a representation redesign.

A later frozen position-only volume study directly tested 29,453, 120,177, and
240,337 nested GRF examples against an equal-compute 1x replay control before paired
PFF validation fine-tuning. All 27 runs and split-access checks completed, but the
primary 8x gate is `blocked`: mean total validation loss improves 2.62% versus scratch
against a 5% threshold and 0.87% versus replay against a 2% threshold. The 4x family
is fractionally best on mean total loss and substantially best on narrow TD loss;
4x and 8x total losses are effectively tied. This supports keeping GRF as an
optimization initialization while rejecting further raw-volume scaling of the same
scenario mixture. Protocol, full seed results, learning curves, and claim boundaries
are recorded in `docs/GRF_POSITION_SCALE_PROTOCOL_V1.md` and
`docs/GRF_POSITION_SCALE_RESULTS_V1.md`. The PFF test split remained untouched.

The first representation-redesign diagnostic is now implemented. A temporal-order
classifier failed cleanly at chance on GRF, so it was replaced by a signed context
motion objective that predicts each entity's endpoint displacement in both forward
and reversed contexts. The one-seed GRF run at `runs/td_jepa/20260710_023707` beats
the zero-motion MSE baseline by about 8% without degrading base future prediction.
After one SkillCorner train-split fine-tune at `runs/td_jepa/20260710_023808`, validation
motion cosine is 0.529 and reversed context raises TD loss by 1.36x. However, the
total-loss falsification summary remains `blocked`: reversed time is caution at 1.09x
and the blocking no-motion control is caution at 1.185x.

The matched seed-7 real-only redesign control is now recorded at
`runs/td_jepa/20260710_155117`. GRF initialization improves validation total loss by
7.4%, base loss by 7.5%, and signed-motion MSE by 6.8%, but worsens the narrow TD loss
by 7.2%. Reversed-time TD sensitivity is effectively tied (1.38x real-only versus
1.36x GRF), indicating that the redesign rather than GRF supplies that behavior. GRF
does improve total-loss no-motion and player-slot ratios and leaves only no-motion as
a blocker, compared with three blockers for real-only. Both runs remain blocked. The
result justifies a prespecified multi-seed matched repeat, not probes, discovery,
interpretation, or annotation.

That repeat is now complete for seeds 7, 11, and 23. GRF initialization reduces
validation total loss in every seed by 7.4%, 15.8%, and 10.9% (11.5% on the mean),
reduces mean base loss by 11.6%, and reduces mean signed-motion MSE by 8.8%. Narrow TD
loss is mixed by seed and improves only 1.8% on the mean. Three-seed falsification
summaries are stored at
`runs/td_jepa/redesign_{real,transfer}_seed7_11_23_falsification_gate_{total,td}/`.
Both total-loss gates remain blocked. GRF transfer clears both player-slot controls
that block real-only, but its no-motion ratio remains caution with a 1.052x minimum,
below the 1.25x pass threshold. Reversed-time TD sensitivity passes for both families
and remains similar, so the temporal gain belongs to the redesigned objective rather
than synthetic initialization. This is repeatable evidence for an optimization and
robustness benefit, not evidence of semantic or tactical understanding.

A future-transition decoder now directly targets the remaining no-motion blocker by
reconstructing aligned future-minus-current `x,y` coordinates from latent motion. In
the matched three-seed repeat, GRF transfer improves mean validation total loss by
11.2%, temporal-motion MSE by 9.8%, and narrow TD loss by 2.5% relative to real-only.
The GRF transition family clears the total-loss falsification aggregate in every seed:
no-motion ratio mean/minimum is 1.287x/1.286x. Real-only remains blocked at
1.246x/1.242x. Artifacts are at
`runs/td_jepa/redesign_{real,transfer}_transition_seed7_11_23_falsification_gate_{total,td}/`.
This is only a targeted gate pass. Reversed-time total loss remains caution,
masked-ball total loss remains fail, and both TD-only aggregates remain blocked by
slot/team controls. Do not interpret `controls_passed` as all controls passing or as
evidence of semantic understanding.

The remaining apparent ball/slot failures were separated into a frozen,
condition-aware policy at `docs/FALSIFICATION_POLICY_MATCH_INVARIANT_V1.md`. It uses
ball-only dynamic reconstruction for masked-ball, total loss for no-motion, TD loss for
causal prediction corruptions, and near-reference symmetry expectations for arbitrary
team/player tensor-slot permutations. On validation, the GRF transition family passes
all conditions; real-only remains blocked only by no-motion. This policy was designed
from validation evidence, so that pass is exploratory. The policy and selected
three-seed checkpoints are now frozen before their first held-out test application.

The frozen policy has now been applied once to the untouched test split and passes
across all three selected GRF-transition seeds. The aggregate at
`runs/td_jepa/redesign_transfer_transition_seed7_11_23_falsification_gate_match_invariant_v1_test/`
has no blockers; minimum ratios are 1.513x masked-ball, 1.299x no-motion, and 1.365x
reversed-time, with every symmetry check inside `[0.80, 1.25]`. This confirms the
falsification battery only. Incremental probes and discovery controls were still
required and are reported next.

Those remaining gates are now complete. Three-seed h2s linear probes at
`runs/probe_suite/grf_transition_v1_h2s_linear_incremental_summary/` remain mixed:
raw-plus-latent improves displacement RMSE by 0.110 unnormalized and 0.558 z-scored,
and z-scored shape macro-F1 by 0.081, but global-x macro-F1 remains negative (-0.016
and -0.019) and unnormalized shape remains negative (-0.018). Discovery controls at
`runs/discovery/grf_transition_v1_control_summary/` improve normalized latent
match-concentration to 0.244 from the earlier GRF pilot's 0.335, but PCA is better at
0.167, minimum held-out cluster support is only one example, and transition-magnitude
concentration remains extreme. The combined artifact at
`runs/integrity/grf_transition_v1_gate_summary.json` is `blocked` by both incremental
probes and discovery controls. Falsification is no longer the blocker; the learned
representation still lacks broad incremental and discovery evidence.

The three-seed geometry-only non-overlap falsification runs are now aggregated
in `runs/td_jepa/v2_nonoverlap_geometry_falsification_gate_extended/`. The gate
status is `blocked`: shuffled future, wrong-match future, pitch reflection,
reversed-time, masked-ball, and home/away label-swap controls pass the current
ratio threshold, while team-slot swap plus context/target player-slot
permutation fail and no-motion remains a caution.

The current-candidate discovery control summary is aggregated in
`runs/discovery/v2_context_w0p05_slot_recon_margin_control_summary/` across
seeds 7, 11, and 23. It includes normalized latent deltas, raw latent deltas,
PCA latent deltas, random-projection deltas, handcrafted structure metrics, and
PCA handcrafted metrics with train-fit / held-out assignment. The normalized
latent clusters are not obviously one-match dominated, but their entropy and
match-concentration are similar to raw/PCA/random controls (`cluster_size_entropy`
mean 0.833 versus raw 0.839 and random 0.835; max top-match fraction 0.263).
Discovery therefore remains a diagnostic partition, not evidence of a distinct
latent structure.

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

The current combined gate artifact at
`runs/integrity/v2_context_w0p05_slot_recon_margin_gate_summary.json` reports
`overall_claim_status: blocked`. Falsification is now `controls_passed`, but the
incremental probe and discovery-control gates remain `diagnostic_only`.
`scripts/summarize_integrity_gates.py` now records machine-readable blocking
conditions for those gates. For the current candidate, the probe gate records
negative seed- and match-level increments for global-x bucket plus unnormalized
team-shape, while the discovery gate records that latent entropy and
match-concentration are not separated from raw/PCA/random controls, with sparse
held-out clusters and transition-magnitude concentration. The gate CLI now
prints those blocker lines along with `next_scientific_action`. Any
visualization remains blinded diagnostic material only.

A current-candidate blinded annotation scaffold exists at
`runs/diagnostics/v2_context_w0p05_slot_recon_margin_blinded_seed7_h02/`. It was
built from the seed-7 normalized-delta latent-residual examples at
`runs/discovery/v2_context_w0p05_slot_recon_margin_seed7/normalized_delta_z_h02/`.
The annotator CSV contains 40 blind rows without cluster IDs or residual scores,
and the private key stores those hidden fields. Diagnostic GIF media now covers
all 40 rows. The first render filled 25 period-1 rows from
`data/processed/skillcorner_windows_h2s.pt`; targeted full-period h2s per-match
caches filled the remaining period-2 rows. The renderer writes a
`render_manifest.json` coverage summary with `missing_windows=0`. This complete
media scaffold is still diagnostic material, not blinded annotation evidence.
`scripts/report_skillcorner_availability.py` now compares raw frame period
coverage with processed-window period coverage. On the current local
SkillCorner files it confirms `raw_periods=1,2` for all ten matches, while the
current h2s artifact reports `periods=1`, `window_count=315400`, and
`missing_processed_periods=2` for every match. This makes the blocker a
processed-artifact coverage/provenance issue, not missing raw period-2 data.
`scripts/prepare_tracking_horizons.py --resume` now checks cached per-match
window chunks against raw match periods before reuse, so the existing
period-1-only `.skillcorner_window_cache` files are flagged as stale and must be
regenerated rather than silently recombined.
The horizon preparer also supports `--match-ids` and `--skip-combine`, and the
diagnostic renderer accepts multiple `--windows` inputs plus glob patterns. This
enables targeted h2s period-2 media recovery for the eight matches referenced by
the current missing blinded rows without writing one giant combined window
artifact.
`scripts/validate_blinded_annotation_package.py` validates annotator/key/manifest
consistency, hidden-field separation, clip-path existence, and blank annotation
cells before human review. Both current diagnostic packages pass locally with 40
rows, 40 clip paths, zero missing clips, and zero filled annotation cells.
`scripts/analyze_blinded_annotations.py` now joins completed annotator rows to
the private key after review and reports completion, label counts, and
positive/control enrichment diagnostics. On the current blank balanced package it
correctly reports `annotation_status: incomplete`, `completed_count: 0`, and
`claim_status: diagnostic_only`. `scripts/summarize_integrity_gates.py` can
optionally include that annotation summary; the current with-annotation gate
remains `blocked` by `probe_incremental`, `discovery_controls`, and
`blinded_annotation`.
`docs/BLINDED_ANNOTATION_GUIDE.md` defines the annotator-facing protocol and
controlled labels: `tactical_pattern`, `routine_motion`, `tracking_artifact`,
and `ambiguous`. The annotation analyzer now rejects filled rows outside this
controlled vocabulary with `annotation_status: invalid_labels`.

A balanced diagnostic scaffold was also generated at
`runs/diagnostics/v2_context_w0p05_slot_recon_margin_blinded_balanced_seed7_h02/`
using 20 high-residual rows and 20 hidden low-residual controls selected from
the same examples file. The annotator CSV remains blinded; the private key
stores `positive_control`, `rank_source`, `control_group`, and
`control_match_reason`. Targeted full-period h2s caches now bring media coverage
to 40 of 40 rows with `missing_windows=0`; this remains diagnostic media only.

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

Scientific validation still requires stronger or confirmatory probe evidence,
discovery enrichment beyond raw/PCA/random controls, possession- or
segment-level uncertainty, and blinded annotation against matched controls.

Current focused verification:

- import smoke: passed
- synthetic data/window/TD-JEPA preparation smoke: passed for legacy overlap and
  future non-overlap modes
- focused invariant tests and touched-file Ruff: passed
- blinded renderer focused tests and touched-file Ruff: passed
- blinded annotation package validation: passed for both current diagnostic
  packages
- blinded annotation analysis: current package correctly reports incomplete
- repo-wide Ruff: passed
- full test suite: `291 passed`

PFF FC World Cup 2022 ingestion is now implemented as a bounded diagnostic path. The complete
`wc2022datav2` delivery contains 64 unique tracking games but no companion metadata. The immutable
48/8/8 match split is `splits/pff_wc2022_64match_inductive_v1.json`. See
`docs/PFF_WORLD_CUP_2022_INTEGRATION.md`.

The full PFF canonical-v2 conversion is now complete: 64 matches, 11,849,815 frames, 2,039
period-aware Parquet shards, zero frame gaps, and source/shard checksums. Deterministic roster slots
replace period-wide jersey accumulation; 15,643 overlap-only substitution rows were omitted and
recorded, leaving only 0.00029% nonstandard player shapes. The finalized all-available,
geometry-only, future-nonoverlap TD manifest contains 1,975,069 unique examples with tensor hashes
and lazy shard-grouped loading. A one-batch production-manifest diagnostic at
`runs/td_jepa/20260712_225735` passed but is plumbing evidence only. The finalized observed-only
control contains 1,135,478 unique examples in 2,039 hashed shards, with 844,195/141,054/150,229
train/validation/test examples and manifest hash
`ca53ef656470aea212ec9365881ffcad996a8995615532d152e1cd5df00ebac2`.

Matched 100-update diagnostics are complete for independently paired seeds 7, 11, and 23. GRF
initialization lowers mean combined validation loss by 12.2% on all-available tracking and 10.3%
on observed-only tracking, with improvement in every seed. All-available narrow TD loss worsens
8.0% on the mean, while observed-only narrow TD loss improves 23.5%. This is repeatable evidence
for an early optimization benefit, not final convergence, downstream value, or tactical
understanding. Exact runs and limitations are recorded in
`docs/PFF_GRF_TRANSFER_DIAGNOSTIC_V1.md`. Metadata provenance, a prespecified longer repeat,
held-out falsification, incremental probes, and discovery controls remain required before claims.

The prespecified longer observed-only repeat is now complete under
`docs/PFF_GRF_TRANSFER_LONGER_PROTOCOL_V1.md`: 2,000 training batches and 500 validation batches
for matched seeds 7, 11, and 23. GRF initialization still lowers combined validation loss in all
three seeds, but the mean benefit shrinks to 1.297%, below the frozen 5% threshold. Mean narrow TD
loss worsens by 10.005%, also violating the frozen non-degradation rule. The machine-readable gate
at `runs/integrity/pff_grf_transfer_longer_v1_gate_summary.json` is `blocked`. No PFF test metrics
or held-out falsification controls were run. A later integrity audit found that the legacy trainer
still exported an embedding from the first test batch after training; this did not affect weights,
validation metrics, or checkpoint selection, but the split was not completely untouched. This
supports a warm-start interpretation, not a persistent representation-quality benefit. See
`docs/TEST_SPLIT_ACCESS_AUDIT_2026_07_14.md` and
`docs/PFF_GRF_TRANSFER_LONGER_RESULT_V1.md`.

The next GRF engineering step is now implemented without scaling the blocked model. A frozen V2
curriculum collected 76,451 raw frames across 106 preassigned episodes spanning three 11v11
difficulties, a perturbed policy, and six reduced-player academy scenario families. A visibility
profile fit only on 874,828 PFF training frames is applied during sharded preparation. The
resulting geometry-only, future-nonoverlap manifest has 36,733 unique examples split
33,591/2,066/1,076 by train/validation/test, with payload hash
`f8d524157196ab007c21c0662a15b18e9458b2468b5a2af349895cec4bb981db`. The 11v11 shards match
the real-training visibility target closely; reduced-player drills intentionally contain fewer
entities. A one-batch integration run at `runs/td_jepa/20260713_143446` passed. This is data and
plumbing validation only. No V2 transfer comparison, PFF test evaluation, probe, discovery, or
tactical claim has been made.

The prespecified easy-only versus balanced-V2 comparison is now complete under
`docs/PFF_GRF_CURRICULUM_COMPARISON_PROTOCOL_V1.md`. Both synthetic families received exactly 263
updates with the same PFF-train-derived visibility masking; matching-seed latest checkpoints then
initialized six 2,000-update PFF observed-only runs. V2 wins total validation loss in two of three
seeds, but its mean total loss is 0.092% worse and mean narrow TD loss is 4.572% worse. The frozen
gate at `runs/integrity/pff_grf_curriculum_comparison_v1_gate_summary.json` is `blocked` by the mean
total-improvement and TD non-degradation criteria. No synthetic or PFF test metrics were computed.
See `docs/PFF_GRF_CURRICULUM_COMPARISON_RESULT_V1.md`.

The follow-up scenario-aware sampling ablation is also complete. A frozen square-root shard sampler
raised academy exposure from 11.5% to 27.7% without using validation outcomes to choose weights.
Against the existing natural-V2 baseline, it wins one of three PFF seeds, improves mean total loss
by only 0.181%, and worsens mean narrow TD loss by 5.783%. The exploratory gate at
`runs/integrity/pff_grf_sampler_comparison_v1_gate_summary.json` is `blocked` on seed wins, material
mean improvement, and TD non-degradation. No test metrics were computed. See
`docs/PFF_GRF_SAMPLER_COMPARISON_RESULT_V1.md`.

A deterministic train-only GRF-to-PFF domain-gap audit is complete under
`docs/GRF_PFF_TRAIN_DOMAIN_GAP_PROTOCOL_V1.md`, using 24,576 contexts per source and all 48 PFF
training matches. Corrected robust rescoring still places player acceleration at `1.4508`: GRF
mean is `7.40 m/s^2` versus `0.82 m/s^2` in PFF. No validation or test shard was loaded.

The provider-neutral causal-velocity preflight is now complete and `blocked`. All identities,
masks, coordinates, splits, and temporal indices match across 33,591 GRF training examples, but
the acceleration gap changes only from `1.4508` to `1.4487` and mean GRF acceleration rises to
`22.97 m/s^2`. The original V1 formal pass was invalidated because a standard-deviation fallback
could hide rare extreme values; V2 was frozen before corrected rescoring and adds physical summary
guards. No model run, validation read, or test read followed. The next step is a frozen train-only
discontinuity audit, not scaled training. See
`docs/GRF_PROVIDER_NEUTRAL_MOTION_RESULT_V2.md`.

That frozen raw-position discontinuity audit is now complete across all 10 GRF train jobs, 92
episodes, and 69,773 frames. It finds 10,478 player accelerations at or above `100 m/s^2`; 100% of
their acceleration mass lies within five frames of a score or game-mode event, and 99.671% is
associated with a one-frame position jump of at least 3 metres. The maximum is `7,868.25 m/s^2`,
with top examples showing 70-78 metre event repositioning. All extremes occur in full-match jobs;
none occur in academy jobs. The frozen rule selects event-boundary segmentation or masking as the
next candidate. Typical causal acceleration remains elevated, so this repairs the catastrophic
tail but is not expected by itself to solve the robust domain gap. No held-out source or model run
was used. See `docs/GRF_POSITION_DISCONTINUITY_AUDIT_RESULT_V1.md`.

The prespecified event-boundary causal reconstruction is also complete and `blocked`. It removes
3,758 event-proximate frames and eliminates unsafe tensor references while preserving exact
identity, masks, coordinates, and stride phase for retained samples. Mean sampled GRF acceleration
falls from `22.97` to `7.01 m/s^2`, confirming repair of the catastrophic reset tail. However, the
robust acceleration gap improves only 1.563% (`1.4508` to `1.4281`) and only 16,785/33,591 train
examples remain (49.969%), failing the frozen 25% gap-improvement and 75% retention rules. No GRF
validation/test job, PFF validation/test tensor, or model run was used. See
`docs/GRF_EVENT_BOUNDARY_RECONSTRUCTION_RESULT_V1.md`.

The subsequent train-only motion feature-view comparison retains all raw frames and segments only
at observed `3 m` player or `10 m` ball jumps. Both candidates retain 30,134/33,591 examples
(89.709%) with exact shared identities, masks, coordinates, and zero boundary crossings. A truly
causal 0.5-second velocity lowers mean acceleration to `5.26 m/s^2` but worsens the corrected robust
gap from `1.4508` to `1.5434`, so its motion gate is `blocked`. Under the frozen rule, the mechanically
projected five-channel position-only view is selected for a future matched model protocol. This is
feature selection only; no model or held-out tensor was run. See
`docs/GRF_MOTION_FEATURE_VIEW_RESULT_V1.md`.

The selected position-only view has now completed its separately frozen matched model study. A
train/validation-only PFF projection contains 985,249 examples in 1,766 shards and no test shards.
At 100 PFF updates, matching-seed GRF initialization improves mean validation total loss by 14.735%
and narrow TD loss by 21.209%, passing every early diagnostic criterion. At 2,000 updates it still
wins all three seeds and improves narrow TD loss by 10.522%, but total-loss improvement shrinks to
1.540%, below the frozen 5% threshold. The longer gate at
`runs/integrity/pff_grf_position_only_longer_v1_gate_summary.json` is therefore `blocked`. All new
run manifests record only train/validation access and no embedding export; no PFF test tensor was
projected or loaded. Position-only removes the prior longer-budget narrow-TD regression but does
not establish a material persistent combined-objective benefit. See
`docs/PFF_GRF_POSITION_ONLY_DIAGNOSTIC_RESULT_V1.md` and
`docs/PFF_GRF_POSITION_ONLY_LONGER_RESULT_V1.md`.

The prespecified complete tracking-backbone convergence study is now complete under
`docs/PFF_4X_TRACKING_BACKBONE_COMPLETE_PROTOCOL_V1.md`. Scratch and the selected 4x GRF
initialization each received exactly 10,000 PFF train updates for seeds 7, 11, and 23. GRF retains
a strong motion-specific effect, lowering mean narrow TD validation loss by 32.375%, but it worsens
mean combined validation loss by 3.930% and loses that broader comparison in all three seeds. The
frozen gate at `runs/pff_4x_tracking_complete_v1/gate_summary.json` is therefore `blocked`, and
scratch is the operational tracking-backbone family. All access and artifact checks passed; every
run loaded only train and validation tensors, embedding export was disabled, and PFF test remained
untouched. See `docs/PFF_4X_TRACKING_BACKBONE_COMPLETE_RESULT_V1.md`.

The first StatsBomb semantic-event phase is now complete under
`docs/STATSBOMB_SEMANTIC_PRETRAIN_PROTOCOL_V1.md`. The pinned Open Data snapshot contains 4,235
matches and a deterministic 3,388/425/422 match-level train/validation/test split. Only train and
validation were tensorized, producing 739,046/93,479 period-safe causal windows. Across seeds 7,
11, and 23, the event-only encoder reaches mean event-type NLL 0.530884, improving 47.978% over the
train-fitted first-order Markov control. Naive sparse-360 conditioning is blocked: mean anchored
event NLL worsens 0.132%, anchored location MAE worsens 6.667%, and 360 loses anchored location in
all three seeds. `event_only` is therefore the operational semantic family. This validates useful
higher-order event prediction, not tactical concepts, cross-modal value, or semantic understanding.
No StatsBomb test event was tensorized or evaluated, and PFF data was not loaded. See
`docs/STATSBOMB_SEMANTIC_PRETRAIN_RESULT_V1.md`.

The prespecified frozen-context integration study is also complete under
`docs/PFF_STATSBOMB_CONTEXT_RESIDUAL_PROTOCOL_V1.md`. A train-only provider audit retained 71,300
PFF events, mapped 93.58% explicitly, preserved 4,575 as unknown, and excluded 60,778 generic `OTB`
interval markers. Twelve matched residual-head runs then compared tracking-only, raw-event, random
encoder, and StatsBomb-pretrained context across seeds 7, 11, and 23. Pretrained context beats
tracking in all three seeds but improves mean latent TD loss by only 0.457%, below the frozen 1%
threshold. It improves only 0.121% over raw events and 0.378% over a random encoder. The gate is
`blocked`, raw also misses its 1% fallback threshold, and tracking remains operational. This is a
small event-context effect on frozen latent prediction, not tactical or semantic understanding.
PFF test events and tracking tensors were not loaded. See
`docs/PFF_STATSBOMB_CONTEXT_RESIDUAL_RESULT_V1.md`.

The first real downstream multi-horizon trajectory gate is now complete under
`docs/PFF_TRAJECTORY_FORECAST_PROTOCOL_V1.md`. It uses 844,195 PFF training contexts, 141,054
validation contexts, and a fixed 64,000-example final validation subset at 0.5/1/2/4 seconds.
The raw learned forecaster improves player ADE 2.561% and ball ADE 27.725% over constant velocity.
The selected frozen tracking backbone beats raw in all three seeds but improves mean player ADE
only 0.758%, below the frozen 2% threshold, while fine-tuning worsens mean player ADE by 0.356%.
The transfer gate is therefore `blocked` and raw is operational. Data-lineage and artifact audits
passed; no PFF test targets were generated, no test tracking tensor was loaded, and no embedding
was exported. See `docs/PFF_TRAJECTORY_FORECAST_RESULT_V1.md`.

The prespecified entity-preserving trajectory follow-up is now complete under
`docs/PFF_TRAJECTORY_FORECAST_ENTITY_PROTOCOL_V1.md`. Relative to the prior global raw model, the
smaller entity-token raw model wins all three seeds and improves mean player ADE by 3.980%, with
player-error gains of 20.208%, 8.215%, 1.885%, and 3.475% at 0.5/1/2/4 seconds. Ball ADE worsens
5.463% and wins only one seed, so the frozen redesign gate is `blocked` and global raw remains
operational. Frozen transfer worsens player ADE 5.085% and ball ADE 10.264%; fine-tuning is
effectively tied with entity raw on players and worsens ball ADE 0.781%. Artifact checks passed,
and PFF test targets and tensors remained untouched. Entity identity is useful for player
forecasting, but the next controlled study needs a dedicated ball head. See
`docs/PFF_TRAJECTORY_FORECAST_ENTITY_RESULT_V1.md`.

That dedicated player/ball type-head study is now complete under
`docs/PFF_TRAJECTORY_FORECAST_TYPE_HEADS_PROTOCOL_V1.md`. Type-head raw retains mean player ADE
but improves ball ADE only 0.522% versus entity raw, misses the frozen 5% target, remains 4.913%
worse than global raw on ball ADE, and worsens ball 4-second FDE 1.138%. Fine-tuning improves
player ADE 1.243% versus entity raw but improves ball ADE only 0.838% and violates the player
short-horizon guard. Frozen is worse on both player and ball means. All redesign and transfer gates
are `blocked`; global raw remains operational. Artifact checks passed, and PFF test targets and
tensors remained untouched. The result rejects head specialization alone and motivates a
scratch-only hybrid ball head with explicit all-entity kinematics before any further transfer
runs. See `docs/PFF_TRAJECTORY_FORECAST_TYPE_HEADS_RESULT_V1.md`.

The scratch-only hybrid-context validation follow-up is complete under
`docs/PFF_TRAJECTORY_FORECAST_HYBRID_CONTEXT_PROTOCOL_V1.md`. Giving the ball decoder explicit
last-state kinematics for all 23 entities improves mean player ADE 0.485% versus entity raw, ball
ADE 3.068% versus global raw, ball 4-second FDE 4.466%, and all-entity ADE 4.213%. Seven of nine
frozen conditions pass across seeds 7, 11, and 23. The gate remains `blocked` only because player
and ball errors at 0.5 seconds worsen 8.778% and 4.138%, respectively; all 1-4 second component
comparisons pass. A post hoc descriptive route using the frozen constant-velocity baseline at 0.5
seconds and hybrid predictions at 1-4 seconds would clear every threshold and improve all-entity
ADE 4.461%, but it is not a formal pass because the rule was selected after validation inspection.
The route was subsequently frozen and evaluated once on all 150,229 examples from the eight PFF
test matches under `docs/PFF_TRAJECTORY_FORECAST_ROUTED_TEST_PROTOCOL_V1.md`. It passes all nine
conditions: mean player ADE improves 0.498% versus entity raw, ball ADE improves 2.029% versus global
raw, ball 4-second FDE improves 2.249%, and all-entity ADE improves 4.746%. The 1-second player guard
passes narrowly at -0.9999% against its -1% limit. Held-out-match intervals are positive for ball
and all-entity ADE, while the smaller player-only interval crosses zero. The artifact audit passes,
no training or checkpoint selection occurred, and the route is now the operational kinematic
baseline. This is a held-out trajectory result, not tactical evidence. The reserve was outcome-
sealed for trajectory metrics but not byte-pristine due the historical embedding access documented
in `docs/TEST_SPLIT_ACCESS_AUDIT_2026_07_14.md`. See
`docs/PFF_TRAJECTORY_FORECAST_ROUTED_TEST_RESULT_V1.md`.

The former repo-wide Ruff debt is retired and the current status is tracked in
`docs/LINT_STATUS.md`.

The local FOOTPASS tactical training release has now passed a source-availability
and adapter audit. Its 48 matches comprise 96 half-match datasets, 157,163,622
player rows, 7,160,226 unique frames, and 91,327 labelled action rows. A frozen
38/5/5 match-level development split, period-aware sample identities, lazy HDF5
reader, variable-player masks, and separate geometry/identity/ROI/label views are
implemented. Both the labelled 14-column and unlabelled 13-column schemas are
covered by focused tests. This is provenance and data plumbing only: FOOTPASS
contains no ball coordinates, and the adapter work alone does not establish
tactical understanding. See `docs/FOOTPASS_INTEGRATION.md`.

Two repeated-team FOOTPASS player-history development studies are now complete.
V1 used 11,908 opportunities from verified Bayern, Napoli, and Lazio identities.
A large prior-history view improved NLL 6.57% over a weak current-match rolling
baseline, but its bootstrap interval crossed zero, a role-mean and one shuffled
history control performed better, turnover NLL worsened 4.01%, and geometry plus
role remained substantially stronger. The V1 gate failed. See
`docs/FOOTPASS_PLAYER_HISTORY_DEVELOPMENT_RESULTS_V1.md`.

V2 then tested a compact 28-value, equal-match-weighted player residual shrunk
toward a leave-one-player-out role prior and compared it directly with geometry
plus role. The true player residual worsened primary NLL 3.13%, worsened two of
three validation matches, lost to a shuffled-player control, and was negative
in all three predeclared internal development folds. Average prior support was
only 2.14 matches and never exceeded five. The V2 gate failed. No confirmation
freeze was created, and FOOTPASS matches 22, 40, and 43 remain outcome-sealed.
These results provide no evidence of predictive persistent-player memory or
tactical understanding in the current FOOTPASS cohort. See
`docs/FOOTPASS_COMPACT_PLAYER_RESIDUAL_RESULTS_V2.md`.

The subsequent player-conditioned event studies are consolidated in
`docs/PLAYER_CONDITIONED_EXPERIMENT_LOG_V1.md`. The key positive result is
narrow: an outcome-free Wyscout pass fingerprint remains identifiable across
club and national-team contexts. Its confirmatory same-role pairwise AUC is
`0.8198`, and all frozen retrieval checks pass. This establishes persistent
on-ball identity signal, not improved tactical prediction.

Downstream predictive tests remain negative or blocked. Wyscout cross-team
action conditioning missed its NLL and bootstrap gates; StatsBomb critical-
event profiles were unstable; and the initially positive StatsBomb recipient-
history development NLL effect reversed on the pooled 82-match tournament
confirmation (`-0.4181%` relative improvement versus rolling involvement).
Together with the PFF and FOOTPASS nulls, the current evidence says that
player-specific signal exists in event histories but the tested mechanisms do
not yet convert it into robust incremental outcome prediction.
