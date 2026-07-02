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

Scientific validation still requires longer/stronger representation runs,
stronger separation from no-motion and team/slot-invariance controls, full
held-out discovery baseline summaries across seeds, possession- or segment-level
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
