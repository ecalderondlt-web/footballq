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

Scientific validation still requires local SkillCorner split ID verification,
representation v2 retraining with `future_nonoverlap_context_only`,
leakage-controlled probes, held-out discovery controls across multiple seeds,
stability checks, and blinded annotation against matched controls.

Current focused verification:

- import smoke: passed
- synthetic data/window/TD-JEPA preparation smoke: passed for legacy overlap and
  future non-overlap modes
- focused invariant tests and touched-file Ruff: passed

Known lint exception: repo-wide `python -m ruff check .` still reports 48
pre-existing issues in older decoder, latent-flow, probe, script, and legacy
test files outside the integrity-sprint edits. Touched sprint files are expected
to remain Ruff-clean unless a final repo-wide lint cleanup is explicitly chosen.
