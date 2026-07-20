# PFF Routed Trajectory Forecast Confirmatory Test Protocol V1

Status: frozen after validation selection and before confirmatory test-target generation or any PFF
test trajectory metric.

## Decision And Scope

The user explicitly approved advancing to the next step on 2026-07-17. This protocol spends the
PFF outcome-sealed test reserve exactly once to answer one question:

> Does the validation-selected route, constant velocity at 0.5 seconds and hybrid-context raw at
> 1, 2, and 4 seconds, retain its trajectory-forecasting gains on held-out matches?

This is a confirmatory kinematic forecasting test. It cannot establish tactical concepts, semantic
understanding, tactical reasoning, or representation transfer.

## Test-Reserve Qualification

The eight PFF test matches are outcome-sealed for trajectory forecasting: no previous trajectory
test metric was computed and no trajectory checkpoint was selected from their outcomes. They are
not byte-pristine. A historical TD-JEPA trainer loaded a small test embedding sample after training,
as documented in `docs/TEST_SPLIT_ACCESS_AUDIT_2026_07_14.md`.

The confirmatory result must therefore be described as an outcome-sealed held-out trajectory test,
not as a completely untouched dataset.

## Frozen Data

- Split: `splits/pff_wc2022_64match_inductive_v1.json`
- Split payload SHA-256: `bee86c2c8f917a52a007dbdb92f6082a50f6a33fd278cc8e3f52213cb31f381b`
- Split file SHA-256: `9f7d56184920e463f1aa5fdcee05dc9b59438184910afc93a7e0c12f4e322226`
- Test matches, in frozen split order: `3855`, `3859`, `10506`, `10517`, `3823`, `3836`,
  `3829`, and `3828`.
- Expected source examples: 150,229.
- Source manifest:
  `data/processed/pff_wc2022_td_jepa_v2/observed_only/dataset_manifest.json`
- Source manifest file SHA-256:
  `798125e2ee58c80180690456c2dd9a6fc4d21f99031f13c539fbc3116db5a634`
- Source manifest payload SHA-256:
  `ca53ef656470aea212ec9365881ffcad996a8995615532d152e1cd5df00ebac2`
- Context: 1.0 second at 10 fps, position-only, observed-only.
- Endpoints: 0.5, 1.0, 2.0, and 4.0 seconds.
- Evaluation scope: every prepared test example; no batch cap and no match selection.

Test preparation must project all and only the frozen test matches to `position_only`. The standard
forecast preparation and dataset APIs continue to reject test access. The confirmation runner must
use the explicit test-only access flag, and that flag rejects individual match selection.

## Frozen Inference Rule

- At 0.5 seconds, use deterministic constant-velocity predictions.
- At 1.0, 2.0, and 4.0 seconds, use the matching-seed `hybrid_context_raw` checkpoint.
- Seeds: 7, 11, and 23.
- Do not average coordinates across seeds. Evaluate each routed seed separately, then average its
  metrics for the aggregate gate.
- Do not retrain, fine-tune, select a seed, alter a checkpoint, adjust a threshold, or change the
  routing rule after any test metric is computed.

The confirmation also evaluates the frozen `entity_raw`, `global_raw`, unrouted
`hybrid_context_raw`, and constant-velocity references on the identical examples. These references
are required by the gate and are not new model candidates.

## Frozen Gate

The routed candidate is confirmed only if all nine validation-era conditions hold on test:

1. Player ADE is no more than 1% worse than matching-seed entity raw in at least two of three seeds.
2. Mean player ADE is no more than 1% worse than entity raw.
3. Mean player error is no more than 1% worse than entity raw at every horizon.
4. Ball ADE is no more than 1% worse than matching-seed global raw in at least two of three seeds.
5. Mean ball ADE improves by at least 4% over entity raw.
6. Mean ball ADE is no more than 1% worse than global raw.
7. Mean ball 4-second FDE is no more than 1% worse than global raw.
8. Mean ball error is no more than 2% worse than global raw at every horizon.
9. Mean all-entity ADE improves by at least 2% over global raw.

The gate uses pooled endpoint metrics, matching the validation study. The report also includes
paired, held-out-match bootstrap intervals for player ADE versus entity raw and ball/all-entity ADE
versus global raw. Those intervals describe match variation but do not change the gate.

## Frozen Checkpoints

| Family | Seed | Checkpoint SHA-256 |
|---|---:|---|
| Global raw | 7 | `1be60bac1cc764aab9b943815a0c30de8c518da0c53e9a26c63b93f2ae706ed7` |
| Global raw | 11 | `d5bd566b0f8e435901dd3162b54632a9566ba2d73ded6889cd78e467ba4ccb41` |
| Global raw | 23 | `6d8887a40cc4baaaaee0d406edbd078b7983cebfc8cc62289e0170f9bc78be6c` |
| Entity raw | 7 | `0d2894f6b057a216b5126b27bc6c5369ef3db427ebbe76f86051db9d248e3bce` |
| Entity raw | 11 | `8d1d48529dfb979fafb4323b66ff3e281ee67234358078487a741864899d91b1` |
| Entity raw | 23 | `c65f4f7e853280813dd990c4c5dbd3721072313c123255f2f18eca3efa50d596` |
| Hybrid-context raw | 7 | `2d995ec13226cda547a3a52094d2e0d2f584e5392800db367e2414e4b73df5cf` |
| Hybrid-context raw | 11 | `859b7ae782007d4d342b9072f0eca090b0e4a6baef25119ce4cf40d62afa2a5a` |
| Hybrid-context raw | 23 | `30f6a7003c441a1c4472cb62a8e8cbd3e5bdbac2e91166f46c73dbb062251ab1` |

All checkpoints are scratch-trained raw-family checkpoints frozen at update 2,000. Their paths and
hashes are repeated in `configs/pff_trajectory_forecast_routed_test_v1.lock.json`.

## Execution And Outputs

Runner: `scripts/run_pff_trajectory_forecast_routed_test_v1.py`

Runner SHA-256 at freeze:
`fd1d50baeba925a4d37da0b3ee2055a7bf3d20da84584a0260583cd5c2a0c2df`

The runner stages are:

1. `preflight`: verify the lock and every frozen input without loading test tensors.
2. `prepare`: project the complete test split and create forecast targets.
3. `evaluate`: compute the test result once; refuse if result files already exist.
4. `verify`: verify frozen inputs, artifact hashes, complete scope, and no training/selection.

Primary outputs:

- `runs/pff_trajectory_forecast_routed_test_v1/test_metrics.json`
- `runs/pff_trajectory_forecast_routed_test_v1/gate_summary.json`
- `runs/pff_trajectory_forecast_routed_test_v1/execution_manifest.json`
- `runs/integrity/pff_trajectory_forecast_routed_test_v1_artifact_audit.json`

No per-example predictions are retained.

## Decision Rule After Test

- If all nine conditions pass, report the route as confirmed for held-out PFF trajectory
  forecasting and freeze it as the kinematic baseline for the next research phase.
- If any condition fails, report `not_confirmed`, retain the full result, and do not redesign against
  these test outcomes. Any later redesign requires a new external confirmation dataset.
- In either case, the next tactical-learning study requires separate labels, leakage controls, and
  raw/PCA/random baselines. This trajectory test alone cannot support a tactical claim.
