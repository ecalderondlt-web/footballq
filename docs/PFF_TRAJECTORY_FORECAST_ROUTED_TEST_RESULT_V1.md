# PFF Routed Trajectory Forecast Confirmatory Test Result V1

Protocol: `docs/PFF_TRAJECTORY_FORECAST_ROUTED_TEST_PROTOCOL_V1.md`

Machine summary: `runs/pff_trajectory_forecast_routed_test_v1/gate_summary.json`

Artifact audit: `runs/integrity/pff_trajectory_forecast_routed_test_v1_artifact_audit.json`

Read-only audit rerun: `python scripts/verify_pff_trajectory_forecast_routed_test_v1.py`

## Outcome

The validation-selected routed forecaster is **confirmed** on the held-out PFF trajectory test. All
nine frozen conditions pass, with no blockers.

The route uses constant velocity at 0.5 seconds and the matching-seed hybrid-context raw model at
1, 2, and 4 seconds. It improves mean all-entity ADE by **4.746%** versus global raw, mean ball ADE
by **2.029%**, and mean player ADE by **0.498%** versus entity raw. Relative to constant velocity at
all horizons, it improves player ADE by **6.405%**, ball ADE by **30.809%**, and all-entity ADE by
**11.945%**.

This confirms a held-out trajectory-forecasting result. It does not demonstrate tactical concepts,
semantic understanding, or tactical reasoning.

## Integrity Boundary

- Frozen split: 48 train / 8 validation / 8 test matches.
- Test examples: 150,229, with every prepared example evaluated.
- Test matches: all eight frozen IDs; no match or batch selection.
- Seeds: 7, 11, and 23.
- Training during confirmation: none.
- Checkpoint or seed selection during confirmation: none.
- Per-example predictions retained: none.
- Test trajectory metric computed: once.
- Artifact audit: passed every check.
- Lock SHA-256: `925a26fcee78051480b5b86ab47f86bbc48a5a148e87f2eb7acfc469abba44d4`.

The test reserve was outcome-sealed for trajectory forecasting but not byte-pristine. Historical
representation code had loaded a small test embedding sample after training. No prior trajectory
test outcome had been computed or used for checkpoint selection.

## Mean Test Results

| Family | Player ADE (m) | Ball ADE (m) | Ball 4 s FDE (m) | All ADE (m) |
|---|---:|---:|---:|---:|
| Constant velocity | 1.5067 | 10.5422 | 28.7325 | 1.8706 |
| Global raw | 1.4894 | 7.4453 | 17.6884 | 1.7293 |
| Entity raw | 1.4173 | 7.9811 | 18.7461 | 1.6816 |
| Hybrid-context raw | 1.4139 | 7.3200 | **17.2906** | 1.6517 |
| Routed candidate | **1.4102** | **7.2942** | **17.2906** | **1.6472** |

The route changes only the 0.5-second endpoint, so its 4-second result equals the hybrid model.

## Horizon Detail

### Players

| Family | 0.5 s (m) | 1.0 s (m) | 2.0 s (m) | 4.0 s (m) |
|---|---:|---:|---:|---:|
| Entity raw | 0.1280 | **0.4136** | 1.4484 | 4.6604 |
| Routed candidate | **0.1254** | 0.4177 | **1.4460** | **4.6252** |
| Improvement | 1.988% | -1.000% | 0.162% | 0.756% |

The 1-second player condition passes narrowly: the routed error is 0.9999% worse against an allowed
1% margin. This is a real pass under the frozen rule, but it should be described as boundary-level,
not as a clear 1-second player advantage.

### Ball

| Family | 0.5 s (m) | 1.0 s (m) | 2.0 s (m) | 4.0 s (m) |
|---|---:|---:|---:|---:|
| Global raw | 1.4735 | 4.2960 | 9.4899 | 17.6884 |
| Routed candidate | **1.4511** | **4.2560** | **9.2679** | **17.2906** |
| Improvement | 1.523% | 0.931% | 2.338% | 2.249% |

Ball performance improves at every frozen horizon.

## Frozen Gate Detail

All conditions pass:

- Player non-degraded seeds: 2/3, required 2/3.
- Mean player ADE improvement versus entity raw: 0.498%.
- Worst player-horizon improvement versus entity raw: -0.9999%, minimum -1%.
- Ball non-degraded seeds: 2/3, required 2/3.
- Mean ball ADE improvement versus entity raw: 8.606%, minimum 4%.
- Mean ball ADE improvement versus global raw: 2.029%.
- Mean ball 4-second FDE improvement versus global raw: 2.249%.
- Worst ball-horizon improvement versus global raw: 0.931%.
- Mean all-entity ADE improvement versus global raw: 4.746%, minimum 2%.

## Match-Level Uncertainty

Paired bootstrap intervals use the eight held-out matches as the resampling units. Positive values
mean the routed candidate is better.

| Comparison | Mean improvement (m) | 95% interval (m) |
|---|---:|---:|
| Player ADE versus entity raw | 0.0062 | -0.0041 to 0.0133 |
| Ball ADE versus global raw | 0.1565 | 0.0229 to 0.2891 |
| All-entity ADE versus global raw | 0.0823 | 0.0743 to 0.0904 |

The ball and integrated gains remain positive across the match-level uncertainty analysis. The
player-only interval crosses zero, so the evidence for a player mean improvement is modest even
though the prespecified pooled gate passes.

## Interpretation

1. Constant velocity is the correct short-horizon tool. It removes the hybrid model's validation
   weakness at 0.5 seconds and improves on the test references at that endpoint.
2. Learned multi-agent context is valuable at longer horizons. A constant-velocity-only system is
   much worse on 1-4 second ball and player movement.
3. Explicit all-entity kinematics in the ball decoder generalizes. The ball gains appear at every
   horizon and remain positive under held-out-match uncertainty.
4. The overall forecasting result is stronger than the player-only gain. The route should be
   treated as an integrated player-and-ball baseline, not evidence of a large player-model advance.
5. Seed 11 remains weaker than seeds 7 and 23, but the frozen repeatability rules pass without
   selecting or discarding a seed.

## Next Scientific Step

Freeze the routed candidate as the operational PFF kinematic forecasting baseline. The next study
should ask whether a learned representation adds held-out value on explicitly defined tactical
tasks beyond raw positions, velocities, spacing, and this forecasting baseline.

That tactical study requires independent labels or aligned events, match-level holdouts, leakage
controls, and raw/PCA/random representation controls. The consumed PFF test outcomes must not be
used for further trajectory-model redesign; a redesigned trajectory candidate would require a new
external confirmation set.
