# PFF Hybrid-Context Ball Forecast Result V1

Protocol: `docs/PFF_TRAJECTORY_FORECAST_HYBRID_CONTEXT_PROTOCOL_V1.md`

Machine summary: `runs/pff_trajectory_forecast_hybrid_context_v1/gate_summary.json`

Artifact audit: `runs/integrity/pff_trajectory_forecast_hybrid_context_v1_artifact_audit.json`

## Outcome

The frozen hybrid-context gate is **blocked**, and the operational family remains `global_raw`.

The result is nevertheless a strong architecture advance. Hybrid raw improves mean player ADE by
**0.485%** over entity raw, mean ball ADE by **3.068%** over global raw, ball 4-second FDE by
**4.466%**, and all-entity ADE by **4.213%**. It passes seven of nine frozen conditions.

Only the two 0.5-second horizon guards block promotion: player error is 8.778% worse than entity
raw and ball error is 4.138% worse than global raw at that horizon. Errors from 1 through 4 seconds
meet their reference requirements.

## Integrity Boundary

- Forecast examples: 844,195 train / 141,054 validation.
- Final validation sample: 64,000 examples with the required identical digest in all three runs.
- Forecast horizons: 0.5, 1.0, 2.0, and 4.0 seconds.
- Learned runs: scratch only, seeds 7, 11, and 23, each trained for 2,000 updates.
- Loaded splits: train and validation only.
- PFF test targets generated: no.
- PFF test tensors loaded: no.
- Embeddings exported: no.
- Frozen-input and final artifact audits: passed.

## Mean Results

| Architecture / family | Player ADE (m) | Player 4 s FDE (m) | Ball ADE (m) | Ball 4 s FDE (m) | All ADE (m) |
|---|---:|---:|---:|---:|---:|
| Prior global raw | 1.5230 | 4.9445 | 7.2099 | 17.4704 | 1.7572 |
| Prior entity raw | 1.4624 | 4.7727 | 7.6038 | 18.0052 | 1.7153 |
| Hybrid-context raw | **1.4553** | **4.7262** | **6.9886** | **16.6901** | **1.6831** |

Explicit all-entity kinematics in the ball decoder recovers the global model's ball performance
while retaining the entity model's player mean. The improvement is not explained by greater total
capacity: hybrid raw has 650,768 parameters versus 900,280 in global raw.

## Horizon Detail

### Players

| Family | 0.5 s (m) | 1.0 s (m) | 2.0 s (m) | 4.0 s (m) |
|---|---:|---:|---:|---:|
| Entity raw reference | **0.1290** | **0.4234** | 1.4898 | 4.7727 |
| Hybrid-context raw | 0.1403 | 0.4257 | **1.4827** | **4.7262** |

Hybrid player error is 0.554% worse at 1 second, then improves 0.477% at 2 seconds and 0.974% at
4 seconds. Those horizons pass. The 8.778% regression at 0.5 seconds is the player blocker.

### Ball

| Family | 0.5 s (m) | 1.0 s (m) | 2.0 s (m) | 4.0 s (m) |
|---|---:|---:|---:|---:|
| Global raw reference | **1.3729** | 3.9963 | 9.0094 | 17.4704 |
| Hybrid-context raw | 1.4297 | **3.9606** | **8.7245** | **16.6901** |

Hybrid ball error improves 0.893%, 3.162%, and 4.466% at 1, 2, and 4 seconds. Its 4.138%
regression at 0.5 seconds exceeds the frozen 2% horizon guard.

## Seed Results

| Seed | Player ADE (m) | Ball ADE (m) | All ADE (m) |
|---:|---:|---:|---:|
| 7 | 1.4428 | 6.6149 | 1.6557 |
| 11 | 1.4984 | 7.2143 | 1.7337 |
| 23 | 1.4248 | 7.1367 | 1.6599 |

Player and ball non-degradation each pass in two of three matching seeds. Seed 11 is the weaker
run, but repeatability itself is not a blocker under the frozen rules.

## Frozen Gate Detail

Passed:

- Player non-degraded seeds: 2/3.
- Mean player-ADE improvement versus entity raw: 0.485%.
- Ball non-degraded seeds: 2/3.
- Mean ball-ADE improvement versus entity raw: 8.090%.
- Mean ball-ADE improvement versus global raw: 3.068%.
- Mean ball-FDE improvement versus global raw: 4.466%.
- Mean all-entity ADE improvement versus global raw: 4.213%.

Blocked:

- Worst player-horizon improvement versus entity raw: -8.778% at 0.5 seconds.
- Worst ball-horizon improvement versus global raw: -4.138% at 0.5 seconds.

## Descriptive Horizon Routing

The previously frozen constant-velocity baseline is better at 0.5 seconds than the learned models:
0.1276 m for players and 1.3531 m for the ball. After observing the hybrid result, a descriptive
calculation replaced only its 0.5-second endpoint with constant velocity and retained hybrid
predictions at 1, 2, and 4 seconds.

| Descriptive candidate | Player ADE (m) | Ball ADE (m) | All ADE (m) |
|---|---:|---:|---:|
| Constant velocity at 0.5 s; hybrid at 1-4 s | 1.4517 | 6.9670 | 1.6788 |

That routed candidate would improve player ADE 0.731% versus entity raw, ball ADE 3.368% versus
global raw, and all-entity ADE 4.461% versus global raw. It mathematically clears every frozen gate
condition, with positive ball improvements at all horizons and no player horizon worse than 0.554%.

This is **not** a formal gate pass. The routing rule was selected after seeing validation results,
so it is a post hoc candidate and cannot retroactively promote the study.

## Interpretation

1. Explicit multi-agent kinematics were the missing ingredient for medium- and long-horizon ball
   forecasting. A separate local ball head alone did not work; the global-context ball head does.
2. Learned residual correction is unnecessary or harmful at 0.5 seconds, where straight-line
   extrapolation is already extremely strong. The problem is horizon-specific rather than a broad
   failure of the hybrid architecture.
3. The hybrid is the strongest validation model on aggregate player, ball, and all-entity ADE, but
   the prespecified guard correctly prevents promotion after a material short-horizon regression.
4. The routed candidate is simple, deterministic, and requires no retraining. Because it was
   chosen post hoc, it needs one locked confirmation rather than more validation tuning.
5. No representation-transfer run is justified yet because the scratch family did not formally
   pass its frozen gate.

## Recommended Next Step

Freeze the inference rule exactly as: constant velocity at 0.5 seconds and hybrid-context raw at
1, 2, and 4 seconds, with no seed, threshold, architecture, or routing changes afterward. Do not
run more validation architecture searches. The next scientific decision is whether to spend the
sealed PFF test split on one final confirmation of this locked candidate; that decision should be
explicit because the test reserve cannot be restored after inspection.

No tactical-concept or semantic-understanding claim is supported by this result.
