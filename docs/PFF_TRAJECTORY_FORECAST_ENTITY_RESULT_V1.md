# PFF Entity-Preserving Trajectory Forecast Result V1

Protocol: `docs/PFF_TRAJECTORY_FORECAST_ENTITY_PROTOCOL_V1.md`

Machine summary: `runs/pff_trajectory_forecast_entity_v1/gate_summary.json`

Artifact audit: `runs/integrity/pff_trajectory_forecast_entity_v1_artifact_audit.json`

## Outcome

The frozen entity-preserving redesign gate is **blocked**, and both representation-transfer gates
are **blocked**. The operational family remains `global_raw`.

The result is mixed but informative. Giving the decoder one contextual token per entity improves
player ADE by **3.980%** relative to the prior global-vector model, wins all three matching seeds,
and improves player error at every forecast horizon. Ball ADE, however, worsens by **5.463%** and
wins only one of three seeds. The redesign therefore misses the prespecified requirement to
improve both players and the ball.

Within the entity-preserving architecture, neither the frozen nor fine-tuned pretrained encoder
adds material value over a random initialization.

## Integrity Boundary

- Forecast examples: 844,195 train / 141,054 validation.
- Final validation sample: 64,000 examples with the required identical digest in all nine runs.
- Forecast horizons: 0.5, 1.0, 2.0, and 4.0 seconds.
- Learned runs: 3 families x 3 seeds x 2,000 updates.
- Loaded splits: train and validation only.
- PFF test targets generated: no.
- PFF test tensors loaded: no.
- Embeddings exported: no.
- Frozen-input and final artifact audits: passed.

## Mean Results

| Architecture / family | Player ADE (m) | Player 4 s FDE (m) | Ball ADE (m) | Ball 4 s FDE (m) |
|---|---:|---:|---:|---:|
| Prior global raw | 1.5230 | 4.9445 | **7.2099** | **17.4704** |
| Entity raw | **1.4624** | **4.7727** | 7.6038 | 18.0052 |
| Entity frozen | 1.5368 | 5.0800 | 8.3842 | 20.8833 |
| Entity fine-tuned | 1.4631 | 4.7791 | 7.6632 | 18.3569 |

The entity model has 520,456 parameters versus 900,280 in the prior global model. Its player gain
therefore is not explained by giving it a larger model, although this study was not designed as a
parameter-matched architecture isolation.

## Player Horizon Detail

| Architecture / family | 0.5 s (m) | 1.0 s (m) | 2.0 s (m) | 4.0 s (m) |
|---|---:|---:|---:|---:|
| Prior global raw | 0.1617 | 0.4613 | 1.5184 | 4.9445 |
| Entity raw | **0.1290** | 0.4234 | 1.4898 | **4.7727** |
| Entity frozen | 0.1351 | 0.4322 | 1.5283 | 5.0800 |
| Entity fine-tuned | 0.1338 | **0.4211** | **1.4844** | 4.7791 |

Relative to the prior global raw model, entity raw improves player error by **20.208%** at 0.5
seconds, **8.215%** at 1 second, **1.885%** at 2 seconds, and **3.475%** at 4 seconds. This is a
consistent player-specific architecture benefit, including the prior model's weak short horizon.

## Frozen Gate Detail

### Entity raw versus prior global raw

- Player-ADE seed wins: 3/3, passed.
- Mean player-ADE improvement: 3.980%, passed the 1% threshold.
- Ball-ADE seed wins: 1/3, blocked below 2/3.
- Mean ball-ADE improvement: -5.463%, blocked below the required 5% gain.
- Worst player-horizon improvement: 1.885%, passed the -1% floor.

### Entity frozen versus entity raw

- Player-ADE seed wins: 1/3, blocked below 2/3.
- Mean player-ADE improvement: -5.085%, blocked below 2%.
- Ball-ADE seed wins: 0/3, blocked below 2/3.
- Mean ball-ADE improvement: -10.264%, blocked below 5%.
- Worst player-horizon improvement: -6.438%, blocked below -1%.

### Entity fine-tuned versus entity raw

- Player-ADE seed wins: 1/3, blocked below 2/3.
- Mean player-ADE improvement: -0.047%, blocked below 2%.
- Ball-ADE seed wins: 2/3, passed.
- Mean ball-ADE improvement: -0.781%, blocked below 5%.
- Worst player-horizon improvement: -3.701%, blocked below -1%.

## Interpretation

1. Preserving entity identity through the encoder materially helps player forecasting. The effect
   is consistent across seeds and horizons and is strongest at 0.5 and 1.0 seconds.
2. One shared prediction head is not sufficient for both players and the ball. The ball has very
   different movement and observation dynamics, and its regression is the only reason the
   redesign gate does not pass.
3. The selected pretrained tracking weights do not improve this downstream architecture. Frozen
   transfer is clearly worse; fine-tuning is effectively tied with entity raw and still misses all
   materiality requirements.
4. More updates to the same transferred design are not supported by these results. The next useful
   change is architectural and narrowly targeted at player-versus-ball prediction.

## Recommended Next Step

Keep the PFF test split sealed. Freeze a train/validation-only type-conditioned decoder study that
retains the successful shared player head and adds a dedicated ball head, with the existing global
raw and entity raw results fixed as external references. Require player gains to remain
non-degraded while ball ADE recovers materially. This directly tests the remaining failure without
changing the data, forecast horizons, seeds, or evaluation sample.

No tactical-concept or semantic-understanding claim is supported by this result.
