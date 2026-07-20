# PFF Player/Ball Type-Head Trajectory Forecast Result V1

Protocol: `docs/PFF_TRAJECTORY_FORECAST_TYPE_HEADS_PROTOCOL_V1.md`

Machine summary: `runs/pff_trajectory_forecast_type_heads_v1/gate_summary.json`

Artifact audit: `runs/integrity/pff_trajectory_forecast_type_heads_v1_artifact_audit.json`

## Outcome

The frozen type-head redesign gate is **blocked**, and both representation-transfer gates are
**blocked**. The operational family remains `global_raw`.

A dedicated ball decoder does not repair the ball regression from the entity-token study. Type-head
raw improves mean ball ADE by only **0.522%** relative to entity raw, far below the prespecified 5%
threshold, and worsens ball 4-second FDE by **1.138%**. Fine-tuning is slightly better but still
improves ball ADE by only **0.838%** and worsens ball FDE by **1.067%**.

Player means remain competitive, but both raw and fine-tuned type heads violate the frozen
short-horizon non-degradation rule. No type-head family passes the complete downstream gate.

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
| Prior entity raw | 1.4624 | 4.7727 | 7.6038 | 18.0052 |
| Type-head raw | 1.4570 | 4.7442 | 7.5641 | 18.2102 |
| Type-head frozen | 1.5001 | 4.9175 | 7.7069 | 18.8251 |
| Type-head fine-tuned | **1.4442** | **4.6867** | 7.5400 | 18.1974 |

Type-head raw improves player ADE by **0.373%** and fine-tuned improves it by **1.243%** relative
to entity raw. Those aggregate player gains do not compensate for the short-horizon regression or
the missed ball criteria.

## Player Horizon Detail

| Architecture / family | 0.5 s (m) | 1.0 s (m) | 2.0 s (m) | 4.0 s (m) |
|---|---:|---:|---:|---:|
| Prior entity raw | **0.1290** | 0.4234 | 1.4898 | 4.7727 |
| Type-head raw | 0.1341 | **0.4217** | 1.4866 | 4.7442 |
| Type-head frozen | 0.1325 | 0.4370 | 1.5075 | 4.9175 |
| Type-head fine-tuned | 0.1363 | 0.4251 | **1.4747** | **4.6867** |

Raw is 3.912% worse at 0.5 seconds; fine-tuned is 5.629% worse. Both exceed the frozen 1%
non-degradation limit even though their longer-horizon player errors improve slightly.

## Ball Horizon Detail

| Architecture / family | 0.5 s (m) | 1.0 s (m) | 2.0 s (m) | 4.0 s (m) |
|---|---:|---:|---:|---:|
| Prior global raw | **1.3729** | **3.9963** | **9.0094** | **17.4704** |
| Prior entity raw | 1.4255 | 4.1576 | 9.9366 | 18.0052 |
| Type-head raw | 1.4327 | 4.1239 | 9.6344 | 18.2102 |
| Type-head frozen | 1.4031 | 4.1298 | 9.7372 | 18.8251 |
| Type-head fine-tuned | 1.4184 | 4.1192 | 9.5686 | 18.1974 |

The separate head helps most around 2 seconds, but none of the type-head families matches the
global raw ball model at any horizon. Specialization alone therefore does not restore the broader
context that the global decoder appears to use.

## Frozen Gate Detail

### Type-head raw versus entity raw

- Player non-degraded seeds: 2/3, passed.
- Mean player-ADE improvement: 0.373%, passed the -1% floor.
- Worst player-horizon change: -3.912%, blocked below -1%.
- Ball-ADE seed wins: 2/3, passed.
- Mean ball-ADE improvement: 0.522%, blocked below 5%.
- Global raw ball-ADE ceiling: 7.5641 m versus 7.2099 m, blocked.
- Mean ball-FDE improvement: -1.138%, blocked below zero.

### Type-head frozen versus entity raw

Frozen fails every redesign criterion. It worsens mean player ADE by 2.578%, ball ADE by 1.356%,
and ball FDE by 4.553%.

### Type-head fine-tuned versus entity raw

- Player non-degraded seeds: 2/3, passed.
- Mean player-ADE improvement: 1.243%, passed the -1% floor.
- Worst player-horizon change: -5.629%, blocked below -1%.
- Ball-ADE seed wins: 2/3, passed.
- Mean ball-ADE improvement: 0.838%, blocked below 5%.
- Global raw ball-ADE ceiling: 7.5400 m versus 7.2099 m, blocked.
- Mean ball-FDE improvement: -1.067%, blocked below zero.

### Representation transfer

Frozen transfer loses raw on player and ball means. Fine-tuning wins two seeds for both player and
ball ADE, but improves their means by only 0.873% and 0.318%, below the required 2% and 5%, and
worsens the weakest player horizon by 1.652%. Neither family supports representation value.

## Interpretation

1. Separating player and ball parameters is not enough. The ball head still receives only the ball
   entity token and its own explicit kinematics, while the successful global decoder receives the
   current kinematics of every entity.
2. The global raw model remains better on the ball at all four horizons. This supports, but does
   not prove, the hypothesis that explicit multi-agent context is important for ball prediction.
3. Ball endpoints are only 4.117% of valid validation endpoints, or one for every 23.29 player
   endpoints. That imbalance may contribute to optimization behavior, but Adam reduces the effect
   of simple gradient rescaling for ball-head-only parameters, so loss weighting is not yet the
   cleanest single explanation.
4. The 50-batch diagnostic curves for seed 11 worsen from update 1,000 to 2,000 on the ball in all
   three families. The protocol forbids selecting the earlier checkpoint; this is evidence of
   instability, not a basis for a post hoc win.
5. Pretrained weights again provide no material downstream advantage. Repeating all transfer
   families before the scratch architecture works is not justified.

## Recommended Next Step

Keep the PFF test split sealed. Run a three-seed, scratch-only hybrid-context decoder study. Retain
the successful per-entity player head exactly, but give the ball head its contextual ball token
plus the last position, velocity, and observation flag for all 23 entities. This is a 243-value
ball input rather than the current 133-value local input and directly tests the missing explicit
multi-agent context without changing data, targets, loss, horizons, or player decoding.

Use prior entity raw as the player reference and prior global raw as the ball reference. Only if
that scratch architecture passes should frozen and fine-tuned transfer be rerun. If it also fails,
the next justified direction is a probabilistic or event-conditioned ball model rather than more
deterministic decoder variations.

No tactical-concept or semantic-understanding claim is supported by this result.
