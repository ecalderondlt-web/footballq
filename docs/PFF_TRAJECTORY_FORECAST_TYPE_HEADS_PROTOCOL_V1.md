# PFF Player/Ball Type-Head Trajectory Forecast Protocol V1

Status: frozen after train-only preflight and before any player/ball-head validation metric was
computed.

## Questions

1. Does retaining one shared player decoder while adding a dedicated ball decoder recover the
   entity-preserving model's ball regression without sacrificing its player gains?
2. Within that type-conditioned architecture, do the selected pretrained tracking weights add
   material value over a random encoder initialization?

These are downstream kinematic tests. They are not tests of tactical concepts, possession,
availability, or semantic understanding.

## Frozen Inputs

- Split: `splits/pff_wc2022_64match_inductive_v1.json`
- Split payload SHA-256: `bee86c2c8f917a52a007dbdb92f6082a50f6a33fd278cc8e3f52213cb31f381b`
- Split file SHA-256: `9f7d56184920e463f1aa5fdcee05dc9b59438184910afc93a7e0c12f4e322226`
- Assignment: 48 train / 8 validation / 8 sealed test matches.
- Forecast manifest: `data/processed/pff_wc2022_trajectory_forecast_v1/dataset_manifest.json`
- Forecast manifest file SHA-256:
  `688761b30c4fbe38d832d09d459e79153acc5851a397ceb600d5bc30c811b537`
- Forecast examples: 844,195 train / 141,054 validation.
- Context: 1.0 second at 10 fps, `position_only`, `observed_only`.
- Endpoints: 0.5, 1.0, 2.0, and 4.0 seconds.
- Final validation subset: first 500 deterministic batches, 64,000 examples.
- Required validation sample digest:
  `5cf1ddab5ee33f318bd6c199674a00d58cdef4a4fd2732f3dcdd522dd0528d8d`.

PFF test target generation and test dataset loading remain rejected by code. No test tracking,
events, targets, or embeddings may be used.

## Frozen Prior Results

The completed entity-token study is the primary external reference:

- Summary: `runs/pff_trajectory_forecast_entity_v1/gate_summary.json`
- Summary SHA-256: `f4ed01a8aa1d470e66712e4b91c27fab1af02d692ac5c7477c2fd050843ece22`
- Artifact audit SHA-256:
  `016a7324e9c3d9e87c5380a9d80109d7339588098feebc08cd01039d2a931011`
- Entity raw mean player ADE: 1.4624011306 m.
- Entity raw mean ball ADE: 7.6037602452 m.
- Entity raw mean ball 4-second FDE: 18.0052379523 m.
- Prior global raw mean player ADE: 1.5230249132 m.
- Prior global raw mean ball ADE: 7.2098667870 m.
- All means average matching seeds 7, 11, and 23 at update 2,000.

## Frozen Backbones

| Seed | Scratch tracking checkpoint SHA-256 |
|---:|---|
| 7 | `267f907a9521fbec1ae31df11b36e931810d087d120b3d5822950c50f7aa7e9f` |
| 11 | `1cc4bdc6fa6e11912baffa1bee8322b95ad2c4dff3f64898f433f0e95b8ae4ff` |
| 23 | `ed6b3ad8b0de7b95ff693ccfe20e85012922ee86f128ed886469a02c1b37a366` |

## Frozen Architecture

The representation remains the entity-preserving Transformer output from the prior study. The
canonical schema fixes the ball at entity index 0 and players at indices 1-22. A shared player MLP
receives each player's contextual token, last position, last-two-point velocity, and observation
flag. A separate MLP with the same shape receives only the ball's corresponding inputs. Both heads
predict four coordinate residuals over the unchanged constant-velocity base.

- Representation mode: `entity_tokens`.
- Decoder mode: `player_ball`.
- Token dimension: 128.
- Hidden dimension per head: 256.
- Total parameters: 622,608.
- Combined decoder parameters: 204,304.
- Prior shared entity model: 520,456 total / 102,152 decoder parameters.
- Prior global model: 900,280 total / 481,976 decoder parameters.
- Loss weighting remains unchanged: mean endpoint displacement over all valid entity-horizon
  pairs. The study changes decoder specialization only; it does not add ball reweighting.

Code hashes at protocol freeze:

- `src/footballq/models/soccer_state_encoder.py`:
  `e21dda98c0605841be9df2fbd20c7ffa4ced5c7d9be910ae782d1ff3453c2eb0`
- `src/footballq/models/trajectory_forecaster.py`:
  `2a98d7c74ed80caaa02aedbdc20746ae724a96390f9678c08640e7658f055226`
- `src/footballq/training/train_trajectory_forecast.py`:
  `ac530ea89dae69878cf4690a9a62a083fd16075c8fac20e32361b85cd1b9648d`
- Imported entity-runner helper code:
  `1e2b0fc53f538cabc8d2b7b25c2699612575f68f4530a3d4f58236f5a6a1ec87`

## Frozen Families And Training

1. `raw`: random entity-token encoder and both type heads, trained end to end.
2. `frozen`: selected pretrained encoder frozen; both type heads trained.
3. `finetuned`: selected pretrained encoder and both type heads trained end to end.

- Seeds: 7, 11, 23.
- Updates: exactly 2,000 per family and seed.
- Batch size: 128.
- AdamW learning rate: `3e-4`; weight decay: `1e-4`.
- Loss: unchanged masked mean endpoint displacement in metres.
- Curves: updates 100, 500, 1,000, and 2,000 on 50 deterministic validation batches.
- Gate: update 2,000 only on 500 deterministic validation batches.
- Curves cannot select checkpoints, budgets, thresholds, families, or examples.
- Config: `configs/pff_trajectory_forecast_type_heads_v1.yaml`
- Config SHA-256: `d9ac4ebf171da48da6d4187a17244a2e12b6d0aac87f802c39326fbdb3d6f775`

## Gate A: Downstream Type-Head Value

Gate A is evaluated separately for `raw`, `frozen`, and `finetuned` against the prior shared-head
entity raw family. A family passes only if every condition holds:

1. Player ADE is no more than 1% worse in at least 2 of 3 matching seeds.
2. Mean player ADE is no more than 1% worse.
3. Mean player error is no more than 1% worse at any horizon.
4. Ball-ADE seed wins occur in at least 2 of 3 matching seeds.
5. Mean ball ADE improves by at least 5%.
6. Mean ball ADE is no worse than the prior global raw value of 7.2098667870 m.
7. Mean ball 4-second FDE does not worsen.

The player rules test retention of the already observed entity-token benefit. The ball rules test
whether specialization repairs the sole blocker from the prior redesign.

## Gate B: Pretrained Representation Value

Either transferred family passes Gate B only if all conditions hold against type-head `raw`:

1. Player-ADE seed wins in at least 2 of 3 matching seeds.
2. Mean player ADE improves by at least 2%.
3. Ball-ADE seed wins in at least 2 of 3 matching seeds.
4. Mean ball ADE improves by at least 5%.
5. Mean player error is not more than 1% worse at any horizon.

Families passing Gate A are downstream-eligible. Among them, the lowest mean player-ADE family is
operational, with lower mean ball ADE and then family order (`raw`, `frozen`, `finetuned`) breaking
exact ties. A transferred family supports a representation-value claim only if it also passes Gate
B. If no family passes Gate A, `global_raw` remains operational. The PFF test split remains sealed
regardless of outcome.

## Preflight Exclusion

Ten-update runs for all three families on one training match confirmed shapes, checkpoint loading,
gradient boundaries, and run-manifest labels. They used `validation_split: train`; their metrics
are excluded from every gate and interpretation.
