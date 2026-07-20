# PFF Entity-Preserving Trajectory Forecast Protocol V1

Status: frozen after train-only preflight and before any entity-token validation metric was
computed.

## Questions

1. Does preserving one contextual Transformer token per tracked entity improve the operational
   global-vector raw forecaster?
2. Within the entity-preserving architecture, do the selected pretrained tracking weights add
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

## Frozen Prior Result

The completed global-vector study is the external redesign baseline:

- Summary: `runs/pff_trajectory_forecast_v1/gate_summary.json`
- Summary SHA-256: `7905300e4d1784a1786c08920e528b54c93b3c82408c63b6f5defa7d233ec688`
- Artifact audit SHA-256:
  `8acd06a1d47d8f95c0d1461e0f3af9e13632eae323f68289fab2b109968b398d`
- Global raw mean player ADE: 1.5230249132 m.
- Global raw mean ball ADE: 7.2098667870 m.
- Global raw means are the average of matching seeds 7, 11, and 23 at update 2,000.

## Frozen Backbones

| Seed | Scratch tracking checkpoint SHA-256 |
|---:|---|
| 7 | `267f907a9521fbec1ae31df11b36e931810d087d120b3d5822950c50f7aa7e9f` |
| 11 | `1cc4bdc6fa6e11912baffa1bee8322b95ad2c4dff3f64898f433f0e95b8ae4ff` |
| 23 | `ed6b3ad8b0de7b95ff693ccfe20e85012922ee86f128ed886469a02c1b37a366` |

## Frozen Architecture

The existing Transformer output is reshaped to `[context, entity, d_model]`. Its contextual token
for each entity is averaged only across that entity's observed context frames. A single shared MLP
receives each entity token plus that entity's last position, last-two-point velocity, and
observation flag. It predicts four coordinate residuals over the same constant-velocity base.

- Representation mode: `entity_tokens`.
- Token dimension: 128.
- Shared decoder hidden dimension: 256.
- Total parameters: 520,456.
- Shared decoder parameters: 102,152.
- Prior global model total parameters: 900,280.
- Prior global decoder parameters: 481,976.

The redesign therefore has less capacity than the prior global model. It is an architecture test,
not a parameter-matched isolation of token layout.

Code hashes at protocol freeze:

- `src/footballq/models/soccer_state_encoder.py`:
  `e21dda98c0605841be9df2fbd20c7ffa4ced5c7d9be910ae782d1ff3453c2eb0`
- `src/footballq/models/trajectory_forecaster.py`:
  `1e48601f6cb86509735ea0907277ef2dc8d598316f425cac4fa7dd4c10796e59`
- `src/footballq/training/train_trajectory_forecast.py`:
  `68b9ca4d9073224d1ace0c991bb3ffdc6237c5bb878d87697d7c98679acbcb92`

## Frozen Families And Training

1. `raw`: random entity-token encoder and shared decoder, trained end to end.
2. `frozen`: selected pretrained encoder frozen; shared decoder trained.
3. `finetuned`: selected pretrained encoder and shared decoder trained end to end.

- Seeds: 7, 11, 23.
- Updates: exactly 2,000 per family and seed.
- Batch size: 128.
- AdamW learning rate: `3e-4`; weight decay: `1e-4`.
- Loss: masked mean endpoint displacement in metres.
- Curves: updates 100, 500, 1,000, and 2,000 on 50 deterministic validation batches.
- Gate: update 2,000 only on 500 deterministic validation batches.
- Curves cannot select checkpoints, budgets, thresholds, families, or examples.
- Config: `configs/pff_trajectory_forecast_entity_v1.yaml`
- Config SHA-256: `8c0e56293b8130d916f85f15c6e0867abc23365dceb24ce38d25d55833b5b8a3`

## Gate A: Entity-Preserving Redesign

`entity raw` passes the redesign gate only if all conditions hold against prior `global raw`:

1. Player-ADE seed wins in at least 2 of 3 matching seeds.
2. Mean player ADE improves by at least 1%.
3. Ball-ADE seed wins in at least 2 of 3 matching seeds.
4. Mean ball ADE improves by at least 5%.
5. Mean player error is not more than 1% worse at any horizon.

## Gate B: Pretrained Representation Value

Either transferred family passes only if all conditions hold against `entity raw`:

1. Player-ADE seed wins in at least 2 of 3 matching seeds.
2. Mean player ADE improves by at least 2%.
3. Ball-ADE seed wins in at least 2 of 3 matching seeds.
4. Mean ball ADE improves by at least 5%.
5. Mean player error is not more than 1% worse at any horizon.

If a transferred family passes Gate B, the lower-player-ADE passing family is operational. If Gate
B blocks but Gate A passes, `entity raw` is operational. If both gates block, `global raw` remains
operational. The PFF test split remains sealed regardless of outcome.

## Preflight Exclusion

Ten-update runs on one training match confirmed shapes, checkpoint loading, and gradient
boundaries. They used `validation_split: train`; their metrics are excluded from every gate and
interpretation.
