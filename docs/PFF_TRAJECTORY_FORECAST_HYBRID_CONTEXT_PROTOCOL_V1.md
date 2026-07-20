# PFF Hybrid-Context Ball Forecast Protocol V1

Status: frozen after train-only preflight and before any hybrid-context validation metric was
computed.

## Question

Can one scratch-trained model retain the entity-token player gains while recovering the global
model's ball performance by giving only the ball decoder explicit last-state kinematics for all 23
entities?

This is a downstream kinematic test. It is not a test of tactical concepts, possession,
availability, or semantic understanding. Representation transfer is out of scope until the scratch
architecture passes.

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

## Frozen External References

The completed entity and global raw families provide component references. The completed local
type-head study is provenance for this follow-up and cannot be rescored or changed.

- Type-head summary: `runs/pff_trajectory_forecast_type_heads_v1/gate_summary.json`
- Type-head summary SHA-256:
  `bb145f9d195743f0d3b1d5d52b6f0d7fdf135a15deb08ea0eb198926e1b5573f`
- Type-head artifact audit SHA-256:
  `da12e4f4c57ab69ac5a623245eb2a303a63e16985b1a96872cd268ed461ba195`
- Entity summary SHA-256:
  `f4ed01a8aa1d470e66712e4b91c27fab1af02d692ac5c7477c2fd050843ece22`
- Global summary SHA-256:
  `7905300e4d1784a1786c08920e528b54c93b3c82408c63b6f5defa7d233ec688`
- Entity raw mean player ADE: 1.4624011306 m.
- Entity raw mean ball ADE: 7.6037602452 m.
- Global raw mean player ADE: 1.5230249132 m.
- Global raw mean ball ADE: 7.2098667870 m.
- Global raw mean ball 4-second FDE: 17.4703664792 m.
- Global raw mean all-entity ADE: 1.7571681641 m.
- All means average matching seeds 7, 11, and 23 at update 2,000.

## Frozen Backbones

The scratch encoder architecture is instantiated from these checkpoints' frozen configs, but no
checkpoint encoder weights are loaded.

| Seed | Tracking checkpoint SHA-256 |
|---:|---|
| 7 | `267f907a9521fbec1ae31df11b36e931810d087d120b3d5822950c50f7aa7e9f` |
| 11 | `1cc4bdc6fa6e11912baffa1bee8322b95ad2c4dff3f64898f433f0e95b8ae4ff` |
| 23 | `ed6b3ad8b0de7b95ff693ccfe20e85012922ee86f128ed886469a02c1b37a366` |

## Frozen Architecture

The player path is unchanged from the type-head study. Its shared MLP receives each player's
128-value contextual token plus that player's five last-state kinematic values. The ball MLP
receives the 128-value contextual ball token plus the flattened five-value last-state kinematics
for every canonical entity: ball at index 0 and players at indices 1-22.

- Representation mode: `entity_tokens`.
- Decoder mode: `player_global_ball`.
- Player input dimension: 133.
- Ball input dimension: 243.
- Hidden dimension per head: 256.
- Encoder parameters: 418,304.
- Player-head parameters: 102,152.
- Ball-head parameters: 130,312.
- Total parameters: 650,768.
- Prior global model parameters: 900,280.
- Loss remains masked mean endpoint displacement over all valid entity-horizon pairs.

Code hashes at protocol freeze:

- `src/footballq/models/soccer_state_encoder.py`:
  `e21dda98c0605841be9df2fbd20c7ffa4ced5c7d9be910ae782d1ff3453c2eb0`
- `src/footballq/models/trajectory_forecaster.py`:
  `d057fe6dacdc1f91762a7cdf74b2a6a107158faeb928c395d8d3aef59850a151`
- `src/footballq/training/train_trajectory_forecast.py`:
  `12440d2def20fb695baf26b0eb75a1844384854ede76aab989e8f91e02dfb4c7`
- Imported entity-runner helper code:
  `1e2b0fc53f538cabc8d2b7b25c2699612575f68f4530a3d4f58236f5a6a1ec87`

## Frozen Training

- Family: `raw` only.
- Seeds: 7, 11, 23.
- Updates: exactly 2,000 per seed.
- Batch size: 128.
- AdamW learning rate: `3e-4`; weight decay: `1e-4`.
- Curves: updates 100, 500, 1,000, and 2,000 on 50 deterministic validation batches.
- Gate: update 2,000 only on 500 deterministic validation batches.
- Curves cannot select checkpoints, budgets, thresholds, or examples.
- Config: `configs/pff_trajectory_forecast_hybrid_context_v1.yaml`
- Config SHA-256: `bb5b55da1e80b22e828df421ca9782753349ad33b75eb1db15273c73b83a0d63`

## Frozen Gate

The hybrid family passes only if every condition holds:

1. Player ADE is no more than 1% worse than matching-seed entity raw in at least 2 of 3 seeds.
2. Mean player ADE is no more than 1% worse than entity raw.
3. Mean player error is no more than 1% worse than entity raw at any horizon.
4. Ball ADE is no more than 1% worse than matching-seed global raw in at least 2 of 3 seeds.
5. Mean ball ADE improves by at least 4% over entity raw.
6. Mean ball ADE is no more than 1% worse than global raw.
7. Mean ball 4-second FDE is no more than 1% worse than global raw.
8. Mean ball error is no more than 2% worse than global raw at any horizon.
9. Mean all-entity ADE improves by at least 2% over global raw.

If all conditions pass, `hybrid_context_raw` becomes operational on validation and a separate
protocol may test representation transfer. Otherwise `global_raw` remains operational. The PFF
test split remains sealed regardless of outcome.

## Preflight Exclusion

One ten-update raw run on one training match confirmed shapes, all-entity ball input, gradients,
checkpoint-config loading, and run-manifest labels. It used `validation_split: train`; its metrics
are excluded from every gate and interpretation.
