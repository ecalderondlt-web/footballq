# PFF Multi-Horizon Trajectory Forecast Protocol V1

Status: frozen before any validation metric, baseline, or model result was computed.

## Question

Does the selected real-tracking TD-JEPA backbone improve downstream PFF player-trajectory
forecasting relative to the same Transformer trained from a random initialization?

This is a downstream kinematic-value test. It is not a tactical-concept, possession,
availability, or semantic-understanding test.

## Frozen Data Boundary

- Split: `splits/pff_wc2022_64match_inductive_v1.json`
- Split payload SHA-256: `bee86c2c8f917a52a007dbdb92f6082a50f6a33fd278cc8e3f52213cb31f381b`
- Split file SHA-256: `9f7d56184920e463f1aa5fdcee05dc9b59438184910afc93a7e0c12f4e322226`
- Assignment: 48 train / 8 validation / 8 sealed test matches.
- Forecast manifest: `data/processed/pff_wc2022_trajectory_forecast_v1/dataset_manifest.json`
- Forecast manifest payload SHA-256:
  `fe75d27de9680fd5ad6e975d34781b6eb2f216c30d63fb73ac1b73c641715ddd`
- Forecast manifest file SHA-256:
  `688761b30c4fbe38d832d09d459e79153acc5851a397ceb600d5bc30c811b537`
- Source TD manifest file SHA-256:
  `8adc7518253a537b25a180be9cd88336312ee6457017c21c42866da5087b8c0f`
- Prepared scope: 844,195 train and 141,054 validation contexts in 1,766 shards.
- Feature and visibility view: five-channel `position_only`, `observed_only`.
- Context: 1.0 second at 10 fps.
- Forecast endpoints: 0.5, 1.0, 2.0, and 4.0 seconds after the final context frame.
- Future support: target entity is observed at the endpoint and appears at least once in context.
- PFF events are not model inputs. They may only be used in a later descriptive slice analysis.

The forecast preparer and dataset reject `test` as an allowed split. No PFF test tracking tensor,
test target, test event shard, or test embedding may be generated or loaded in this study.

## Frozen Backbones

The operational scratch checkpoints from the completed 10,000-update tracking study are used:

| Seed | Checkpoint SHA-256 |
|---:|---|
| 7 | `267f907a9521fbec1ae31df11b36e931810d087d120b3d5822950c50f7aa7e9f` |
| 11 | `1cc4bdc6fa6e11912baffa1bee8322b95ad2c4dff3f64898f433f0e95b8ae4ff` |
| 23 | `ed6b3ad8b0de7b95ff693ccfe20e85012922ee86f128ed886469a02c1b37a366` |

## Frozen Comparisons

1. `last_position`: repeats the latest observed position.
2. `constant_velocity`: extrapolates the last-two-observation velocity.
3. `raw`: the matched Transformer and residual decoder trained end to end from random weights.
4. `frozen`: the selected pretrained Transformer is frozen; only the matched decoder trains.
5. `finetuned`: the selected pretrained Transformer and matched decoder train end to end.

All learned families use the same architecture, kinematic inputs, constant-velocity residual base,
decoder initialization per seed, optimizer, batches, and update budget. The only intended learned
family difference is encoder initialization and whether it is frozen.

## Frozen Training And Evaluation

- Seeds: 7, 11, 23.
- Updates: exactly 2,000 per learned family and seed.
- Batch size: 128.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`.
- Loss: masked mean endpoint displacement in metres over the four horizons.
- Validation curves: 100, 500, 1,000, and 2,000 updates on the first 50 deterministic batches.
- Final gate: update 2,000 only, on the first 500 deterministic validation batches.
- Curves cannot select a checkpoint, budget, family, threshold, or validation subset.
- The exact validation sample-order digest must match across every baseline and learned run.
- Config: `configs/pff_trajectory_forecast_v1.yaml`
- Config SHA-256: `b9e64d591c75269b5d1d4717b6a24b9ff51e662d5d55de4089544effad0359fc`

Primary metric: player ADE in metres, averaged over valid player endpoints at all four horizons.
Also report player error at each horizon, player 4-second FDE, all-entity metrics, and ball metrics.
Ball metrics are secondary because the ball is absent from many observed-only PFF frames.

## Frozen Gate

For either transferred family (`frozen` or `finetuned`) to pass, all conditions must hold:

1. It beats `raw` player ADE in at least 2 of 3 matching seeds.
2. Its three-seed mean player ADE improves at least 2% over `raw`.
3. Its mean player error is not more than 1% worse than `raw` at any frozen horizon.
4. Its three-seed mean player ADE improves at least 1% over constant velocity.

If both transferred families pass, the lower mean player ADE is operational. If neither passes,
the gate is blocked and the lower-error option among `raw` and constant velocity remains
operational. Last-position remains descriptive and cannot be selected over constant velocity.

## Preflight Exclusion

Ten-update runs on one training match were used only to test data alignment, checkpoint loading,
and gradient boundaries. They used `validation_split: train`; their losses and errors are excluded
from this protocol, gate, model selection, and scientific interpretation.

## Interpretation Limits

A pass would show downstream trajectory-forecast value under this fixed budget and validation
sample. A block would show that this representation does not add material forecasting value under
the frozen design. Neither outcome alone establishes tactical concepts or a complete world model.
The eight PFF test matches remain sealed for a separately declared one-time final evaluation.
