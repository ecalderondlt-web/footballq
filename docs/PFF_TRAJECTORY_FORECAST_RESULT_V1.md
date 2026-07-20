# PFF Multi-Horizon Trajectory Forecast Result V1

Protocol: `docs/PFF_TRAJECTORY_FORECAST_PROTOCOL_V1.md`

Machine summary: `runs/pff_trajectory_forecast_v1/gate_summary.json`

Artifact audit: `runs/integrity/pff_trajectory_forecast_v1_artifact_audit.json`

## Outcome

The frozen downstream gate is **blocked**. The operational learned family is `raw`.

The frozen backbone beat the raw model in all three matching seeds, but its mean player-ADE gain
was only **0.758%**, below the prespecified **2%** materiality threshold. Fine-tuning won only one
seed and worsened mean player ADE by **0.356%** relative to raw.

This does not mean trajectory forecasting failed. It means the selected pretrained backbone did
not add enough downstream value over training the same forecasting architecture from scratch.

## Integrity Boundary

- Forecast examples: 844,195 train / 141,054 validation.
- Final validation sample: 64,000 examples, identical digest in every baseline and learned run.
- Forecast horizons: 0.5, 1.0, 2.0, and 4.0 seconds.
- Learned runs: 3 families x 3 seeds x 2,000 updates.
- Loaded splits: train and validation only.
- PFF test targets generated: no.
- PFF test tensors loaded: no.
- Embeddings exported: no.
- Data-lineage and final artifact audits: passed.

## Baselines

| Model | Player ADE (m) | Player 4 s FDE (m) | Ball ADE (m) | Ball 4 s FDE (m) |
|---|---:|---:|---:|---:|
| Last position | 3.3122 | 7.5812 | 9.5352 | 19.9686 |
| Constant velocity | 1.5631 | 5.1995 | 9.9756 | 27.0730 |

Constant velocity is a strong player baseline, especially at short horizons. It overshoots the
ball badly, where last position is better.

## Learned Means

| Family | Player ADE (m) | Player 4 s FDE (m) | Ball ADE (m) | Ball 4 s FDE (m) |
|---|---:|---:|---:|---:|
| Raw | **1.5230** | **4.9445** | **7.2099** | **17.4704** |
| Frozen | 1.5115 | 4.9609 | 9.6427 | 25.6866 |
| Fine-tuned | 1.5284 | 4.9741 | 7.6097 | 18.6374 |

Raw improves player ADE **2.561%** over constant velocity and ball ADE **27.725%**. It therefore
learns useful forecasting behavior even though representation transfer is blocked.

## Player Horizon Detail

| Family | 0.5 s (m) | 1.0 s (m) | 2.0 s (m) | 4.0 s (m) |
|---|---:|---:|---:|---:|
| Constant velocity | **0.1276** | **0.4365** | 1.5428 | 5.1995 |
| Raw | 0.1617 | 0.4613 | 1.5184 | **4.9445** |
| Frozen | 0.1391 | 0.4408 | **1.5070** | 4.9609 |
| Fine-tuned | 0.1666 | 0.4560 | 1.5169 | 4.9741 |

The frozen representation helps most at 0.5 and 1.0 seconds relative to raw, remains slightly
better at 2.0 seconds, and becomes 0.332% worse at 4.0 seconds. The gain is consistent but too
small in the aggregate to pass. Constant velocity remains strongest at the two shortest horizons.

## Frozen Gate Detail

### Frozen vs raw

- Seed wins: 3/3, passed.
- Mean player-ADE improvement: 0.758%, blocked below 2%.
- Worst horizon change: -0.332%, passed the -1% non-degradation floor.
- Mean player-ADE improvement vs constant velocity: 3.300%, passed.

### Fine-tuned vs raw

- Seed wins: 1/3, blocked below 2/3.
- Mean player-ADE improvement: -0.356%, blocked below 2%.
- Worst horizon change: -3.031% at 0.5 seconds, blocked below -1%.
- Mean player-ADE improvement vs constant velocity: 2.215%, passed.

## Interpretation

1. A learned multi-agent model adds modest player value beyond straight-line extrapolation and
   large ball value beyond that same baseline.
2. The selected global pooled backbone gives a reliable but narrow short-horizon player benefit.
   It does not provide a material overall advantage under this matched 2,000-update design.
3. Full fine-tuning does not preserve the small frozen benefit, so more optimization alone is not
   supported as the next move.
4. Frozen ball forecasting is poor relative to raw. A plausible, unproven explanation is that one
   global latent compresses away entity-specific detail needed by the decoder.

## Recommended Next Step

Do not open the PFF test split or scale this architecture yet. Run one train/validation-only,
prespecified entity-preserving decoder study: expose per-entity encoder tokens to the matched
forecast head, retain the same horizons/baselines/seeds, and test whether it fixes the 4-second and
ball weaknesses. A simple horizon hybrid using constant velocity at 0.5-1.0 seconds and the raw
model at 2.0-4.0 seconds is also a useful deployment baseline, but it is descriptive rather than
evidence for representation quality.

No tactical-concept or semantic-understanding claim is supported by this result.
