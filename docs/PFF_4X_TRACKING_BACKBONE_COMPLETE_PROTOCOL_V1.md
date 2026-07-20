# PFF 4x Tracking-Backbone Complete Training Protocol V1

Status: frozen before any 10,000-update run on 2026-07-16.

## Purpose

Determine whether the validation advantage from 4x GRF initialization persists after substantially
more real tracking training, rather than disappearing as scratch catches up. This is the complete
training gate for the position-only tracking backbone. It is not the semantic event-model phase and
does not establish tactical understanding.

## Frozen Inputs

- PFF tensor manifest:
  `data/processed/pff_wc2022_td_jepa_position_only_train_val_v1/observed_only/dataset_manifest.json`
- manifest file SHA-256:
  `8adc7518253a537b25a180be9cd88336312ee6457017c21c42866da5087b8c0f`
- manifest payload SHA-256:
  `37acb8a6a00e4842a8aef8dce2700417fd7dfa24c827c3a9f46c7dac782c24ae`
- split manifest: `splits/pff_wc2022_64match_inductive_v1.json`
- split file SHA-256:
  `9f7d56184920e463f1aa5fdcee05dc9b59438184910afc93a7e0c12f4e322226`
- population: 844,195 train examples and 141,054 validation examples
- projected tensor shards: 1,505 train, 261 validation, zero test
- seeds: 7, 11, and 23

The paired 4x initializations are the fixed synthetic `latest.pt` checkpoints:

| Seed | Checkpoint SHA-256 |
| ---: | --- |
| 7 | `0dd86ea4aa21f197ea7c6d2d42c1cb2c20f6c9eef0a688664138fa4abbbd827d` |
| 11 | `268f030e2b45afdd912c217f331b7391295970c3b34db392395537ef19376930` |
| 23 | `7d9b238b2ae1ae24e39055e682d12beaa37c92cec98d24768b94377844949427` |

## Frozen Training

The families are scratch and 4x GRF initialization. Both use identical architecture, losses,
optimizer, batch ordering logic, seed, and PFF data view. Initialization transfers model weights
only; each PFF run creates a fresh AdamW optimizer.

- feature view: `position_only`
- objective: `future_nonoverlap_context_only`
- context: 1 second at 10 fps
- prediction gap: 1 second
- target: separate 1-second future context
- batch size: 128
- fixed budget: 10,000 optimizer updates
- epoch ceiling: 3, allowing the global fixed budget to finish
- approximate exposure: 1.52 passes over the 844,195-example train population
- validation curve: updates 100, 250, 500, 1,000, 2,000, 5,000, and 10,000
- curve evaluation: first 50 validation batches, descriptive only
- final gate evaluation: 500 validation batches at update 10,000
- checkpoint selection: none; use fixed-budget `latest.pt`
- embedding export: disabled

No validation result changes the update budget, checkpoint, loss, architecture, or seed set.

## Frozen Persistence Gate

The 4x persistence hypothesis passes only when all conditions hold:

1. all six final validation runs are finite
2. 4x has lower final total loss than paired scratch in at least two of three seeds
3. 4x lowers mean final total loss by at least 1% relative to scratch
4. 4x mean narrow TD loss does not exceed scratch mean narrow TD loss
5. every final `z_online_std_mean` exceeds 0.05
6. every run loads exactly train and validation tensors, never test
7. every curve has the seven frozen update points and every final evaluation is update 10,000 over
   64,000 validation examples

If the gate passes, 4x is the operational tracking-backbone family. If it is blocked but remains
lower on both mean total and narrow TD loss, it may remain an engineering initialization but no
persistent-benefit claim is permitted. If scratch is lower, scratch becomes the operational family.
All three seeds are retained; no single seed is selected from validation.

## Semantic Phase Boundary

No local StatsBomb or Wyscout event files were present when this protocol was frozen. Event streams
cannot be inserted directly into coordinate tensors. They require a separate event encoder and an
explicit alignment or auxiliary-objective contract. That phase follows this backbone gate and must
receive its own data manifest, split, and frozen validation protocol before training.

## Test And Claim Boundary

The eight PFF test matches remain sealed. This study uses validation for model-family selection and
is not an independent confirmation. It cannot establish tactical concepts, semantic understanding,
downstream value, or final test performance.
