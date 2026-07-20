# PFF StatsBomb Context Residual Protocol V1

Status: frozen before PFF validation event preparation or result-bearing runs.

## Question

Does a frozen StatsBomb event-sequence representation add repeatable predictive value to the
selected real-tracking backbone when both are aligned only through causal PFF events from the same
match and period?

This study measures future tracking-latent prediction. It does not test tactical understanding,
semantic understanding, player intent, match outcome, or cross-provider timestamp alignment.

## Frozen Sources

- tracking source: the final seed-matched `scratch` checkpoints selected by
  `docs/PFF_4X_TRACKING_BACKBONE_COMPLETE_PROTOCOL_V1.md`
- event source: the final seed-matched `event_only` checkpoints selected by
  `docs/STATSBOMB_SEMANTIC_PRETRAIN_PROTOCOL_V1.md`
- PFF split: `splits/pff_wc2022_64match_inductive_v1.json`
- tracking tensors:
  `data/processed/pff_wc2022_td_jepa_position_only_train_val_v1/observed_only/dataset_manifest.json`
- StatsBomb tensors: `data/processed/statsbomb_event_sequence_v1/manifest.json`
- PFF train-only mapping audit:
  `runs/integrity/pff_statsbomb_event_context_v1_train_audit.json`

The train-only audit was completed before this freeze. It retained 71,300 PFF events from all 48
training matches. Exactly 66,725 were mapped by an explicit provider table, 4,575 remained the
StatsBomb unknown token, and 60,778 generic `OTB` interval markers were excluded. Unknown labels
must not be guessed after validation is prepared.

## Causal Alignment

Each tracking example is identified by `(match_id, period, frame_t)`. Its event history contains at
most 32 PFF events from the same match and period whose event frame is no later than the last frame
of the observed tracking context. Target frames, later events, other matches, StatsBomb matches,
and PFF test matches are unavailable to the join.

PFF provider labels are mapped as follows:

| PFF label | StatsBomb event type |
| --- | --- |
| `PA` | Pass |
| `CH` | Duel |
| `BC` | Carry |
| `CL` | Clearance |
| `CR` | Pass |
| `RE` | Ball Receipt* |
| `SH` | Shot |
| `FIRSTKICKOFF`, `SECONDKICKOFF` | Half Start |
| `END` | Half End |
| `SUB` | Substitution |
| `OFF` | Player Off |
| `ON` | Player On |

All other retained labels use the frozen unknown token. `OTB` is excluded because it is a generic
interval wrapper that often duplicates a possession event.

## Model Families

All families use the same frozen seed-matched tracking checkpoint and the same 66,432-parameter
residual head. Only that head is trainable.

1. `tracking`: zero event vector; the head can learn only a global residual correction.
2. `raw`: deterministic 128-dimensional last-event, event-frequency, coverage, unknown-rate, and
   recency summary from the same history.
3. `random`: the same StatsBomb encoder architecture with frozen deterministic random weights.
4. `pretrained`: the selected frozen StatsBomb `event_only` checkpoint.

No family may fine-tune the tracking backbone or event encoder. No embeddings are exported.

## Matched Training

- seeds: 7, 11, and 23
- batch size: 128
- updates: 2,000
- optimizer: AdamW
- learning rate: 0.001
- weight decay: 0.0001
- validation curve: updates 100, 500, 1,000, and 2,000 over 50 batches
- final validation: first 500 deterministic validation batches, exactly 64,000 examples
- selection: final checkpoint only; no best-seed or best-step replacement
- objective: normalized latent TD loss against the frozen tracking target encoder

The four families must see matching training order for a seed. Source checkpoints are paired by
seed. Validation is never used to change mappings, architecture, optimization, update count, or
thresholds.

## Frozen Gate

Lower TD loss is better. Relative improvement is `(reference - candidate) / reference`.

Integrity requirements:

- all 12 runs finish at update 2,000 with finite metrics
- all runs evaluate exactly 64,000 validation examples
- event-history example counts match across families within each seed
- frozen base tracking TD loss matches across families within each seed to absolute tolerance
  `1e-10`
- every run manifest records only train and validation tensors, no test load, and no embedding
  export

Representation-value requirements for `pretrained`:

- beats `tracking` in at least 2 of 3 seeds and improves mean TD loss by at least 1%
- beats `raw` in at least 2 of 3 seeds and improves mean TD loss by at least 0.5%
- beats `random` in at least 2 of 3 seeds and improves mean TD loss by at least 1%
- correct event context beats its event-ablated evaluation in at least 2 of 3 seeds and by at least
  1% on the mean

Operational selection:

- select `pretrained` only if every integrity and representation-value requirement passes
- otherwise select `raw` only if it beats `tracking` in at least 2 seeds and improves mean TD loss
  by at least 1%
- otherwise retain `tracking`

A blocked gate means the tested event-context route is not accepted for the operational model. It
does not invalidate the separate StatsBomb next-event result or prove that event information is
useless.

## Access Boundary

After this freeze, validation event shards may be prepared using the unchanged mapping and audited
for structural integrity. PFF test files and tensors remain unavailable. Result-bearing runs may
load only PFF train/validation tracking tensors, PFF train/validation event tensors, the frozen
StatsBomb vocabulary/weights, and the paired frozen tracking checkpoint.
