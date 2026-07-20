# StatsBomb Semantic Event Pretraining Protocol V1

Status: frozen before any result-bearing 5,700-update run on 2026-07-16.

## Purpose

This study tests whether sparse StatsBomb 360 geometry adds validation value to a separately
trained causal event-sequence encoder. It does not join StatsBomb matches to PFF tracking, modify
the scratch tracking backbone, or claim tactical or semantic understanding from event-prediction
loss alone.

The matched feature views are:

- `event_only`: event categories and continuous event features; all 360-derived continuous fields
  are zeroed and the matched geometry branch receives an always-missing geometry token.
- `event_plus_360`: the same architecture and event inputs plus a permutation-invariant set encoder
  over available StatsBomb 360 freeze frames.

Both views have identical parameter names and shapes. Missing 360 is explicit and is never encoded
as an empty defensive shape.

## Frozen Source

- upstream repository: `https://github.com/statsbomb/open-data`
- upstream commit: `b0bc9f22dd77c206ddedc1d742893b3bbe64baec`
- archive SHA-256: `924e21e17f97ddf6149944bcfe6a8ec7ef42361dea45b76f8bbd8cb4cbd134cb`
- source-manifest payload SHA-256:
  `9166425ddd878e2943e4b1cc9f583892bff264bf03f0bcfd0186b5fdcba7cda8`
- train-only schema-audit payload SHA-256:
  `65165efa5c4d6e109b9663cef66765bc212698abdd7430406727b5e901ab5296`
- train-derived vocabulary payload SHA-256:
  `4274448fe6d9ba3712957c393fd171af4b15d930c787748cff32f202eeb61a90`

The snapshot contains 4,235 event/lineup matches. Match metadata exists for 3,961; 274 event and
lineup pairs are explicitly marked metadata-orphaned. There are 426 matches with a 360 file.
Metadata availability and 360 presence are split strata; neither score nor event outcome is used
to assign matches.

## Frozen Split

The immutable match split is
`splits/statsbomb_open_data_b0bc9f2_match_inductive_v1.json`.

- canonical split SHA-256: `3e6253ff4f7c303442eb66372dd6f79e7735e4a6559ca050e2967b874925eb4d`
- train: 3,388 matches
- validation: 425 matches
- test: 422 matches
- protocol: match-inductive, stratified by 360 and match-metadata availability

All vocabularies are fit on training events only. Missing and validation-unknown category indices
are separate. The finalized validation tensors contain zero unknown categorical IDs.

## Frozen Event Tensors

The finalized manifest is `data/processed/statsbomb_event_sequence_v1/manifest.json`.

- file SHA-256: `c3076ddbee491a98bfbac0db360863b4f9be5842b776cf512f8f2cd4e3167772`
- payload SHA-256: `95a0a4d1ea1e4b10927c21227139610a91c6f33a3e7d5201b13eb5ba961c7b87`
- tensor-audit payload SHA-256:
  `f14d053a83a9b23c25a361f1ff8ca77396eb95a3b8ac09f80dc8c3606b21319f`
- train: 11,890,025 events and 739,046 causal windows
- validation: 1,503,962 events and 93,479 causal windows
- test: no tensors and no windows

Each input contains 32 events and predicts the next event at every position. Windows use stride 16
and never cross a match or period boundary. The five categorical fields are event type, play
pattern, player position, event subtype, and outcome. Seventeen continuous fields preserve
locations, endpoint locations, presence/in-bounds masks, duration, inter-event time, possession
change, pressure flags, period, and optional 360 availability/visible-area values.

Freeze frames contain at most the train-observed maximum of 22 players. Each visible player has
normalized clipped coordinates plus teammate, actor, goalkeeper, and coordinate-in-bounds flags.
The set encoder is permutation-invariant.

One malformed training 360 file is treated as unavailable geometry while retaining its events.
Across train and validation, 17,425 stale 360 rows do not match current event UUIDs and are dropped;
1,224,970 rows join successfully. These conditions are fixed before training.

## Raw Controls

Raw controls are fit on training transitions and scored on all validation windows with Laplace
alpha 1.0. The frozen report is `runs/integrity/statsbomb_event_baselines_v1.json`, payload SHA-256
`611a3051aceb5ff4e3f496309496811f10bab4ad29c054b25f3d9df20918c30d`.

- validation event targets: 2,991,328
- 360-anchored validation event targets: 278,803
- global-frequency event-type NLL: 1.944380
- first-order Markov event-type NLL: 1.020508
- 360-anchored first-order Markov NLL: 0.907201
- copy-current-location MAE: 0.179904
- 360-anchored copy-current-location MAE: 0.157832

The first-order Markov predictor is the representation-value control. A neural score that does not
beat it is not evidence of useful higher-order event context.

## Matched Training

The frozen configurations are:

- `configs/statsbomb_event_only_pretrain_v1.yaml`, SHA-256
  `248e5dd65fce6ca061de867cc15c721c617c63e85551bf1ad3bb14ffe84c3751`
- `configs/statsbomb_event_plus_360_pretrain_v1.yaml`, SHA-256
  `2d6b2d064ff4a4dbf563fa837eea4bda20300ab91802f836e51ef2afe69fbdbb`

For each view:

- seeds: 7, 11, and 23
- sequence length: 32
- categorical embedding width: 24
- model width: 128
- causal Transformer: 3 layers and 4 heads
- dropout: 0.1
- batch size: 128 windows
- optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`
- gradient clipping: 1.0
- objective: next-event-type cross-entropy plus next-location Smooth L1 with weight 1.0
- fixed training endpoint: 5,700 updates, or 729,600 windows per run
- descriptive curves: updates 100, 500, 1,000, 2,500, and 5,700 over 50 validation batches
- final decision: all 93,479 validation windows at update 5,700

No best checkpoint, individual seed, curve point, or validation subset replaces the fixed final
endpoint. All three seeds are retained.

## Frozen Gate

All conditions must pass:

1. all six runs are finite, end at exactly 5,700 updates, and record all five curve points;
2. every run loads exactly train and validation, never test, and exports no embeddings;
3. mean `event_only` final event-type NLL improves at least 1% over the first-order Markov NLL;
4. `event_plus_360` wins anchored event-type NLL in at least two of three paired seeds;
5. mean anchored event-type NLL improves at least 1% versus `event_only`;
6. `event_plus_360` wins anchored next-location MAE in at least two of three seeds;
7. mean anchored next-location MAE improves at least 1% versus `event_only`;
8. mean overall event-type NLL and location MAE each worsen by no more than 1% with 360;
9. every final evaluation covers exactly 93,479 windows and 2,991,328 event targets, with a
   positive anchored target count.

If all conditions pass, `event_plus_360` becomes the operational semantic encoder. If only the
Markov criterion passes, `event_only` remains the operational semantic encoder and 360 integration
is blocked. If the Markov criterion fails, neither neural family is selected for cross-modal
integration.

## Test And Claim Boundary

No StatsBomb test event file is tensorized, trained on, evaluated, or used for model selection. Raw
source hashing reads bytes from every file. Before this protocol was frozen, a one-time structural
JSON syntax check parsed the 42 test-partition 360 files and retained only the fact that none was
malformed; no event values, distributions, labels, or metrics were retained or used. This means
the StatsBomb test partition is not literally byte-untouched, but it remains outcome- and
metric-sealed.

The PFF test split remains untouched by this phase. StatsBomb and PFF matches are not cross-provider
aligned, so no timestamp join or frame-level fusion is permitted. Passing this gate supports an
event-representation engineering choice only. It does not establish tactical concepts, semantic
understanding, tactical surprise, downstream tracking value, or paper-level claims.
