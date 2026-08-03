# RLCS Identity-Matchup Experiment V1

## Decision and claim boundary

This branch redirects the next identity-conditioned proof to native RLCS 2025 replay telemetry.
It tests whether player and opponent identities add chronologically held-out predictive signal
after complete geometry is known. It does **not** claim that Rocket League weights transfer to
football, that a football coach model exists, or that generic motion realism is tactical insight.

The primary outcome is factorized next-touch entity plus next-touch zone NLL on critical,
all-identities-known test states. Full identity conditioning must beat anonymous, actor-only,
roster-only, and identity-shuffled controls under official-series blocking.

**Final V1 status (August 2, 2026): validation gate failed; the test split remains sealed.**
All 12 validation-only runs completed, but full identity conditioning was worse than anonymous in
all three seeds. No test unlock was created, no test inference was run, and no claim of an
identity-conditioned advantage is supported by V1.

## Local viability result

Checked before corpus acquisition on August 1, 2026 (America/Mexico_City):

| Resource | Result |
|---|---:|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| GPU memory | 12,227 MiB |
| System memory | 31.43 GiB |
| Free workspace drive | 163.94 GiB |
| PyTorch / CUDA | 2.11.0+cu128 / available, BF16 supported |
| Representative batch 256 throughput | 5,960.93 samples/s |
| Representative peak VRAM | 0.70 GiB |
| Native sample replay parse | 10,914 frames and 693 events in about 1.7 s |

The device passes the compute gate. The final 1.48M-parameter model is small relative to the
available GPU. Acquisition/parsing and all 12 matched runs are intended to run locally first.

The train-only 5,000-sample memorization preflight subsequently reached factorized joint NLL
0.066389 (required at most 0.10) at step 1,000 in 183.3 seconds using CUDA BF16. The first
scientific run exposed a CPU-bound Python loop in reflection augmentation. Replacing that loop
with the algebraically identical closed-form matrix conjugation reduced time to the first
validation checkpoint from roughly ten minutes to roughly two while preserving the augmentation
definition and passing an explicit equivalence test. Peak scientific-run use remained about
2.6 GiB VRAM and 7.5 GiB private system memory. The local device was therefore viable.

Ballchasing authentication succeeded on August 1, 2026. The official inventory is now frozen at
1,595 unique replay IDs and the 100-replay local feasibility gate has passed. Full acquisition is
resumable under the persisted 200-download rolling hourly cap.

## Reproducible environment

The dependency solution is frozen in `uv.lock`, generated with `uv==0.11.28`. The locally tested
runtime pins PyTorch 2.11.0 and `analyzerl-parser==1.0.5`. AnalyzerL replaces the original plan's
Ubuntu-only parser assumption because its current ABI3 wheel works with this Windows/Python
device and exports both exact native frames and play-by-play events.

```powershell
uv sync --extra rlcs --extra dev
```

Every matchup training manifest records the SHA-256 of `uv.lock`.

## Day 1: token, inventory, and 100-replay feasibility gate

Generate an API token at Ballchasing, then either set it in the current environment:

```powershell
$env:BALLCHASING_TOKEN = "<token>"

# Or: Copy-Item .env.example .env, then edit .env locally.

uv run python scripts/acquire_rlcs.py `
  --groups `
    europe-bizraz5v3p `
    north-america-dxx4sk6tc3 `
    europe-63apu41301 `
    north-america-j9hk5zna34 `
  --output data/raw/rlcs_2025 `
  --page-size 200 `
  --download-rps 1 `
  --download-hourly-cap 200 `
  --download-limit 100 `
  --resume

uv run python scripts/build_rlcs_dataset.py `
  --raw data/raw/rlcs_2025 `
  --output data/processed/rlcs_identity_matchup_v1 `
  --fps 10 `
  --context-seconds 2.0 `
  --min-next-touch-dt 0.20 `
  --max-next-touch-dt 4.00 `
  --exclude-goal-reset-seconds 2.0 `
  --require-standard-3v3 `
  --parse-limit 100 `
  --audit-only
```

The `.env` option is useful when Codex was launched before the environment variable was set.
It is ignored by Git. Never paste the token into chat, command output, or committed files.

Stop before further download if API access exceeds the cap, parser success is below 95%,
standard 3v3 is unavailable, or unresolved replay identities exceed 2%. The parser cache and
inventory are resumable.

### Pilot protocol amendment (before full-corpus processing)

The 100-replay QC-only pilot exposed two parser-adapter issues before any model dataset, training,
validation result, or test result existed:

1. AnalyzerL 1.0.5 can emit both a classified goal and a synthesized official-goal row for one
   scoring play, so its exported cumulative score columns can increment twice. The adapter now
   reconstructs scores from deduplicated chronological goal clusters and fails closed when the
   reconstructed final score disagrees with the Ballchasing inventory.
2. The original 600-second gameplay ceiling rejected a structurally valid 647-second overtime
   replay (continuous frames, standard map, 3v3 roster, coherent seven-goal sequence). The
   corruption guard is amended to 1,800 seconds; all other duration and structural gates remain.

Cached native frame/event tables retain their parser provenance, but QC is recomputed whenever the
adapter or thresholds change. These amendments were made from acquisition/QC evidence only, before
constructing decision samples or inspecting model outcomes.

### Pilot result

| Gate | Result |
|---|---:|
| Frozen inventory | 1,595 unique replay IDs |
| Downloaded files with matching size and SHA-256 | 100/100 |
| Parser success | 100/100 (100%) |
| Standard 3v3 with six player slots | 100/100 (100%) |
| Strict QC accepted | 95/100 (95%) |
| Identity-resolved among QC-accepted | 95/95 (100%) |
| Unresolved identity replay rate | 0% |

Five replays are quarantined as `score_incoherent` because deduplicated chronological goal clusters
do not reproduce the Ballchasing final score. They are not repaired heuristically and cannot enter
the scientific dataset. The gate decision is **pass**: continue the prespecified full corpus locally.

## Full corpus and data gates

Resume acquisition without `--download-limit`, then build the full dataset:

```powershell
uv run python scripts/acquire_rlcs.py `
  --groups `
    europe-bizraz5v3p `
    north-america-dxx4sk6tc3 `
    europe-63apu41301 `
    north-america-j9hk5zna34 `
  --output data/raw/rlcs_2025 `
  --page-size 200 `
  --download-rps 1 `
  --download-hourly-cap 200 `
  --resume

uv run python scripts/build_rlcs_dataset.py `
  --raw data/raw/rlcs_2025 `
  --output data/processed/rlcs_identity_matchup_v1 `
  --fps 10 `
  --context-seconds 2.0 `
  --min-next-touch-dt 0.20 `
  --max-next-touch-dt 4.00 `
  --exclude-goal-reset-seconds 2.0 `
  --require-standard-3v3

uv run python scripts/audit_rlcs_identity.py
```

The audit fails unless the preregistered gates pass: 1,595 hashed files, at least 1,400 accepted
replays, 75,000 clean decisions, 10,000 critical test decisions, 80 test series, 30 players with
20 earlier games, 10 recurring matchups, and 60% all-known primary coverage. Do not train around
a failed gate.

The builder converts the checked-in selector template into a frozen replay-ID manifest only after
the immutable inventory exists. Training rejects selector placeholders. Train is all Split 1 EU
and NA; validation is Split 2 Regional 1; test is sealed Split 2 Regionals 2 and 3.

### Full-corpus result

| Corpus/data gate | Observed | Required | Result |
|---|---:|---:|---:|
| Downloaded files with matching hashes | 1,595 | 1,595 | pass |
| Parser success | 100% | at least 95% | pass |
| Strict-QC and identity-accepted replays | 1,445 | at least 1,400 | pass |
| Unresolved identity replay rate | 0% | at most 2% | pass |
| Clean decision samples | 117,704 | at least 75,000 | pass |
| Train / validation / sealed-test samples | 63,458 / 18,879 / 35,367 | frozen | pass |
| Critical sealed-test samples | 11,142 | at least 10,000 | pass |
| Critical all-known test coverage | 69.11% | at least 60% | pass |
| Test series | 107 | at least 80 | pass |
| Players with at least 20 earlier games | 90 | at least 30 | pass |
| Repeated train-test matchups | 14 | at least 10 | pass |

The frozen dataset-manifest SHA-256 is
`6bd2af5e40842ed1ce96ad4466e4ac155e8b6bd953c8c96402af9af6bbdb9760`.
Its split Parquet hashes are recorded in `dataset_manifest.json`; the sealed-test Parquet was used
only for predeclared volume/coverage metadata gates and was never loaded by training or validation.

### Identity and power audits

The metadata-only collision audit found 28 native platform IDs associated with more than one exact
display handle. Seventy-four platform-scoped, date-bounded alias rows were manually frozen using
native platform ID plus official-series roster, teammate, team, and date continuity. Outcomes were
not used for resolution. After review, all 1,445 QC-accepted replays resolved and the unresolved
rate was 0%. The checked-in registry is `provenance/rlcs_identity_aliases_v1.csv`.

The preregistered series-level power gate also passed before training. A conservative
validation-series bootstrap/sign-flip simulation estimated power 1.0 for a 5% NLL effect at
alpha 0.01 (required at least 0.8), using 46 validation series, 1,000 simulation trials, and
10,000 sign-flip permutations. The immutable audit and its input hashes are in
`data/processed/rlcs_identity_matchup_v1/identity_audit.json`.

## Matched training and sealed test

```powershell
uv run python scripts/smoke_rlcs_overfit.py

uv run python scripts/train_rlcs_matchup.py `
  --config configs/rlcs_identity_matchup_v1.yaml

uv run python scripts/summarize_rlcs_matchup.py `
  --config configs/rlcs_identity_matchup_v1.yaml `
  --run-root runs/rlcs_identity_matchup_v1 `
  --write-unlock runs/rlcs_identity_matchup_v1/test_unlock.json
```

The summarizer refuses to create an unlock unless full beats anonymous by at least 2% validation
NLL in two of seeds 17, 23, and 41, and all 12 condition/seed checkpoints share one dataset hash.
If the gate fails, stop without loading test data.

### Validation result and terminal decision

All conditions used the same 63,458 train samples, 18,879 validation samples, batch order seed,
augmentation rule, optimizer schedule, model capacity, and validation sample-ID digest. Each row
below is the validation-selected checkpoint NLL; positive lift would mean that full is better.

| Seed | Anonymous | Actor-only | Roster-only | Full | Full vs anonymous |
|---:|---:|---:|---:|---:|---:|
| 17 | 2.712673 | 2.725091 | 2.733312 | 2.730788 | -0.668% |
| 23 | 2.706329 | 2.721318 | 2.726149 | 2.726284 | -0.737% |
| 41 | 2.727498 | 2.736550 | 2.742594 | 2.739154 | -0.427% |

Passing seeds: **0 of 3**; required: **2 of 3**. The validation gate therefore failed. The frozen
summary is `runs/rlcs_identity_matchup_v1/validation_summary.json` (SHA-256
`7c7d98e61fa6f5b63ff3e04e4df172bf3c4ab50ccf6f3c0f1ace14b6cb958169`). This path is intentionally
ignored by Git because it points to local run artifacts; the durable machine-readable ledger,
including all checkpoint and run-manifest hashes, is
`provenance/rlcs_identity_matchup_v1_validation.json`. The 12 run manifests each record
`test_loaded: false`.

For V1, the evaluation commands below are retained as protocol documentation only and **must not
be run**. `test_unlock.json` does not exist. Identity-shuffle, test bootstrap, and test sign-flip
controls are correctly skipped because they require the sealed evaluation and cannot rescue a
failed validation gate.

After the unlock exists, perform the one sealed evaluation:

```powershell
uv run python scripts/eval_rlcs_matchup.py `
  --config configs/rlcs_identity_matchup_v1.yaml `
  --unlock runs/rlcs_identity_matchup_v1/test_unlock.json `
  --output runs/rlcs_identity_matchup_v1/sealed_test

uv run python scripts/summarize_rlcs_matchup.py `
  --test-results runs/rlcs_identity_matchup_v1/sealed_test/test_results.json
```

Evaluation consumes the unlock with an exclusive receipt, ensembles three seeds, runs 20 fixed
within-roster and matched-opponent permutations, blocks inference by official series, applies
10,000 sign flips and 10,000 BCa series bootstraps, and Holm-adjusts the three main comparisons.

## Implemented safeguards

- Native replay inventory is written before bytes and records SHA-256, size, status, and time.
- Handles use exact NFKC/casefold/whitespace normalization; fuzzy aliases are forbidden.
- Alias rows affect resolution only when reviewed as `approved` or `frozen`.
- Platform/handle collisions are blocked by metadata-only audit; outcomes never resolve identity.
- Only training replays define the player vocabulary, feature statistics, and model identities.
- Context selection is at-or-before each 10 Hz grid time and rejects gaps, reused frames,
  goal/kickoff crossings, and parser-segment crossings.
- Identity ablations alter only six identity indices; geometry, clock, score, batches, seed, and
  stopping rule remain matched.
- Ordinary training code raises before it can open the test Parquet.

## External references

- Ballchasing API and rate limits: <https://ballchasing.com/doc/api>
- AnalyzerL parser package: <https://pypi.org/project/analyzerl-parser/>
