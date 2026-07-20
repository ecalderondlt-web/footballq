# PFF-GRF Sampler Comparison Protocol V1

Status: frozen before candidate training on 2026-07-13.

## Purpose

Test whether source-count-derived scenario-aware sampling makes the balanced GRF V2 dataset a
better initializer for observed-only PFF validation than its existing natural-frequency sampler.

This is an exploratory validation-only study. The natural-sampling baseline results were known
before candidate training. The candidate sampler and thresholds are fixed from synthetic source
counts, not chosen from PFF labels or candidate outcomes. Any apparent pass would require an
independent repeat before a confirmatory claim.

## Frozen Data And Samplers

Both families use the same balanced V2 dataset manifest with payload SHA-256
`f8d524157196ab007c21c0662a15b18e9458b2468b5a2af349895cec4bb981db`, the same PFF-train-only
visibility profile, and the same geometry-only future-nonoverlap objective.

Baseline sampler:

- natural example frequency with shard-grouped reads
- synthetic runs: seed 7 `20260713_150316`, seed 11 `20260713_150347`, seed 23 `20260713_150420`
- PFF runs: seed 7 `20260713_150747`, seed 11 `20260713_151244`, seed 23 `20260713_151826`

Candidate sampler:

- per-shard probability mass proportional to `source_example_count ** 0.5`
- fixed sample budget: 33,664 examples, exactly 263 batches of 128
- deterministic largest-remainder allocation followed by seeded within-shard permutations
- config: `configs/td_jepa_gfootball_v2_sqrt_sampler_matched_v1.yaml`
- allocation plan: `runs/integrity/gfootball_v2_sqrt_sampler_v1_allocation_plan.json`
- allocation-plan file SHA-256:
  `050a2782d68d7935e5348c428aa714a5b8499b93f17c28b47686b76cb8368a97`

The natural source contains 88.5% full-match windows and 11.5% academy windows. The frozen
square-root allocation assigns 72.3% of samples to the four full-match/policy shards and 27.7% to
academy shards. It does not use outcomes, possession, events, tactical labels, or PFF validation
metrics. The smallest pass-and-shoot shard receives 250 of 33,664 samples, so its high reuse is
bounded to 0.74% of the training budget.

## Frozen Training

- paired seeds: 7, 11, and 23
- candidate synthetic budget: exactly 263 optimizer updates; partial batches dropped
- transferred checkpoint: candidate `latest.pt` at update 263, not synthetic-validation `best.pt`
- baseline synthetic checkpoints: the matching natural-V2 `latest.pt` checkpoints listed above
- PFF config: `configs/td_jepa_pff_wc2022_grf_curriculum_comparison_v1.yaml`
- PFF data: observed-only manifest, 48 train / 8 validation / 8 untouched test matches
- PFF budget: 2,000 training batches and 500 validation batches per candidate run
- optimizer, architecture, loss, data order, and seed are paired with the frozen baseline

## Frozen Validation Gate

Square-root sampling passes this exploratory gate only when all conditions hold:

1. all six baseline/candidate PFF runs have finite validation metrics
2. the candidate has lower total validation loss in at least two of three paired seeds
3. the candidate lowers mean total validation loss by at least 2%
4. candidate mean narrow TD loss does not exceed baseline mean narrow TD loss
5. every run has validation `z_online_std_mean` above 0.05

A blocked result ends this sampler study without test access. An exploratory pass permits only an
independent validation repeat with the sampler unchanged. It does not permit PFF test access,
probes, discovery, interpretation, or tactical claims.

## Claim Boundary

This protocol tests the sampler, not whether any latent variable represents football tactics. A
blocked result means the fixed square-root reweighting did not demonstrate a stable validation
advantage under this compute budget. It does not rule out other prespecified synthetic objectives
or data-generation policies.
