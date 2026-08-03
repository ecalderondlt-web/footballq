# RLCS Identity-Matchup V1: Simple Summary

## Bottom line

The experiment is finished under its original stopping rules, and it did not pass validation.

We tested whether a model predicts the next ball touch better when it knows:

- which player just touched the ball; and
- the identities of the opposing players.

The comparison model saw the same physical game state but did not receive player identities. In all three repeated runs, adding identity made the result slightly worse.

| Seed | Anonymous model NLL | Full identity model NLL | Identity model change |
|---:|---:|---:|---:|
| 17 | 2.712673 | 2.730788 | 0.668% worse |
| 23 | 2.706329 | 2.726284 | 0.737% worse |
| 41 | 2.727498 | 2.739154 | 0.427% worse |

Lower NLL is better.

## What worked

- The local laptop was viable: all acquisition, parsing, checks, and training ran locally on the NVIDIA RTX 5070 Ti Laptop GPU.
- No cloud compute was needed.
- The frozen corpus contained 1,595 replay files. Of those, 1,445 passed strict quality and identity checks.
- The final dataset contained 117,704 prediction examples: 63,458 train, 18,879 validation, and 35,367 sealed test examples.
- All data-quality, identity-resolution, sample-size, statistical-power, leakage, and local-compute gates passed.
- The small overfitting preflight passed, and all 12 planned validation runs completed.

This means the experiment stopped because of its scientific result, not because the data pipeline or computer failed.

## Why the experiment ends here

The preregistered safety gate required the full identity model to improve validation NLL by at least 2% in at least two of the three seeds.

The result was 0 passing seeds out of 3. The protocol explicitly says to stop without opening the sealed test when this gate fails. Therefore:

- no test-unlock file was created;
- the sealed test data was never evaluated;
- the test-only identity-shuffle and statistical controls were correctly not run; and
- the result was not tuned after seeing test performance.

This is the proper end of V1. It reached a planned stopping condition, rather than its hoped-for positive result. Any further experiment would be a separately designed and preregistered V2, not unfinished V1 work.

## What the result means

For this model and next-touch prediction task, player and opponent identities did not add useful predictive information beyond the complete physical game state on validation. The identity-conditioned model was slightly worse in every seed.

This does **not** prove that player identity never matters, that the result transfers directly to football, or that identity-conditioned coaching models are impossible. It is a controlled negative result for this particular RLCS objective and implementation.

## Reproducible records

- Detailed protocol and results: `docs/RLCS_IDENTITY_MATCHUP_V1.md`
- Frozen experiment configuration: `configs/rlcs_identity_matchup_v1.yaml`
- Machine-readable validation ledger: `provenance/rlcs_identity_matchup_v1_validation.json`
- Chronological split manifest: `splits/rlcs_2025_chronological_v1.json`
