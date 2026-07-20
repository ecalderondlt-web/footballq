# GRF-to-PFF Train-Only Domain-Gap Audit V1

This report compares observable kinematic and geometric distributions only. It does not
use validation/test examples or establish tactical or semantic concepts.

## Sampling

- scope: `train_only`
- shared context examples per source: 24,576
- PFF training matches represented: 48
- PFF shards per training match: 4
- GRF scenario cap per job shard: 5,000
- deterministic seed: 20260713

## Largest Global Gaps

The gap score is quantile-Wasserstein distance divided by pooled robust spread. A score
near 1 means the average distribution shift is roughly one pooled interquartile scale.

| Rank | Metric | Unit | Gap score | PFF mean | GRF mean | PFF median | GRF median |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `player_acceleration_mps2` | m/s^2 | 1.4508 | 0.8177 | 7.4006 | 0.6701 | 5.1416 |
| 2 | `player_high_speed_indicator` | rate | 1.1607 | 0.0034 | 0.1195 | 0.0000 | 0.0000 |
| 3 | `player_turn_deg` | degrees | 1.1005 | 1.7093 | 7.1910 | 1.1288 | 2.5539 |
| 4 | `player_high_acceleration_indicator` | rate | 1.0101 | 0.0012 | 0.5097 | 0.0000 | 1.0000 |
| 5 | `player_speed_mps` | m/s | 0.8294 | 2.0213 | 3.9005 | 1.7406 | 3.6989 |
| 6 | `visible_team_centroid_distance_m` | m | 0.6904 | 6.6976 | 10.3091 | 6.2261 | 6.5095 |
| 7 | `visible_team_x_span_m` | m | 0.6347 | 21.6378 | 31.1538 | 21.5990 | 26.5395 |
| 8 | `visible_player_count` | players | 0.5818 | 13.4847 | 7.7579 | 14.0000 | 8.0000 |
| 9 | `visible_team_y_span_m` | m | 0.5245 | 36.2474 | 28.4354 | 36.0100 | 29.0649 |
| 10 | `ball_turn_deg` | degrees | 0.3516 | 6.3086 | 8.5287 | 0.7157 | 0.0485 |
| 11 | `player_stationary_indicator` | rate | 0.3275 | 0.0844 | 0.1323 | 0.0000 | 0.0000 |
| 12 | `ball_high_speed_indicator` | rate | 0.2061 | 0.0301 | 0.0516 | 0.0000 | 0.0000 |

## Scenario Diagnostics

### `11_vs_11_easy_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_acceleration_mps2` | 1.4786 | 0.8177 | 7.5895 |
| `player_turn_deg` | 1.1677 | 1.7093 | 7.2151 |
| `player_high_speed_indicator` | 1.1039 | 0.0034 | 0.1057 |
| `player_high_acceleration_indicator` | 1.0303 | 0.0012 | 0.5189 |
| `player_speed_mps` | 0.8294 | 2.0213 | 3.8436 |

### `11_vs_11_hard_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_acceleration_mps2` | 1.4846 | 0.8177 | 7.7513 |
| `player_turn_deg` | 1.1765 | 1.7093 | 7.4534 |
| `player_high_speed_indicator` | 1.1593 | 0.0034 | 0.1199 |
| `player_high_acceleration_indicator` | 1.0707 | 0.0012 | 0.5319 |
| `player_speed_mps` | 0.8913 | 2.0213 | 3.9983 |

### `11_vs_11_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_acceleration_mps2` | 1.4801 | 0.8177 | 7.5682 |
| `player_high_speed_indicator` | 1.1723 | 0.0034 | 0.1163 |
| `player_turn_deg` | 1.0986 | 1.7093 | 7.1801 |
| `player_high_acceleration_indicator` | 1.0505 | 0.0012 | 0.5217 |
| `player_speed_mps` | 0.8388 | 2.0213 | 3.9091 |

### `academy_3_vs_1_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 7.1510 | 6.6976 | 34.6217 |
| `visible_player_count` | 1.6243 | 13.4847 | 3.7694 |
| `visible_team_y_span_m` | 1.5804 | 36.2474 | 12.8490 |
| `player_high_speed_indicator` | 1.4521 | 0.0034 | 0.1601 |
| `nearest_player_distance_m` | 1.1737 | 6.3425 | 18.9435 |

### `academy_corner`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_high_speed_indicator` | 1.8010 | 0.0034 | 0.2176 |
| `player_acceleration_mps2` | 1.5801 | 0.8177 | 7.1616 |
| `player_speed_mps` | 1.1273 | 2.0213 | 4.8778 |
| `player_high_acceleration_indicator` | 1.0570 | 0.0012 | 0.5292 |
| `player_turn_deg` | 0.9007 | 1.7093 | 5.3663 |

### `academy_counterattack_easy`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_high_speed_indicator` | 1.9907 | 0.0034 | 0.2425 |
| `player_acceleration_mps2` | 1.2773 | 0.8177 | 6.7161 |
| `visible_team_x_span_m` | 1.0662 | 21.6378 | 41.1343 |
| `player_turn_deg` | 0.9859 | 1.7093 | 6.0436 |
| `player_high_acceleration_indicator` | 0.9293 | 0.0012 | 0.4648 |

### `academy_counterattack_hard`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_x_span_m` | 1.4056 | 21.6378 | 46.4678 |
| `player_acceleration_mps2` | 1.2024 | 0.8177 | 5.6600 |
| `player_speed_mps` | 1.0103 | 2.0213 | 4.8458 |
| `player_high_acceleration_indicator` | 0.7677 | 0.0012 | 0.3845 |
| `player_turn_deg` | 0.6558 | 1.7093 | 5.4456 |

### `academy_pass_and_shoot_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 12.2209 | 6.6976 | 37.4078 |
| `ball_speed_mps` | 2.6912 | 6.7967 | 19.1977 |
| `nearest_player_distance_m` | 2.0457 | 6.3425 | 24.3524 |
| `visible_team_y_span_m` | 1.9735 | 36.2474 | 19.8929 |
| `visible_player_count` | 1.8376 | 13.4847 | 3.3333 |

### `academy_run_pass_and_shoot_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 16.0803 | 6.6976 | 41.6948 |
| `visible_player_count` | 1.9146 | 13.4847 | 2.9902 |
| `visible_team_y_span_m` | 1.7440 | 36.2474 | 17.1623 |
| `nearest_player_distance_m` | 1.5279 | 6.3425 | 22.4791 |
| `player_stationary_indicator` | 0.9495 | 0.0844 | 0.5576 |

## Boundary

Use these train-only measurements to freeze a targeted simulator or objective change.
Do not tune against PFF validation, inspect PFF test, or interpret these measurements
as learned tactical concepts.

## Decision

`player_acceleration_mps2` satisfies the frozen redesign-selection rule. Its global gap score is
`1.4508`, and it is the largest gap in easy, standard, and hard full-match GRF scenarios. GRF mean
player acceleration is `7.4006 m/s^2` versus `0.8177 m/s^2` in PFF; the 95th percentiles are
`23.1409` and `1.8230`. The high-acceleration rate above `5 m/s^2` is `50.97%` in GRF and `0.12%`
in PFF.

Player turn angle also clears the global score threshold (`1.1005`) and ranks in the full-match top
five. Player speed is elevated (`3.9005 m/s` versus `2.0213 m/s`) but has a smaller score (`0.8294`).
Ball speed is much closer, with score `0.1582` and means `7.3821` versus `6.7967 m/s`. The selected
redesign therefore targets player velocity construction rather than globally slowing every entity.

The audit also exposes a separate visibility discrepancy. Sampled PFF training contexts average
`13.48` visible players at the endpoint, while masked GRF contexts average `7.76`; the earlier
frame-level PFF visibility profile target was `8.11`. The frame-level calibration is therefore not
matched to the context-conditioned tensor distribution. This remains a separate data-preparation
issue and is not bundled into the motion redesign.

The next train-only protocol is frozen in
`docs/GRF_PROVIDER_NEUTRAL_MOTION_PROTOCOL_V1.md`. It recomputes GRF velocity causally from position
differences using the same path used when provider velocity components are absent, then requires a
preflight domain-gap reduction before any model training.
