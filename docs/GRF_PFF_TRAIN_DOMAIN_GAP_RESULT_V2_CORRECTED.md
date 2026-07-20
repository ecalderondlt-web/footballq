# GRF-to-PFF Train-Only Domain-Gap Audit V2

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

Continuous-metric gaps divide quantile-Wasserstein distance by pooled interquartile
scale, with median absolute deviation only as a zero-scale fallback. Rate metrics use
the fixed probability range 1.0. Standard deviation never reduces a gap score.

| Rank | Metric | Unit | Gap score | PFF mean | GRF mean | PFF median | GRF median |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `ball_turn_deg` | degrees | 1.9239 | 6.3086 | 8.5287 | 0.7157 | 0.0485 |
| 2 | `player_acceleration_mps2` | m/s^2 | 1.4508 | 0.8177 | 7.4006 | 0.6701 | 5.1416 |
| 3 | `player_turn_deg` | degrees | 1.2648 | 1.7093 | 7.1910 | 1.1288 | 2.5539 |
| 4 | `ball_acceleration_mps2` | m/s^2 | 1.2214 | 10.6334 | 14.7353 | 2.9970 | 2.9990 |
| 5 | `player_speed_mps` | m/s | 0.8294 | 2.0213 | 3.9005 | 1.7406 | 3.6989 |
| 6 | `visible_team_centroid_distance_m` | m | 0.6904 | 6.6976 | 10.3091 | 6.2261 | 6.5095 |
| 7 | `visible_team_x_span_m` | m | 0.6347 | 21.6378 | 31.1538 | 21.5990 | 26.5395 |
| 8 | `visible_player_count` | players | 0.5818 | 13.4847 | 7.7579 | 14.0000 | 8.0000 |
| 9 | `visible_team_y_span_m` | m | 0.5245 | 36.2474 | 28.4354 | 36.0100 | 29.0649 |
| 10 | `player_high_acceleration_indicator` | rate | 0.5051 | 0.0012 | 0.5097 | 0.0000 | 1.0000 |
| 11 | `player_ball_distance_m` | m | 0.1903 | 19.5085 | 17.0504 | 18.3510 | 14.9865 |
| 12 | `ball_speed_mps` | m/s | 0.1582 | 6.7967 | 7.3821 | 4.2384 | 5.6844 |

## Scenario Diagnostics

### `11_vs_11_easy_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `ball_turn_deg` | 1.5791 | 6.3086 | 7.5740 |
| `player_acceleration_mps2` | 1.4786 | 0.8177 | 7.5895 |
| `player_turn_deg` | 1.2598 | 1.7093 | 7.2151 |
| `ball_acceleration_mps2` | 1.0754 | 10.6334 | 14.2607 |
| `player_speed_mps` | 0.8294 | 2.0213 | 3.8436 |

### `11_vs_11_hard_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `ball_turn_deg` | 2.1435 | 6.3086 | 9.0993 |
| `player_acceleration_mps2` | 1.4846 | 0.8177 | 7.7513 |
| `ball_acceleration_mps2` | 1.4244 | 10.6334 | 15.7402 |
| `player_turn_deg` | 1.3058 | 1.7093 | 7.4534 |
| `player_speed_mps` | 0.8913 | 2.0213 | 3.9983 |

### `11_vs_11_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `ball_turn_deg` | 1.6434 | 6.3086 | 7.8483 |
| `ball_acceleration_mps2` | 1.4852 | 10.6334 | 15.7077 |
| `player_acceleration_mps2` | 1.4801 | 0.8177 | 7.5682 |
| `player_turn_deg` | 1.2629 | 1.7093 | 7.1801 |
| `player_speed_mps` | 0.8388 | 2.0213 | 3.9091 |

### `academy_3_vs_1_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 7.1510 | 6.6976 | 34.6217 |
| `visible_player_count` | 1.6243 | 13.4847 | 3.7694 |
| `visible_team_y_span_m` | 1.5804 | 36.2474 | 12.8490 |
| `ball_acceleration_mps2` | 1.5756 | 10.6334 | 11.2363 |
| `ball_turn_deg` | 1.2594 | 6.3086 | 7.6434 |

### `academy_corner`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `ball_turn_deg` | 2.3657 | 6.3086 | 14.3135 |
| `ball_acceleration_mps2` | 1.7845 | 10.6334 | 34.9430 |
| `player_acceleration_mps2` | 1.5801 | 0.8177 | 7.1616 |
| `player_speed_mps` | 1.1273 | 2.0213 | 4.8778 |
| `player_turn_deg` | 0.9007 | 1.7093 | 5.3663 |

### `academy_counterattack_easy`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `ball_turn_deg` | 1.8219 | 6.3086 | 8.1028 |
| `player_acceleration_mps2` | 1.2773 | 0.8177 | 6.7161 |
| `ball_acceleration_mps2` | 1.1925 | 10.6334 | 12.8883 |
| `visible_team_x_span_m` | 1.0662 | 21.6378 | 41.1343 |
| `player_turn_deg` | 1.0272 | 1.7093 | 6.0436 |

### `academy_counterattack_hard`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `ball_turn_deg` | 2.8840 | 6.3086 | 11.1784 |
| `ball_acceleration_mps2` | 1.9014 | 10.6334 | 20.8531 |
| `visible_team_x_span_m` | 1.4056 | 21.6378 | 46.4678 |
| `player_acceleration_mps2` | 1.2024 | 0.8177 | 5.6600 |
| `player_speed_mps` | 1.0103 | 2.0213 | 4.8458 |

### `academy_pass_and_shoot_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 12.2209 | 6.6976 | 37.4078 |
| `ball_speed_mps` | 2.6912 | 6.7967 | 19.1977 |
| `nearest_player_distance_m` | 2.0732 | 6.3425 | 24.3524 |
| `visible_team_y_span_m` | 1.9735 | 36.2474 | 19.8929 |
| `visible_player_count` | 1.8376 | 13.4847 | 3.3333 |

### `academy_run_pass_and_shoot_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 16.0803 | 6.6976 | 41.6948 |
| `ball_turn_deg` | 4.2661 | 6.3086 | 14.0569 |
| `ball_acceleration_mps2` | 2.8866 | 10.6334 | 7.5843 |
| `visible_player_count` | 1.9146 | 13.4847 | 2.9902 |
| `visible_team_y_span_m` | 1.7440 | 36.2474 | 17.1623 |

## Boundary

Use these train-only measurements to freeze a targeted simulator or objective change.
Do not tune against PFF validation, inspect PFF test, or interpret these measurements
as learned tactical concepts.
