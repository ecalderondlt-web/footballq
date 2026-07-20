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
| 1 | `player_acceleration_mps2` | m/s^2 | 1.4487 | 0.8177 | 22.9713 | 0.6701 | 5.2608 |
| 2 | `player_turn_deg` | degrees | 1.3438 | 1.7093 | 7.6695 | 1.1288 | 2.7279 |
| 3 | `ball_acceleration_mps2` | m/s^2 | 1.1130 | 10.6334 | 21.1020 | 2.9970 | 3.5432 |
| 4 | `ball_turn_deg` | degrees | 1.1006 | 6.3086 | 8.5308 | 0.7157 | 0.1235 |
| 5 | `player_speed_mps` | m/s | 0.8229 | 2.0213 | 4.8350 | 1.7406 | 3.7339 |
| 6 | `visible_team_centroid_distance_m` | m | 0.6904 | 6.6976 | 10.3091 | 6.2261 | 6.5095 |
| 7 | `visible_team_x_span_m` | m | 0.6347 | 21.6378 | 31.1538 | 21.5990 | 26.5395 |
| 8 | `visible_player_count` | players | 0.5818 | 13.4847 | 7.7579 | 14.0000 | 8.0000 |
| 9 | `visible_team_y_span_m` | m | 0.5245 | 36.2474 | 28.4354 | 36.0100 | 29.0649 |
| 10 | `player_high_acceleration_indicator` | rate | 0.5152 | 0.0012 | 0.5182 | 0.0000 | 1.0000 |
| 11 | `player_ball_distance_m` | m | 0.1903 | 19.5085 | 17.0504 | 18.3510 | 14.9865 |
| 12 | `ball_speed_mps` | m/s | 0.1666 | 6.7967 | 7.7313 | 4.2384 | 5.6536 |

## Scenario Diagnostics

### `11_vs_11_easy_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_acceleration_mps2` | 2.9734 | 0.8177 | 34.2998 |
| `player_turn_deg` | 1.4498 | 1.7093 | 8.0367 |
| `ball_acceleration_mps2` | 1.1346 | 10.6334 | 26.5735 |
| `ball_turn_deg` | 1.0523 | 6.3086 | 9.0203 |
| `player_speed_mps` | 0.8239 | 2.0213 | 5.2270 |

### `11_vs_11_hard_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_acceleration_mps2` | 1.4850 | 0.8177 | 13.7918 |
| `player_turn_deg` | 1.3495 | 1.7093 | 7.8493 |
| `ball_acceleration_mps2` | 1.1232 | 10.6334 | 17.7668 |
| `ball_turn_deg` | 0.9321 | 6.3086 | 8.2405 |
| `player_speed_mps` | 0.8768 | 2.0213 | 4.4555 |

### `11_vs_11_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_acceleration_mps2` | 1.4605 | 0.8177 | 20.2309 |
| `player_turn_deg` | 1.3004 | 1.7093 | 7.4941 |
| `ball_acceleration_mps2` | 1.1131 | 10.6334 | 22.0711 |
| `ball_turn_deg` | 0.8693 | 6.3086 | 7.7915 |
| `player_speed_mps` | 0.8315 | 2.0213 | 4.6575 |

### `academy_3_vs_1_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 7.1510 | 6.6976 | 34.6217 |
| `ball_acceleration_mps2` | 1.7285 | 10.6334 | 9.4415 |
| `visible_player_count` | 1.6243 | 13.4847 | 3.7694 |
| `visible_team_y_span_m` | 1.5804 | 36.2474 | 12.8490 |
| `nearest_player_distance_m` | 1.1737 | 6.3425 | 18.9435 |

### `academy_corner`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_acceleration_mps2` | 1.5508 | 0.8177 | 7.0590 |
| `ball_turn_deg` | 1.2651 | 6.3086 | 3.5439 |
| `ball_acceleration_mps2` | 1.1979 | 10.6334 | 15.2428 |
| `player_speed_mps` | 1.1304 | 2.0213 | 4.8474 |
| `player_turn_deg` | 1.1250 | 1.7093 | 6.2927 |

### `academy_counterattack_easy`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `ball_turn_deg` | 1.7022 | 6.3086 | 12.4693 |
| `ball_acceleration_mps2` | 1.4824 | 10.6334 | 18.7501 |
| `player_acceleration_mps2` | 1.2914 | 0.8177 | 6.5151 |
| `visible_team_x_span_m` | 1.0662 | 21.6378 | 41.1343 |
| `player_turn_deg` | 1.0151 | 1.7093 | 5.9452 |

### `academy_counterattack_hard`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_x_span_m` | 1.4056 | 21.6378 | 46.4678 |
| `player_acceleration_mps2` | 1.2022 | 0.8177 | 5.5268 |
| `ball_acceleration_mps2` | 1.1951 | 10.6334 | 15.0505 |
| `ball_turn_deg` | 1.0163 | 6.3086 | 8.3857 |
| `player_speed_mps` | 1.0020 | 2.0213 | 4.8313 |

### `academy_pass_and_shoot_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 12.2209 | 6.6976 | 37.4078 |
| `ball_acceleration_mps2` | 2.5762 | 10.6334 | 28.4628 |
| `nearest_player_distance_m` | 2.0732 | 6.3425 | 24.3524 |
| `visible_team_y_span_m` | 1.9735 | 36.2474 | 19.8929 |
| `visible_player_count` | 1.8376 | 13.4847 | 3.3333 |

### `academy_run_pass_and_shoot_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 16.0803 | 6.6976 | 41.6948 |
| `visible_player_count` | 1.9146 | 13.4847 | 2.9902 |
| `ball_acceleration_mps2` | 1.7964 | 10.6334 | 6.7281 |
| `visible_team_y_span_m` | 1.7440 | 36.2474 | 17.1623 |
| `nearest_player_distance_m` | 1.5279 | 6.3425 | 22.4791 |

## Boundary

Use these train-only measurements to freeze a targeted simulator or objective change.
Do not tune against PFF validation, inspect PFF test, or interpret these measurements
as learned tactical concepts.
