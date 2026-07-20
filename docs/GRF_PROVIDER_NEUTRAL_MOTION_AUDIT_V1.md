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
| 1 | `player_high_speed_indicator` | rate | 1.2629 | 0.0034 | 0.1204 | 0.0000 | 0.0000 |
| 2 | `player_turn_deg` | degrees | 1.1197 | 1.7093 | 7.6695 | 1.1288 | 2.7279 |
| 3 | `player_high_acceleration_indicator` | rate | 1.0303 | 0.0012 | 0.5182 | 0.0000 | 1.0000 |
| 4 | `visible_team_centroid_distance_m` | m | 0.6904 | 6.6976 | 10.3091 | 6.2261 | 6.5095 |
| 5 | `visible_team_x_span_m` | m | 0.6347 | 21.6378 | 31.1538 | 21.5990 | 26.5395 |
| 6 | `visible_player_count` | players | 0.5818 | 13.4847 | 7.7579 | 14.0000 | 8.0000 |
| 7 | `visible_team_y_span_m` | m | 0.5245 | 36.2474 | 28.4354 | 36.0100 | 29.0649 |
| 8 | `player_speed_mps` | m/s | 0.4140 | 2.0213 | 4.8350 | 1.7406 | 3.7339 |
| 9 | `ball_turn_deg` | degrees | 0.2722 | 6.3086 | 8.5308 | 0.7157 | 0.1235 |
| 10 | `player_stationary_indicator` | rate | 0.2669 | 0.0844 | 0.1222 | 0.0000 | 0.0000 |
| 11 | `ball_high_speed_indicator` | rate | 0.2070 | 0.0301 | 0.0508 | 0.0000 | 0.0000 |
| 12 | `player_ball_distance_m` | m | 0.1903 | 19.5085 | 17.0504 | 18.3510 | 14.9865 |

## Scenario Diagnostics

### `11_vs_11_easy_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_turn_deg` | 1.1593 | 1.7093 | 8.0367 |
| `player_high_speed_indicator` | 1.0928 | 0.0034 | 0.1087 |
| `player_high_acceleration_indicator` | 1.0505 | 0.0012 | 0.5299 |
| `visible_team_x_span_m` | 0.5643 | 21.6378 | 28.9806 |
| `visible_player_count` | 0.5079 | 13.4847 | 8.2062 |

### `11_vs_11_hard_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_high_speed_indicator` | 1.2641 | 0.0034 | 0.1201 |
| `player_turn_deg` | 1.1653 | 1.7093 | 7.8493 |
| `player_high_acceleration_indicator` | 1.0707 | 0.0012 | 0.5392 |
| `player_speed_mps` | 0.6501 | 2.0213 | 4.4555 |
| `visible_team_x_span_m` | 0.5805 | 21.6378 | 29.2139 |

### `11_vs_11_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_high_speed_indicator` | 1.1726 | 0.0034 | 0.1162 |
| `player_turn_deg` | 1.1377 | 1.7093 | 7.4941 |
| `player_high_acceleration_indicator` | 1.0707 | 0.0012 | 0.5311 |
| `visible_team_x_span_m` | 0.5854 | 21.6378 | 29.6017 |
| `visible_player_count` | 0.5031 | 13.4847 | 8.2648 |

### `academy_3_vs_1_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 7.1510 | 6.6976 | 34.6217 |
| `visible_player_count` | 1.6243 | 13.4847 | 3.7694 |
| `visible_team_y_span_m` | 1.5804 | 36.2474 | 12.8490 |
| `player_high_speed_indicator` | 1.4492 | 0.0034 | 0.1527 |
| `nearest_player_distance_m` | 1.1737 | 6.3425 | 18.9435 |

### `academy_corner`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_high_speed_indicator` | 1.8147 | 0.0034 | 0.2112 |
| `player_acceleration_mps2` | 1.5508 | 0.8177 | 7.0590 |
| `player_speed_mps` | 1.1304 | 2.0213 | 4.8474 |
| `player_turn_deg` | 1.1250 | 1.7093 | 6.2927 |
| `player_high_acceleration_indicator` | 1.0101 | 0.0012 | 0.5019 |

### `academy_counterattack_easy`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_high_speed_indicator` | 1.9144 | 0.0034 | 0.2397 |
| `player_acceleration_mps2` | 1.2914 | 0.8177 | 6.5151 |
| `visible_team_x_span_m` | 1.0662 | 21.6378 | 41.1343 |
| `player_turn_deg` | 1.0151 | 1.7093 | 5.9452 |
| `player_high_acceleration_indicator` | 0.9495 | 0.0012 | 0.4760 |

### `academy_counterattack_hard`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_x_span_m` | 1.4056 | 21.6378 | 46.4678 |
| `player_acceleration_mps2` | 1.2022 | 0.8177 | 5.5268 |
| `player_speed_mps` | 1.0020 | 2.0213 | 4.8313 |
| `player_turn_deg` | 0.8104 | 1.7093 | 5.3430 |
| `player_high_acceleration_indicator` | 0.7814 | 0.0012 | 0.3902 |

### `academy_pass_and_shoot_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 12.2209 | 6.6976 | 37.4078 |
| `nearest_player_distance_m` | 2.0457 | 6.3425 | 24.3524 |
| `visible_team_y_span_m` | 1.9735 | 36.2474 | 19.8929 |
| `visible_player_count` | 1.8376 | 13.4847 | 3.3333 |
| `player_acceleration_mps2` | 1.2992 | 0.8177 | 3.9587 |

### `academy_run_pass_and_shoot_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 16.0803 | 6.6976 | 41.6948 |
| `visible_player_count` | 1.9146 | 13.4847 | 2.9902 |
| `visible_team_y_span_m` | 1.7440 | 36.2474 | 17.1623 |
| `nearest_player_distance_m` | 1.5279 | 6.3425 | 22.4791 |
| `player_stationary_indicator` | 0.9495 | 0.0844 | 0.5552 |

## Boundary

Use these train-only measurements to freeze a targeted simulator or objective change.
Do not tune against PFF validation, inspect PFF test, or interpret these measurements
as learned tactical concepts.
