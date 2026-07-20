# GRF-to-PFF Train-Only Domain-Gap Audit V4

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
| 1 | `ball_acceleration_mps2` | m/s^2 | 1.6401 | 10.6334 | 10.3034 | 2.9970 | 6.5871 |
| 2 | `player_acceleration_mps2` | m/s^2 | 1.5434 | 0.8177 | 5.2646 | 0.6701 | 4.6428 |
| 3 | `player_turn_deg` | degrees | 1.0393 | 1.7093 | 6.2094 | 1.1288 | 2.6246 |
| 4 | `player_speed_mps` | m/s | 0.7648 | 2.0213 | 3.8779 | 1.7406 | 3.7518 |
| 5 | `visible_team_centroid_distance_m` | m | 0.6988 | 6.6976 | 10.2175 | 6.2261 | 6.5168 |
| 6 | `visible_team_x_span_m` | m | 0.6407 | 21.6378 | 31.3543 | 21.5990 | 26.4908 |
| 7 | `visible_player_count` | players | 0.5838 | 13.4847 | 7.7039 | 14.0000 | 8.0000 |
| 8 | `visible_team_y_span_m` | m | 0.5345 | 36.2474 | 28.2447 | 36.0100 | 28.6994 |
| 9 | `ball_turn_deg` | degrees | 0.4723 | 6.3086 | 7.0187 | 0.7157 | 2.0427 |
| 10 | `player_high_acceleration_indicator` | rate | 0.4646 | 0.0012 | 0.4648 | 0.0000 | 0.0000 |
| 11 | `player_ball_distance_m` | m | 0.2096 | 19.5085 | 16.6640 | 18.3510 | 14.7628 |
| 12 | `ball_speed_mps` | m/s | 0.2064 | 6.7967 | 6.8551 | 4.2384 | 5.6077 |

## Scenario Diagnostics

### `11_vs_11_easy_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_acceleration_mps2` | 1.6348 | 0.8177 | 5.3174 |
| `ball_acceleration_mps2` | 1.5535 | 10.6334 | 10.6361 |
| `player_turn_deg` | 1.0226 | 1.7093 | 6.0853 |
| `player_speed_mps` | 0.7816 | 2.0213 | 3.8632 |
| `visible_team_x_span_m` | 0.5655 | 21.6378 | 28.8568 |

### `11_vs_11_hard_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_acceleration_mps2` | 1.6421 | 0.8177 | 5.4834 |
| `ball_acceleration_mps2` | 1.6107 | 10.6334 | 10.5831 |
| `player_turn_deg` | 1.0685 | 1.7093 | 6.4919 |
| `player_speed_mps` | 0.8230 | 2.0213 | 3.9824 |
| `visible_team_x_span_m` | 0.5776 | 21.6378 | 29.1194 |

### `11_vs_11_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `ball_acceleration_mps2` | 1.6366 | 10.6334 | 10.3412 |
| `player_acceleration_mps2` | 1.6190 | 0.8177 | 5.4175 |
| `player_turn_deg` | 1.0525 | 1.7093 | 6.2494 |
| `player_speed_mps` | 0.7675 | 2.0213 | 3.8845 |
| `visible_team_x_span_m` | 0.5806 | 21.6378 | 29.3898 |

### `academy_3_vs_1_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 7.1510 | 6.6976 | 34.6217 |
| `ball_acceleration_mps2` | 2.0691 | 10.6334 | 9.5046 |
| `visible_player_count` | 1.6243 | 13.4847 | 3.7694 |
| `visible_team_y_span_m` | 1.5804 | 36.2474 | 12.8490 |
| `nearest_player_distance_m` | 1.1737 | 6.3425 | 18.9435 |

### `academy_corner`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_acceleration_mps2` | 1.6230 | 0.8177 | 5.5695 |
| `ball_acceleration_mps2` | 1.5505 | 10.6334 | 19.4123 |
| `player_speed_mps` | 1.1007 | 2.0213 | 4.6522 |
| `player_turn_deg` | 0.9681 | 1.7093 | 6.5748 |
| `visible_team_y_span_m` | 0.5611 | 36.2474 | 28.8426 |

### `academy_counterattack_easy`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `ball_acceleration_mps2` | 1.5522 | 10.6334 | 13.7318 |
| `player_acceleration_mps2` | 1.2962 | 0.8177 | 4.8034 |
| `visible_team_x_span_m` | 1.0662 | 21.6378 | 41.1343 |
| `player_turn_deg` | 0.9580 | 1.7093 | 5.7710 |
| `player_speed_mps` | 0.7614 | 2.0213 | 4.2815 |

### `academy_counterattack_hard`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `ball_acceleration_mps2` | 1.5604 | 10.6334 | 12.7170 |
| `visible_team_x_span_m` | 1.4056 | 21.6378 | 46.4678 |
| `player_acceleration_mps2` | 1.0923 | 0.8177 | 4.1831 |
| `player_speed_mps` | 0.9622 | 2.0213 | 4.7410 |
| `player_turn_deg` | 0.7036 | 1.7093 | 4.1077 |

### `academy_pass_and_shoot_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 12.2209 | 6.6976 | 37.4078 |
| `nearest_player_distance_m` | 2.0732 | 6.3425 | 24.3524 |
| `ball_acceleration_mps2` | 2.0621 | 10.6334 | 25.2702 |
| `visible_team_y_span_m` | 1.9735 | 36.2474 | 19.8929 |
| `visible_player_count` | 1.8376 | 13.4847 | 3.3333 |

### `academy_run_pass_and_shoot_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 16.0803 | 6.6976 | 41.6948 |
| `visible_player_count` | 1.9146 | 13.4847 | 2.9902 |
| `visible_team_y_span_m` | 1.7440 | 36.2474 | 17.1623 |
| `nearest_player_distance_m` | 1.5279 | 6.3425 | 22.4791 |
| `ball_acceleration_mps2` | 1.4839 | 10.6334 | 5.4019 |

## Boundary

Use these train-only measurements to freeze a targeted simulator or objective change.
Do not tune against PFF validation, inspect PFF test, or interpret these measurements
as learned tactical concepts.
