# GRF-to-PFF Train-Only Domain-Gap Audit V3

This report compares observable kinematic and geometric distributions only. It does not
use validation/test examples or establish tactical or semantic concepts.

## Sampling

- scope: `train_only`
- shared context examples per source: 16,785
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
| 1 | `player_acceleration_mps2` | m/s^2 | 1.4281 | 0.8192 | 7.0095 | 0.6701 | 5.1302 |
| 2 | `player_turn_deg` | degrees | 1.2586 | 1.6980 | 7.2273 | 1.1285 | 2.7437 |
| 3 | `ball_acceleration_mps2` | m/s^2 | 1.1930 | 11.2588 | 12.6447 | 2.9970 | 3.4212 |
| 4 | `ball_turn_deg` | degrees | 1.0787 | 6.2755 | 8.8006 | 0.7080 | 0.1327 |
| 5 | `visible_team_centroid_distance_m` | m | 0.9145 | 6.6953 | 12.0452 | 6.2461 | 6.7194 |
| 6 | `player_speed_mps` | m/s | 0.8471 | 2.0266 | 4.0072 | 1.7439 | 3.9142 |
| 7 | `visible_team_x_span_m` | m | 0.6439 | 21.6424 | 32.6040 | 21.5670 | 26.3483 |
| 8 | `visible_player_count` | players | 0.6222 | 13.5077 | 7.3316 | 14.0000 | 6.0000 |
| 9 | `visible_team_y_span_m` | m | 0.6186 | 36.1688 | 26.7889 | 35.9660 | 26.3249 |
| 10 | `player_high_acceleration_indicator` | rate | 0.5051 | 0.0014 | 0.5096 | 0.0000 | 1.0000 |
| 11 | `nearest_player_distance_m` | m | 0.2319 | 6.3246 | 7.6285 | 5.4643 | 5.4942 |
| 12 | `ball_speed_mps` | m/s | 0.2303 | 6.8427 | 6.3002 | 4.3637 | 5.2757 |

## Scenario Diagnostics

### `11_vs_11_easy_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_acceleration_mps2` | 1.5004 | 0.8192 | 7.2484 |
| `player_turn_deg` | 1.2849 | 1.6980 | 7.2561 |
| `ball_turn_deg` | 1.2149 | 6.2755 | 9.1254 |
| `ball_acceleration_mps2` | 1.0880 | 11.2588 | 13.2867 |
| `player_speed_mps` | 0.9050 | 2.0266 | 4.0274 |

### `11_vs_11_hard_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_acceleration_mps2` | 1.5274 | 0.8192 | 7.6078 |
| `player_turn_deg` | 1.3170 | 1.6980 | 7.7373 |
| `ball_acceleration_mps2` | 1.2698 | 11.2588 | 13.2778 |
| `player_speed_mps` | 0.9255 | 2.0266 | 4.1117 |
| `ball_turn_deg` | 0.8616 | 6.2755 | 8.0951 |

### `11_vs_11_stochastic`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_acceleration_mps2` | 1.4802 | 0.8192 | 7.3172 |
| `player_turn_deg` | 1.2667 | 1.6980 | 7.1884 |
| `ball_acceleration_mps2` | 1.2547 | 11.2588 | 13.2161 |
| `player_speed_mps` | 0.9068 | 2.0266 | 4.1163 |
| `ball_turn_deg` | 0.8494 | 6.2755 | 8.0163 |

### `academy_3_vs_1_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 7.1219 | 6.6953 | 34.6217 |
| `ball_acceleration_mps2` | 1.6326 | 11.2588 | 9.4415 |
| `visible_player_count` | 1.6243 | 13.5077 | 3.7694 |
| `visible_team_y_span_m` | 1.5788 | 36.1688 | 12.8490 |
| `nearest_player_distance_m` | 1.1768 | 6.3246 | 18.9435 |

### `academy_corner`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `player_acceleration_mps2` | 1.5509 | 0.8192 | 7.0590 |
| `ball_turn_deg` | 1.2421 | 6.2755 | 3.5439 |
| `ball_acceleration_mps2` | 1.1543 | 11.2588 | 15.2428 |
| `player_turn_deg` | 1.1293 | 1.6980 | 6.2927 |
| `player_speed_mps` | 1.1269 | 2.0266 | 4.8474 |

### `academy_counterattack_easy`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `ball_turn_deg` | 1.7283 | 6.2755 | 12.4693 |
| `ball_acceleration_mps2` | 1.5573 | 11.2588 | 18.7501 |
| `player_acceleration_mps2` | 1.2914 | 0.8192 | 6.5151 |
| `visible_team_x_span_m` | 1.0694 | 21.6424 | 41.1343 |
| `player_turn_deg` | 1.0195 | 1.6980 | 5.9452 |

### `academy_counterattack_hard`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_x_span_m` | 1.4104 | 21.6424 | 46.4678 |
| `player_acceleration_mps2` | 1.2023 | 0.8192 | 5.5268 |
| `ball_acceleration_mps2` | 1.1365 | 11.2588 | 15.0505 |
| `ball_turn_deg` | 1.0456 | 6.2755 | 8.3857 |
| `player_speed_mps` | 0.9991 | 2.0266 | 4.8313 |

### `academy_pass_and_shoot_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 12.1428 | 6.6953 | 37.4078 |
| `ball_acceleration_mps2` | 2.5351 | 11.2588 | 28.4628 |
| `nearest_player_distance_m` | 2.0784 | 6.3246 | 24.3524 |
| `visible_team_y_span_m` | 1.9738 | 36.1688 | 19.8929 |
| `visible_player_count` | 1.8376 | 13.5077 | 3.3333 |

### `academy_run_pass_and_shoot_with_keeper`

| Metric | Gap score | PFF mean | Scenario mean |
| --- | ---: | ---: | ---: |
| `visible_team_centroid_distance_m` | 15.9616 | 6.6953 | 41.6948 |
| `visible_player_count` | 1.9146 | 13.5077 | 2.9902 |
| `visible_team_y_span_m` | 1.7424 | 36.1688 | 17.1623 |
| `ball_acceleration_mps2` | 1.6996 | 11.2588 | 6.7281 |
| `nearest_player_distance_m` | 1.5315 | 6.3246 | 22.4791 |

## Boundary

Use these train-only measurements to freeze a targeted simulator or objective change.
Do not tune against PFF validation, inspect PFF test, or interpret these measurements
as learned tactical concepts.
