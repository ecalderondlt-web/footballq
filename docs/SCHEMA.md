# Canonical Schema

Canonical pitch:

- Length: 105.0 meters
- Width: 68.0 meters
- Origin: `x=0` left goal line, `y=0` top touchline in broadcast/minimap convention
- Center: `x=52.5`, `y=34.0`

## tracking.parquet

One row per frame-agent:

- `match_id`: string
- `dataset`: string, for example `metrica`, `skillcorner`, `soccertrack`, `synthetic`
- `period`: integer or nullable integer
- `frame_id`: integer
- `time_s`: float
- `agent_id`: string
- `agent_type`: one of `player`, `ball`, `referee`, `unknown`
- `team_id`: string or null, using `home`, `away`, `ball`, etc. where applicable
- `player_id`: string or null
- `jersey_number`: integer or null
- `role`: string or null
- `x_m`: float
- `y_m`: float
- `z_m`: float or null
- `raw_x`: float or null
- `raw_y`: float or null
- `is_visible`: boolean or null
- `source_file`: string or null

## events.parquet

- `match_id`
- `dataset`
- `period`
- `time_s`
- `frame_id`
- `team_id`
- `player_id`
- `event_type`
- `event_subtype`
- `x_m`
- `y_m`
- `end_x_m`
- `end_y_m`
- `outcome`
- `raw_event`

## features.parquet

One row per frame-agent, joined to tracking identifiers:

- `vx_mps`
- `vy_mps`
- `speed_mps`
- `ax_mps2`
- `ay_mps2`
- `accel_mps2`
- `distance_to_ball_m`
- `nearest_teammate_distance_m`
- `nearest_opponent_distance_m`
- `team_centroid_x_m`
- `team_centroid_y_m`
- `team_width_m`
- `team_length_m`

## windows.npz

- `X_history`: float array `[num_windows, history_steps, max_agents, num_features]`
- `Y_future`: float array `[num_windows, future_steps, max_agents, target_features]`
- `agent_mask_history`: bool array
- `agent_mask_future`: bool array
- `agent_ids`: string array
- `feature_names`: string array
- `target_names`: string array

`window_meta.parquet` stores `match_id`, `period`, `start_time_s`, `end_time_s`,
`history_end_time_s`, and `future_start_time_s`.

