"""SoccerTrack v2 adapter stub."""

from __future__ import annotations

import pandas as pd

from footballq.constants import DATASET_SOCCERTRACK
from footballq.io.base import TrackingDataAdapter


class SoccerTrackAdapter(TrackingDataAdapter):
    """Schema-compatible placeholder for SoccerTrack v2.

    SoccerTrack v2 GSR annotations already expose pitch-meter `x, y`, persistent `track_id`,
    `player_id`, `role`, `jersey_number`, and `team_side`. A full adapter should map `image_id`
    to `frame_id`, `track_id` to `agent_id`, `team_side` to canonical `team_id`, and preserve
    role/referee rows. Phase 1 keeps this as an explicit stub because demo and tests do not need
    the dataset.
    """

    dataset = DATASET_SOCCERTRACK

    def load_tracking(self) -> pd.DataFrame:
        raise NotImplementedError(
            "SoccerTrack v2 loading is a Phase 1 stub. Future work should map GSR annotations "
            "with image_id, track_id, player_id, role, jersey_number, team_side, x, and y."
        )

