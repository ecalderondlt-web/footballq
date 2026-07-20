"""Dataset adapters."""

from footballq.io.gfootball import GFootballAdapter
from footballq.io.metrica import MetricaAdapter
from footballq.io.pff import PFFAdapter
from footballq.io.skillcorner import SkillCornerAdapter
from footballq.io.soccertrack import SoccerTrackAdapter
from footballq.io.statsbomb import StatsBombAdapter

__all__ = [
    "GFootballAdapter",
    "MetricaAdapter",
    "PFFAdapter",
    "SkillCornerAdapter",
    "SoccerTrackAdapter",
    "StatsBombAdapter",
]

