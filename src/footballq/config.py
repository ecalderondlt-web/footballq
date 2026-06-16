"""Validated configuration objects."""

from pathlib import Path

from pydantic import BaseModel, Field


class PitchConfig(BaseModel):
    """Canonical pitch dimensions."""

    length_m: float = Field(default=105.0, gt=0)
    width_m: float = Field(default=68.0, gt=0)


class WindowConfig(BaseModel):
    """Fixed-length trajectory window configuration."""

    history_s: float = Field(default=5.0, gt=0)
    future_s: float = Field(default=5.0, gt=0)
    fps: float = Field(default=10.0, gt=0)
    max_agents: int = Field(default=23, gt=0)

    @property
    def history_steps(self) -> int:
        return int(round(self.history_s * self.fps))

    @property
    def future_steps(self) -> int:
        return int(round(self.future_s * self.fps))


class IngestConfig(BaseModel):
    """Common ingest configuration."""

    raw_dir: Path
    out_dir: Path
    match_id: str

