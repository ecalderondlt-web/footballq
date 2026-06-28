"""Repository-root import shim for the ``src`` layout.

This lets lightweight smoke commands run from a checkout before an editable
install, while packaging still uses ``src/footballq`` as the real package.
"""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "footballq"
__path__ = [str(_SRC_PACKAGE)]

__all__ = ["PITCH_LENGTH_M", "PITCH_WIDTH_M"]


def __getattr__(name: str) -> float:
    if name == "PITCH_LENGTH_M":
        from footballq.constants import PITCH_LENGTH_M

        return PITCH_LENGTH_M
    if name == "PITCH_WIDTH_M":
        from footballq.constants import PITCH_WIDTH_M

        return PITCH_WIDTH_M
    raise AttributeError(name)
