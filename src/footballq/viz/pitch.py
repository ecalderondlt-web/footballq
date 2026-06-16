"""Matplotlib pitch drawing."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.axes import Axes

from footballq.constants import PITCH_LENGTH_M, PITCH_WIDTH_M


def draw_pitch(
    ax: Axes | None = None,
    line_color: str = "#242424",
    pitch_color: str = "#f5f7f2",
) -> Axes:
    """Draw a 105m x 68m football pitch."""

    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.set_facecolor(pitch_color)
    ax.set_xlim(-2, PITCH_LENGTH_M + 2)
    ax.set_ylim(PITCH_WIDTH_M + 2, -2)
    ax.set_aspect("equal")
    ax.axis("off")

    lw = 1.2
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            PITCH_LENGTH_M,
            PITCH_WIDTH_M,
            fill=False,
            ec=line_color,
            lw=lw,
        )
    )
    ax.plot([PITCH_LENGTH_M / 2, PITCH_LENGTH_M / 2], [0, PITCH_WIDTH_M], color=line_color, lw=lw)
    ax.add_patch(
        patches.Circle(
            (PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2),
            9.15,
            fill=False,
            ec=line_color,
            lw=lw,
        )
    )
    ax.add_patch(patches.Circle((PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2), 0.18, color=line_color))

    penalty_y = (PITCH_WIDTH_M - 40.32) / 2
    six_y = (PITCH_WIDTH_M - 18.32) / 2
    for left in [True, False]:
        x0 = 0 if left else PITCH_LENGTH_M - 16.5
        six_x = 0 if left else PITCH_LENGTH_M - 5.5
        spot_x = 11.0 if left else PITCH_LENGTH_M - 11.0
        goal_x = -1.5 if left else PITCH_LENGTH_M
        ax.add_patch(
            patches.Rectangle((x0, penalty_y), 16.5, 40.32, fill=False, ec=line_color, lw=lw)
        )
        ax.add_patch(
            patches.Rectangle((six_x, six_y), 5.5, 18.32, fill=False, ec=line_color, lw=lw)
        )
        ax.add_patch(patches.Circle((spot_x, PITCH_WIDTH_M / 2), 0.18, color=line_color))
        ax.add_patch(
            patches.Rectangle(
                (goal_x, PITCH_WIDTH_M / 2 - 3.66),
                1.5,
                7.32,
                fill=False,
                ec=line_color,
                lw=lw,
            )
        )

    return ax
