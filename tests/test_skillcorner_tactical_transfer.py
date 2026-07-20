from __future__ import annotations

import pandas as pd
import torch

from footballq.analysis.skillcorner_tactical_transfer import (
    TARGET_PENALTY_ENTRY,
    TARGET_TURNOVER,
    TacticalExamples,
    _penalty_entry_frames,
    align_phase_start,
    binary_metrics,
    validate_preflight,
)


def test_align_phase_start_is_strictly_causal() -> None:
    ends = [9, 11, 13, 15]
    indices = [90, 110, 130, 150]
    assert align_phase_start(ends, indices, phase_start=14, max_gap_frames=2) == (130, 13)
    assert align_phase_start(ends, indices, phase_start=13, max_gap_frames=2) == (110, 11)
    assert align_phase_start(ends, indices, phase_start=18, max_gap_frames=2) is None


def test_binary_metrics_reports_balanced_scores() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    probabilities = torch.tensor([0.1, 0.8, 0.9, 0.2])
    metrics = binary_metrics(labels, probabilities)
    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["macro_f1"] == 0.5
    assert metrics["confusion"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}


def test_penalty_entry_frames_preserve_later_candidates() -> None:
    events = pd.DataFrame(
        {
            "period": [1, 1, 1],
            "phase_index": [4, 4, 5],
            "frame_start": [98, 104, 110],
            "penalty_area_start": [True, True, False],
            "penalty_area_end": [False, False, True],
        }
    )
    frames = _penalty_entry_frames(events)
    assert frames[(1, 4)] == [98, 104]
    assert frames[(1, 5)] == [110]


def test_preflight_rejects_context_that_touches_phase() -> None:
    examples = TacticalExamples(
        state=torch.zeros(4, 2, 23, 5),
        mask=torch.ones(4, 2, 23, dtype=torch.bool),
        raw_flat=torch.zeros(4, 276),
        labels={
            TARGET_TURNOVER: torch.tensor([0, 1, 0, 1]),
            TARGET_PENALTY_ENTRY: torch.tensor([0, 1, 0, 1]),
        },
        label_masks={
            TARGET_TURNOVER: torch.ones(4, dtype=torch.bool),
            TARGET_PENALTY_ENTRY: torch.ones(4, dtype=torch.bool),
        },
        match_id=["a", "a", "b", "b"],
        period=[1, 1, 1, 1],
        phase_index=[0, 1, 0, 1],
        phase_start_frame=[10, 20, 30, 40],
        context_end_frame=[9, 20, 29, 39],
        source_sample_index=[1, 2, 3, 4],
        split_indices={"train": [0, 1], "val": [2, 3], "test": []},
        metadata={"included_splits": ["train", "val"]},
    )
    config = {
        "gates": {
            "minimum_examples_per_split": 1,
            "minimum_positive_examples_per_target_split": 1,
        }
    }
    failures = validate_preflight(examples, config)
    assert "at least one context reaches or crosses its phase start" in failures
