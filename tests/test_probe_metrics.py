import math

import torch

from footballq.probes.metrics import classification_metrics, regression_metrics


def test_classification_metrics():
    metrics = classification_metrics(
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 1, 1]),
        num_classes=2,
    )
    assert metrics["accuracy"] == 0.75
    assert abs(metrics["macro_f1"] - ((2 / 3) + 0.8) / 2) < 1e-6
    assert metrics["per_class_counts"] == [2, 2]
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_regression_metrics():
    metrics = regression_metrics(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([1.0, 4.0, 3.0]),
    )
    assert abs(metrics["mae"] - (2 / 3)) < 1e-6
    assert abs(metrics["rmse"] - math.sqrt(4 / 3)) < 1e-6
    assert abs(metrics["r2"] - (-1.0)) < 1e-6
    assert metrics["target_mean"] == 2.0
