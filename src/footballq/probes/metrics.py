"""Metrics for frozen probe evaluation."""

from __future__ import annotations

import math

import torch


def classification_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> dict[str, object]:
    """Compute accuracy, F1 scores, and class-count diagnostics."""

    y_true = y_true.long().view(-1)
    y_pred = y_pred.long().view(-1)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true_value, pred_value in zip(y_true.tolist(), y_pred.tolist(), strict=False):
        if 0 <= true_value < num_classes and 0 <= pred_value < num_classes:
            confusion[true_value, pred_value] += 1
    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    class_counts = confusion.sum(dim=1)
    per_class_accuracy: list[float | None] = []
    f1_scores: list[float] = []
    for idx in range(num_classes):
        support = int(class_counts[idx].item())
        true_positive = float(confusion[idx, idx].item())
        false_positive = float(confusion[:, idx].sum().item() - true_positive)
        false_negative = float(confusion[idx, :].sum().item() - true_positive)
        precision_denom = true_positive + false_positive
        recall_denom = true_positive + false_negative
        precision = true_positive / precision_denom if precision_denom > 0 else 0.0
        recall = true_positive / recall_denom if recall_denom > 0 else 0.0
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        f1_scores.append(float(f1))
        per_class_accuracy.append(
            float(true_positive / support) if support > 0 else None
        )
    macro_f1 = float(sum(f1_scores) / max(num_classes, 1))
    weighted_denom = max(int(class_counts.sum().item()), 1)
    weighted_f1 = float(
        sum(f1 * int(class_counts[idx].item()) for idx, f1 in enumerate(f1_scores))
        / weighted_denom
    )
    return {
        "accuracy": float(correct / total) if total else math.nan,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_counts": [int(value) for value in class_counts.tolist()],
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
        "num_examples": total,
    }


def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    """Compute MAE, RMSE, R2, and target-distribution diagnostics."""

    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)
    if y_true.numel() == 0:
        return {
            "mae": math.nan,
            "rmse": math.nan,
            "r2": math.nan,
            "target_mean": math.nan,
            "target_std": math.nan,
            "num_examples": 0,
        }
    error = y_pred - y_true
    mae = torch.mean(torch.abs(error))
    rmse = torch.sqrt(torch.mean(error.square()))
    target_mean = torch.mean(y_true)
    target_std = torch.std(y_true, unbiased=False)
    total = torch.sum((y_true - target_mean).square())
    residual = torch.sum(error.square())
    r2 = torch.tensor(0.0) if float(total.item()) <= 1e-12 else 1.0 - residual / total
    return {
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "r2": float(r2.item()),
        "target_mean": float(target_mean.item()),
        "target_std": float(target_std.item()),
        "num_examples": int(y_true.numel()),
    }
