"""Run a compact Experiment 3 frozen-probe comparison suite."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.probes.dataset import load_probe_dataset  # noqa: E402
from footballq.probes.io import save_json  # noqa: E402
from footballq.probes.training import train_probe_from_config  # noqa: E402

DEFAULT_TARGET_ORDER = [
    "future_ball_global_x_bucket",
    "team_shape_change_bucket",
    "future_ball_displacement_m",
    "possession_team",
    "has_ball_or_possession_available",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    return parser.parse_args()


def _row_from_result(
    target: str,
    task_type: str,
    feature_source: str,
    probe_type: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    metrics = result["test_metrics"]
    return {
        "target": target,
        "task_type": task_type,
        "feature_source": feature_source,
        "probe_type": probe_type,
        "test_accuracy": metrics.get("accuracy"),
        "test_macro_f1": metrics.get("macro_f1"),
        "test_mae": metrics.get("mae"),
        "test_rmse": metrics.get("rmse"),
        "test_r2": metrics.get("r2"),
        "num_train": metrics.get("num_train_examples"),
        "num_val": metrics.get("num_val_examples"),
        "num_test": metrics.get("num_test_examples"),
        "run_dir": str(result["run_dir"]),
        "best_checkpoint": str(result["best_checkpoint"]),
        "error": None,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "task_type",
        "feature_source",
        "probe_type",
        "test_accuracy",
        "test_macro_f1",
        "test_mae",
        "test_rmse",
        "test_r2",
        "num_train",
        "num_val",
        "num_test",
        "run_dir",
        "best_checkpoint",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    data = load_probe_dataset(args.dataset)
    available = set(data.examples.get("targets", {}))
    requested = args.targets or [
        target for target in DEFAULT_TARGET_ORDER if target in available
    ]
    targets = [target for target in requested if target in available]
    if not targets:
        raise SystemExit(
            f"No requested targets are available. Dataset targets: {sorted(available)}"
        )
    rows: list[dict[str, Any]] = []
    first_target = targets[0]
    for target in targets:
        task_type = data.target_types[target]
        combos = [
            ("td_jepa", "linear"),
            ("random_same_shape", "linear"),
            ("raw_state_summary", "linear"),
        ]
        if target == first_target:
            combos.extend([("td_jepa", "mlp"), ("raw_state_summary", "mlp")])
        for feature_source, probe_type in combos:
            cfg = {
                "seed": args.seed,
                "data": {"probe_dataset": str(args.dataset)},
                "target": {"name": target, "task_type": task_type},
                "features": {"source": feature_source, "random_seed": args.seed},
                "model": {
                    "probe_type": probe_type,
                    "hidden_dim": 128,
                    "dropout": 0.1,
                },
                "training": {
                    "batch_size": args.batch_size,
                    "max_epochs": args.max_epochs,
                    "patience": args.patience,
                    "learning_rate": args.learning_rate,
                    "weight_decay": 1e-4,
                    "device": args.device,
                    "seed": args.seed,
                    "run_root": str(args.run_root),
                },
            }
            try:
                result = train_probe_from_config(cfg)
                rows.append(_row_from_result(target, task_type, feature_source, probe_type, result))
            except Exception as exc:  # pragma: no cover - keeps suite diagnostics inspectable
                rows.append(
                    {
                        "target": target,
                        "task_type": task_type,
                        "feature_source": feature_source,
                        "probe_type": probe_type,
                        "test_accuracy": None,
                        "test_macro_f1": None,
                        "test_mae": None,
                        "test_rmse": None,
                        "test_r2": None,
                        "num_train": None,
                        "num_val": None,
                        "num_test": None,
                        "run_dir": None,
                        "best_checkpoint": None,
                        "error": str(exc),
                    }
                )
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.out / "results.csv")
    save_json({"results": rows}, args.out / "results.json")
    print(f"results_csv: {args.out / 'results.csv'}")
    print(f"results_json: {args.out / 'results.json'}")
    for row in rows:
        if row["task_type"] == "classification":
            metric = row["test_macro_f1"]
            metric_name = "macro_f1"
        else:
            metric = row["test_rmse"]
            metric_name = "rmse"
        print(
            f"{row['target']} | {row['feature_source']} | {row['probe_type']} | "
            f"{metric_name}={metric} | error={row['error']}"
        )


if __name__ == "__main__":
    main()
