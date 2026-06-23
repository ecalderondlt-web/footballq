"""Run Experiment 4C decoder learning-curve diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.decoding.learning_curve import run_decoder_learning_curve  # noqa: E402
from footballq.latent_flow.io import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--datasets", nargs="*", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--match-counts", nargs="*")
    parser.add_argument("--stress-percentile", type=float)
    parser.add_argument("--expected-horizons", nargs="*", type=float)
    parser.add_argument("--require-real-split", action="store_true")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-root", type=Path)
    return parser.parse_args()


def _option(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _config_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    suite_cfg = cfg.get("suite", {})
    datasets = (
        args.datasets
        or data_cfg.get("decoder_datasets")
        or suite_cfg.get("datasets")
        or None
    )
    dataset = args.dataset or data_cfg.get("decoder_dataset") or data_cfg.get("path")
    if datasets is None and dataset is not None:
        datasets = [dataset]
    out = args.out or suite_cfg.get("out")
    if not datasets or out is None:
        raise ValueError("Set --dataset/--datasets and --out, or provide them in --config.")
    datasets = _as_list(datasets)
    return {
        "dataset": Path(datasets[0]),
        "datasets": [Path(value) for value in datasets],
        "out": Path(out),
        "match_counts": _option(args.match_counts, suite_cfg.get("match_counts"), ["1", "3", "all"]),
        "models": _option(args.models, suite_cfg.get("models"), None),
        "stress_percentile": float(
            _option(args.stress_percentile, suite_cfg.get("stress_percentile"), 0.75)
        ),
        "expected_horizons": _option(
            args.expected_horizons,
            suite_cfg.get("expected_horizons"),
            [2.0, 4.0, 6.0],
        ),
        "require_real_split": bool(
            args.require_real_split or suite_cfg.get("require_real_split", False)
        ),
        "split": _option(args.split, suite_cfg.get("split"), "test"),
        "device": _option(args.device, suite_cfg.get("device"), "auto"),
        "epochs": int(_option(args.epochs, suite_cfg.get("epochs"), 1)),
        "max_train_batches": _option(
            args.max_train_batches,
            suite_cfg.get("max_train_batches"),
            20,
        ),
        "max_eval_batches": _option(
            args.max_eval_batches,
            suite_cfg.get("max_eval_batches"),
            20,
        ),
        "batch_size": int(_option(args.batch_size, suite_cfg.get("batch_size"), 256)),
        "seed": int(_option(args.seed, suite_cfg.get("seed"), 123)),
        "run_root": Path(_option(args.run_root, suite_cfg.get("run_root"), "runs")),
    }


def main() -> None:
    args = parse_args()
    options = _config_args(args)
    result = run_decoder_learning_curve(
        options["dataset"],
        options["out"],
        datasets=options["datasets"],
        match_counts=options["match_counts"],
        models=options["models"],
        stress_percentile=options["stress_percentile"],
        require_real_split=options["require_real_split"],
        expected_horizons=options["expected_horizons"],
        split=options["split"],
        device=options["device"],
        epochs=options["epochs"],
        max_train_batches=options["max_train_batches"],
        max_eval_batches=options["max_eval_batches"],
        batch_size=options["batch_size"],
        seed=options["seed"],
        run_root=options["run_root"],
    )
    print(f"results_csv: {result['results_csv']}")
    print(f"stress_results_csv: {result['stress_results_csv']}")
    print(f"summary_json: {result['summary_json']}")
    print(f"num_available_matches: {result['summary']['num_available_matches']}")
    if result["summary"].get("limited_to_three_matches"):
        print("warning: all-data learning curve is limited to three or fewer matches")
    for diagnostics in result["summary"].get("subset_diagnostics", []):
        print(
            "subset: "
            f"dataset={diagnostics['dataset_label']} matches={diagnostics['num_matches']} "
            f"train={';'.join(diagnostics['train_match_ids'])} "
            f"val={';'.join(diagnostics['val_match_ids'])} "
            f"test={';'.join(diagnostics['test_match_ids'])}"
        )
        if diagnostics.get("smoke_split"):
            print(
                "warning: "
                f"subset={diagnostics['subset_label']} uses a smoke split; "
                "train/val/test are not fully disjoint by match_id"
            )
    best_future = result["summary"].get("best_future_decoder")
    if best_future:
        print(
            "best_future: "
            f"{best_future['model']} all_entity_ADE_m={best_future['all_entity_ADE_m']}"
        )
    best_current = result["summary"].get("best_current_reconstruction")
    if best_current:
        print(
            "best_current: "
            f"{best_current['model']} current_all_entity_error_m="
            f"{best_current['current_all_entity_error_m']}"
        )


if __name__ == "__main__":
    main()
