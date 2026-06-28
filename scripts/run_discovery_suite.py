"""Run Experiment 5 latent transition discovery suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.discovery.report import run_discovery_suite  # noqa: E402
from footballq.latent_flow.io import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--delta-steps", nargs="+", type=int)
    parser.add_argument("--k", nargs="+", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument(
        "--feature",
        choices=["raw_delta_z", "normalized_delta_z", "z_t_delta_z"],
        default=None,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--fit-sample-size", type=int)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--scientific-mode", action="store_true")
    return parser.parse_args()


def _option(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def main() -> None:
    args = parse_args()
    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    suite_cfg = cfg.get("suite", {})
    embeddings = _option(args.embeddings, data_cfg.get("embeddings"), None)
    windows = _option(args.windows, data_cfg.get("windows"), None)
    out = _option(args.out, suite_cfg.get("out"), None)
    split_manifest = _option(
        args.split_manifest,
        suite_cfg.get("split_manifest", data_cfg.get("split_manifest")),
        None,
    )
    if embeddings is None or windows is None or out is None:
        raise ValueError("Set --embeddings, --windows, and --out, or provide --config.")
    result = run_discovery_suite(
        embeddings=Path(embeddings),
        windows=Path(windows),
        out=Path(out),
        delta_steps=[
            int(value)
            for value in _option(args.delta_steps, suite_cfg.get("delta_steps"), [2, 5, 10])
        ],
        k_values=[int(value) for value in _option(args.k, suite_cfg.get("k"), [8, 16, 32, 64])],
        fps=float(_option(args.fps, suite_cfg.get("fps"), 10.0)),
        feature=str(_option(args.feature, suite_cfg.get("feature"), "normalized_delta_z")),
        seed=int(_option(args.seed, suite_cfg.get("seed"), 123)),
        max_iter=int(_option(args.max_iter, suite_cfg.get("max_iter"), 30)),
        fit_sample_size=int(_option(args.fit_sample_size, suite_cfg.get("fit_sample_size"), 50000)),
        split_manifest_path=Path(split_manifest) if split_manifest is not None else None,
        scientific_mode=bool(args.scientific_mode or suite_cfg.get("scientific_mode", False)),
    )
    print(f"summary_json: {result['summary_json']}")
    print(f"report_md: {result['report_md']}")
    print(f"transition_dataset: {result['transition_dataset_path']}")
    print(f"num_transitions: {result['transition_dataset']['num_examples']}")
    print(f"num_matches: {result['transition_dataset']['num_matches']}")
    print(f"recommended_k: {result['recommended_k']}")


if __name__ == "__main__":
    main()
