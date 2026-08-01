"""Run the development-only StatsBomb player-history signal diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from footballq.analysis.statsbomb_player_history_signal import (
    build_development_examples,
    evaluate_development_cache,
    load_config,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}.")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statsbomb_player_history_signal_v1.yaml",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(config["output"]["run_dir"])
    cache_path = run_dir / "development_cache.pt"
    audit_path = run_dir / "source_audit.json"
    result_path = run_dir / "development_results.json"
    if args.rebuild_cache or not cache_path.is_file():
        cache, audit = build_development_examples(config)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        _write_json(audit_path, audit)
    else:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        audit = cache["audit"]
        _write_json(audit_path, audit)
    if bool(audit.get("sealed_test_loaded")):
        raise RuntimeError("Development diagnostic unexpectedly loaded the sealed test cohort.")
    results = evaluate_development_cache(cache, config, device=args.device)
    _write_json(result_path, results)
    print(
        json.dumps(
            {
                "audit": str(audit_path),
                "results": str(result_path),
                "examples": int(audit["development_examples"]),
                "sealed_test_loaded": bool(audit["sealed_test_loaded"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
