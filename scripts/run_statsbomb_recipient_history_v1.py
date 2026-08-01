"""Run the low-cost StatsBomb recipient-history development experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from footballq.analysis.statsbomb_recipient_history import (
    build_development_cache,
    evaluate_development_cache,
    load_config,
)
from footballq.repro.manifest import build_run_manifest, write_run_manifest


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}.")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statsbomb_recipient_history_v1.yaml",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(config["output"]["run_dir"])
    cache_path = run_dir / "development_cache.pt"
    audit_path = run_dir / "source_audit.json"
    result_path = run_dir / "development_results.json"
    manifest_path = run_dir / "run_manifest.json"
    if args.rebuild_cache or not cache_path.is_file():
        cache, audit = build_development_cache(config)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
    else:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        audit = cache["audit"]
    _write_json(audit_path, audit)
    if bool(audit.get("sealed_test_loaded")):
        raise RuntimeError("Development runner unexpectedly loaded sealed test data.")
    manifest = build_run_manifest(
        command=[sys.executable, *sys.argv],
        config_path=args.config,
        split_manifest_path=config["data"]["split_manifest"],
        evaluation_protocol=str(config["experiment_protocol"]),
        feature_view=str(config["feature_view"]),
        objective_mode=str(config["objective_mode"]),
        dataset_paths={
            "statsbomb_open_data": config["data"]["statsbomb_root"],
            "split_manifest": config["data"]["split_manifest"],
        },
        output_paths={
            "cache": cache_path,
            "audit": audit_path,
            "results": result_path,
        },
        warnings=[
            "Development-only; Bayer Leverkusen is an opened development test.",
            "UEFA Euro 2024 outcome labels are not loaded by this runner.",
            (
                "StatsBomb 360 availability anchors query events, but V1 does not "
                "consume anonymous freeze-frame geometry as a model feature."
            ),
        ],
    )
    write_run_manifest(manifest_path, manifest)
    results = evaluate_development_cache(cache, config)
    results["run_manifest_path"] = str(manifest_path)
    results["run_manifest_split_sha256"] = manifest["split_manifest_sha256"]
    results.pop("result_payload_sha256", None)
    results["result_payload_sha256"] = _stable_hash(results)
    _write_json(result_path, results)
    print(
        json.dumps(
            {
                "audit": str(audit_path),
                "manifest": str(manifest_path),
                "results": str(result_path),
                "examples": int(audit["development_examples"]),
                "sealed_test_loaded": bool(audit["sealed_test_loaded"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
