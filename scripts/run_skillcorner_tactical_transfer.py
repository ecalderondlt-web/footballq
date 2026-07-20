"""Run the frozen SkillCorner tactical transfer pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.analysis.skillcorner_tactical_transfer import (  # noqa: E402
    build_tactical_examples,
    json_dump,
    load_config,
    run_benchmark,
    source_file_inventory,
    support_summary,
    validate_preflight,
)
from footballq.repro.manifest import file_sha256, git_metadata  # noqa: E402
from footballq.repro.splits import load_split_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/skillcorner_tactical_transfer_v1.yaml"),
    )
    parser.add_argument(
        "--stage", choices=["preflight", "run"], default="preflight"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("runs/skillcorner_tactical_transfer_v1")
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def protocol_manifest(config: dict, config_path: Path) -> dict:
    data_cfg = config["data"]
    split_path = ROOT / data_cfg["split_manifest"]
    split = load_split_manifest(split_path)
    raw_root = ROOT / data_cfg["raw_root"]
    checkpoint_rows = []
    for family, entries in config["checkpoints"].items():
        for entry in entries:
            path = ROOT / entry["path"]
            actual = file_sha256(path)
            if actual != entry["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch before freeze: {path}")
            checkpoint_rows.append(
                {
                    "family": family,
                    "seed": int(entry["seed"]),
                    "path": str(path),
                    "sha256": actual,
                }
            )
    td_path = ROOT / data_cfg["td_path"]
    implementation_paths = [
        ROOT / "src/footballq/analysis/skillcorner_tactical_transfer.py",
        ROOT / "scripts/run_skillcorner_tactical_transfer.py",
        ROOT / "tests/test_skillcorner_tactical_transfer.py",
    ]
    return {
        "experiment": config["experiment"],
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "git": git_metadata(),
        **split.metadata(),
        "td_path": str(td_path),
        "td_sha256": file_sha256(td_path),
        "implementation_files": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in implementation_paths
        ],
        "label_sources": source_file_inventory(raw_root, split.all_match_ids),
        "checkpoints": checkpoint_rows,
        "test_status_at_freeze": {
            "model_predictions_opened": False,
            "label_support_manually_audited": bool(
                config["integrity"][
                    "test_label_support_manually_audited_before_protocol_freeze"
                ]
            ),
        },
        "integrity": config["integrity"],
    }


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    config = load_config(config_path)
    frozen_path = out_dir / "frozen_protocol_manifest.json"

    if args.stage == "preflight":
        examples = build_tactical_examples(
            config, workspace_root=ROOT, include_splits=("train", "val")
        )
        failures = validate_preflight(examples, config)
        manifest = protocol_manifest(config, config_path)
        manifest["preflight"] = {
            "included_splits": ["train", "val"],
            "support": support_summary(examples),
            "failures": failures,
            "passed": not failures,
        }
        json_dump(frozen_path, manifest)
        torch.save(examples.to_dict(), out_dir / "preflight_examples.pt")
        print(f"preflight passed={not failures}; wrote {frozen_path}")
        if failures:
            raise SystemExit(2)
        return

    if not frozen_path.exists():
        raise FileNotFoundError("Run preflight before opening test model predictions.")
    frozen = __import__("json").loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["config_sha256"] != file_sha256(config_path):
        raise ValueError("Frozen protocol config hash no longer matches the config file.")
    if frozen["test_status_at_freeze"]["model_predictions_opened"]:
        raise ValueError("Frozen protocol says test model predictions were already opened.")

    examples = build_tactical_examples(config, workspace_root=ROOT)
    failures = validate_preflight(examples, config)
    if failures:
        raise ValueError("Full benchmark preflight failed: " + "; ".join(failures))
    torch.save(examples.to_dict(), out_dir / "tactical_examples.pt")
    result, embeddings = run_benchmark(
        examples, config, workspace_root=ROOT, device=args.device
    )
    result["frozen_protocol_manifest"] = str(frozen_path)
    result["frozen_protocol_manifest_sha256"] = file_sha256(frozen_path)
    json_dump(out_dir / "results.json", result)
    torch.save(embeddings, out_dir / "embeddings.pt")
    print(
        f"completed {config['experiment']} with {len(examples.match_id):,} examples; "
        f"wrote {out_dir / 'results.json'}"
    )


if __name__ == "__main__":
    main()
