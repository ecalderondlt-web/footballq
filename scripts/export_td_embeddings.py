"""Export TD-JEPA embeddings for downstream probe experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.repro.manifest import build_run_manifest, write_run_manifest  # noqa: E402
from footballq.training.export_td_embeddings import export_td_embeddings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = export_td_embeddings(
        checkpoint=args.checkpoint,
        data_path=args.data,
        out=args.out,
        split=args.split,
        device=args.device,
    )
    payload = torch.load(out, map_location="cpu", weights_only=False)
    split_manifest = payload.get("split_manifest_path")
    if payload.get("scientific_mode") and split_manifest:
        manifest_path = out.with_name(f"{out.stem}_run_manifest.json")
        manifest = build_run_manifest(
            command=sys.argv,
            config_path=None,
            split_manifest_path=split_manifest,
            evaluation_protocol="inductive",
            feature_view=str(payload.get("feature_view", "unknown")),
            objective_mode=str(payload.get("objective_mode", "embedding_export")),
            dataset_paths={"checkpoint": args.checkpoint, "td_jepa": args.data},
            output_paths={"embeddings": out, "run_manifest": manifest_path},
            warnings=[],
        )
        write_run_manifest(manifest_path, manifest)
    print(f"embeddings: {out}")


if __name__ == "__main__":
    main()
