"""Create blinded diagnostic annotation CSV scaffolds."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def write_blinded_annotation_files(
    rows: list[dict[str, object]],
    annotator_csv: str | Path,
    key_csv: str | Path,
) -> tuple[Path, Path]:
    """Write blinded annotator rows and a separate key file."""

    annotator_path = Path(annotator_csv)
    key_path = Path(key_csv)
    annotator_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    annotator_fields = ["blind_id", "match_id", "period", "frame_t", "clip_path", "annotation"]
    key_fields = ["blind_id", "cluster_id", "latent_residual_score", "positive_control"]
    with annotator_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=annotator_fields)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(
                {
                    "blind_id": f"blind_{idx:05d}",
                    "match_id": row.get("match_id", ""),
                    "period": row.get("period", ""),
                    "frame_t": row.get("frame_t", ""),
                    "clip_path": row.get("clip_path", ""),
                    "annotation": "",
                }
            )
    with key_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=key_fields)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(
                {
                    "blind_id": f"blind_{idx:05d}",
                    "cluster_id": row.get("cluster_id", ""),
                    "latent_residual_score": row.get("latent_residual_score", ""),
                    "positive_control": row.get("positive_control", ""),
                }
            )
    return annotator_path, key_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator-csv", type=Path, required=True)
    parser.add_argument("--key-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_blinded_annotation_files([], args.annotator_csv, args.key_csv)
    print(f"annotator_csv: {args.annotator_csv}")
    print(f"key_csv: {args.key_csv}")


if __name__ == "__main__":
    main()
