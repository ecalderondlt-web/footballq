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
    parser.add_argument("--examples", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=40)
    parser.add_argument("--blinded", action="store_true")
    parser.add_argument("--annotator-csv", type=Path, default=None)
    parser.add_argument("--key-csv", type=Path, default=None)
    return parser.parse_args()


def _read_examples(path: Path, max_rows: int) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[: int(max_rows)]


def main() -> None:
    args = parse_args()
    if args.examples is not None:
        if args.out is None:
            raise ValueError("--examples requires --out.")
        rows = _read_examples(args.examples, args.max_rows)
        annotator_csv = args.out / "annotator" / "annotations.csv"
        key_csv = args.out / "private" / "annotation_key.csv"
    else:
        if args.annotator_csv is None or args.key_csv is None:
            raise ValueError("Set --examples/--out or --annotator-csv/--key-csv.")
        rows = []
        annotator_csv = args.annotator_csv
        key_csv = args.key_csv
    write_blinded_annotation_files(rows, annotator_csv, key_csv)
    print(f"annotator_csv: {annotator_csv}")
    print(f"key_csv: {key_csv}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
