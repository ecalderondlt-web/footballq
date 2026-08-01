"""Hash the immutable StatsBomb Open Data source snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from footballq.analysis.wyscout_player_memory import (
    file_sha256,
    stable_payload_hash,
)

SOURCE_SECTIONS = ("events", "lineups", "matches", "three-sixty")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise ValueError(f"StatsBomb source root is missing: {root}")
    paths = [root / "competitions.json"]
    for section in SOURCE_SECTIONS:
        section_root = root / section
        if not section_root.is_dir():
            raise ValueError(f"StatsBomb source section is missing: {section_root}")
        paths.extend(
            path
            for path in section_root.rglob("*")
            if path.is_file()
        )
    paths = sorted(paths, key=lambda path: path.relative_to(root).as_posix())
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    payload = {
        "name": "statsbomb_open_data_source_inventory_v1",
        "version": 1,
        "source_root": str(root),
        "source_commit": str(args.source_commit),
        "included_sections": ["competitions.json", *SOURCE_SECTIONS],
        "file_count": len(files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        "files": files,
    }
    payload["inventory_payload_sha256"] = stable_payload_hash(payload)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "file_count": payload["file_count"],
                "total_size_bytes": payload["total_size_bytes"],
                "inventory_payload_sha256": payload[
                    "inventory_payload_sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
