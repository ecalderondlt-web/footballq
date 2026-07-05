"""Render blinded GIF clips into static contact-sheet PNGs.

Annotators that cannot play animated GIFs (including model annotators) need a
static rendering of each blinded clip. This script reads only the
annotator-facing package (annotations.csv plus the GIFs it references) and
writes one PNG contact sheet per clip next to a sheets manifest. It never reads
private key files, so blinding is preserved: sheets carry only the blind clip
identity already exposed to annotators.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageSequence


def build_contact_sheet(
    gif_path: Path, max_frames: int, columns: int, tile_width: int
) -> Image.Image:
    with Image.open(gif_path) as gif:
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(gif)]
    if not frames:
        raise ValueError(f"no frames decoded from {gif_path}")
    if len(frames) > max_frames:
        step = (len(frames) - 1) / (max_frames - 1)
        indices = sorted({round(i * step) for i in range(max_frames)})
    else:
        indices = list(range(len(frames)))
    selected = [(idx, frames[idx]) for idx in indices]
    scale = tile_width / selected[0][1].width
    tile_height = int(selected[0][1].height * scale)
    label_height = 14
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + label_height)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    for position, (frame_idx, frame) in enumerate(selected):
        tile = frame.resize((tile_width, tile_height))
        col = position % columns
        row = position // columns
        x = col * tile_width
        y = row * (tile_height + label_height)
        sheet.paste(tile, (x, y + label_height))
        draw.text((x + 2, y + 1), f"frame {frame_idx}", fill=(0, 0, 0))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=320)
    args = parser.parse_args()

    package_root = args.annotator_csv.parent.parent
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    with args.annotator_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        clip_path = row.get("clip_path", "").strip()
        blind_id = row.get("blind_id", "").strip()
        if not clip_path:
            entries.append({"blind_id": blind_id, "sheet_path": "", "status": "missing_clip"})
            continue
        gif_path = Path(clip_path)
        if not gif_path.is_absolute():
            gif_path = package_root / clip_path
        if not gif_path.exists():
            entries.append({"blind_id": blind_id, "sheet_path": "", "status": "clip_not_found"})
            continue
        sheet = build_contact_sheet(gif_path, args.max_frames, args.columns, args.tile_width)
        sheet_path = args.out_dir / f"{blind_id}_sheet.png"
        sheet.save(sheet_path)
        entries.append(
            {"blind_id": blind_id, "sheet_path": str(sheet_path), "status": "rendered"}
        )
    manifest = {
        "annotator_csv": str(args.annotator_csv),
        "rendered": sum(1 for e in entries if e["status"] == "rendered"),
        "missing": sum(1 for e in entries if e["status"] != "rendered"),
        "entries": entries,
    }
    manifest_path = args.out_dir / "sheets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"rendered={manifest['rendered']} missing={manifest['missing']} out={args.out_dir}")


if __name__ == "__main__":
    main()
