"""Project selected TD-JEPA shard splits to a narrower frozen feature view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.td_jepa_projection import project_td_jepa_feature_view  # noqa: E402
from footballq.repro.feature_views import FEATURE_VIEW_NAMES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-view", choices=sorted(FEATURE_VIEW_NAMES), required=True)
    parser.add_argument(
        "--include-split",
        action="append",
        dest="included_splits",
        help="Split to project; repeat for multiple splits. Defaults to every source split.",
    )
    args = parser.parse_args()

    path = project_td_jepa_feature_view(
        args.source_manifest,
        args.out,
        target_feature_view=args.feature_view,
        included_splits=set(args.included_splits) if args.included_splits else None,
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(f"dataset_manifest: {path}")
    print(f"feature_view: {manifest['feature_view']}")
    print(f"feature_names: {manifest['feature_names']}")
    print(f"included_splits: {manifest['included_splits']}")
    print(f"examples: {manifest['example_count']}")
    print(f"manifest_sha256: {manifest['manifest_payload_sha256']}")


if __name__ == "__main__":
    main()
