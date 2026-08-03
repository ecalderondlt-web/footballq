from __future__ import annotations

import argparse
from pathlib import Path

from footballq.data.rlcs_ballchasing import (
    DEFAULT_ROOT_LABELS,
    acquire_rlcs,
    load_ballchasing_token,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory and download the preregistered RLCS 2025 replay corpus."
    )
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw/rlcs_2025"))
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--download-rps", type=float, default=1.0)
    parser.add_argument("--download-hourly-cap", type=int, default=200)
    parser.add_argument("--download-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    token = load_ballchasing_token()
    if not token:
        parser.error(
            "Set BALLCHASING_TOKEN in the environment or in the ignored .env file "
            "before acquisition."
        )
    unknown = sorted(set(args.groups) - set(DEFAULT_ROOT_LABELS))
    if unknown:
        print("warning: unrecognized root group IDs: " + ", ".join(unknown))
    path = acquire_rlcs(
        token=token,
        root_group_ids=[str(value) for value in args.groups],
        output_dir=args.output,
        page_size=args.page_size,
        requests_per_second=args.download_rps,
        hourly_cap=args.download_hourly_cap,
        resume=args.resume,
        download_limit=args.download_limit,
    )
    print(f"inventory: {path}")


if __name__ == "__main__":
    main()
