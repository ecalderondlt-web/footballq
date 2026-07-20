"""Audit PFF event-context shards and write a machine-readable report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.pff_event_context import audit_pff_event_context_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    report = audit_pff_event_context_dataset(
        args.manifest,
        require_train_only=args.train_only,
        verify_hashes=not args.skip_hashes,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"status: {report['status']}")
    print(f"audit: {output}")
    print(f"payload_sha256: {report['audit_payload_sha256']}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
