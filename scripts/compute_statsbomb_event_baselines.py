"""Fit train-only raw event controls and score validation windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.analysis.statsbomb_event_baselines import (  # noqa: E402
    compute_statsbomb_event_baselines,
)
from footballq.data.statsbomb_events import write_immutable_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs" / "integrity" / "statsbomb_event_baselines_v1.json",
    )
    parser.add_argument("--laplace-alpha", type=float, default=1.0)
    args = parser.parse_args()

    report = compute_statsbomb_event_baselines(
        args.manifest,
        laplace_alpha=args.laplace_alpha,
    )
    write_immutable_json(args.out, report)
    print(f"frequency_nll: {report['global_frequency_event_type_nll']:.6f}")
    print(f"markov_nll: {report['first_order_markov_event_type_nll']:.6f}")
    print(f"copy_location_mae: {report['copy_current_location_mae']:.6f}")
    print(f"validation_targets: {int(report['validation_target_weight'])}")
    print(f"report_payload_sha256: {report['report_payload_sha256']}")


if __name__ == "__main__":
    main()
