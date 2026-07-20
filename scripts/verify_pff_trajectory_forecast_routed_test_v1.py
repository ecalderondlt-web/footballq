"""Re-run the read-only artifact audit for the frozen routed PFF test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pff_trajectory_forecast_routed_test_v1 import (  # noqa: E402
    verify_artifacts,
)


def main() -> None:
    print(json.dumps(verify_artifacts(), indent=2))


if __name__ == "__main__":
    main()
