import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.cli import run_synthetic_demo  # noqa: E402

if __name__ == "__main__":
    outputs = run_synthetic_demo(ROOT / "artifacts" / "synthetic_demo")
    for name, path in outputs.items():
        print(f"{name}: {path}")

