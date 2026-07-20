"""Run the frozen train-only GRF-to-PFF kinematic domain-gap audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.analysis.domain_gap import run_train_domain_gap_audit  # noqa: E402


def _format(value: Any) -> str:
    return f"{float(value):.4f}"


def _markdown(report: dict[str, Any], *, report_version: str = "V1") -> str:
    lines = [
        f"# GRF-to-PFF Train-Only Domain-Gap Audit {report_version}",
        "",
        "This report compares observable kinematic and geometric distributions only. It does not",
        "use validation/test examples or establish tactical or semantic concepts.",
        "",
        "## Sampling",
        "",
        f"- scope: `{report['scope']}`",
        f"- shared context examples per source: {report['sampling']['shared_context_examples']:,}",
        f"- PFF training matches represented: {report['real']['train_match_count']}",
        f"- PFF shards per training match: {report['sampling']['real_shards_per_match']}",
        "- GRF scenario cap per job shard: "
        f"{report['sampling']['synthetic_scenario_context_cap_per_shard']:,}",
        f"- deterministic seed: {report['seed']}",
        "",
        "## Largest Global Gaps",
        "",
        "Continuous-metric gaps divide quantile-Wasserstein distance by pooled interquartile",
        "scale, with median absolute deviation only as a zero-scale fallback. Rate metrics use",
        "the fixed probability range 1.0. Standard deviation never reduces a gap score.",
        "",
        "| Rank | Metric | Unit | Gap score | PFF mean | GRF mean | PFF median | GRF median |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(report["global_gap_ranking"][:12], start=1):
        lines.append(
            f"| {rank} | `{row['metric']}` | {row['unit']} | {_format(row['gap_score'])} | "
            f"{_format(row['real']['mean'])} | {_format(row['synthetic']['mean'])} | "
            f"{_format(row['real']['p50'])} | {_format(row['synthetic']['p50'])} |"
        )
    lines.extend(["", "## Scenario Diagnostics", ""])
    for scenario, rows in report["scenario_gap_rankings"].items():
        lines.append(f"### `{scenario}`")
        lines.append("")
        lines.append("| Metric | Gap score | PFF mean | Scenario mean |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in rows[:5]:
            lines.append(
                f"| `{row['metric']}` | {_format(row['gap_score'])} | "
                f"{_format(row['real']['mean'])} | {_format(row['synthetic']['mean'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            "Use these train-only measurements to freeze a targeted simulator or objective change.",
            "Do not tune against PFF validation, inspect PFF test, or interpret these measurements",
            "as learned tactical concepts.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-markdown", type=Path, required=True)
    parser.add_argument("--sample-examples", type=int, default=24576)
    parser.add_argument("--real-shards-per-match", type=int, default=4)
    parser.add_argument("--scenario-examples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--report-version", default="V1")
    args = parser.parse_args()

    report = run_train_domain_gap_audit(
        args.real_manifest,
        args.synthetic_manifest,
        sample_examples=args.sample_examples,
        real_shards_per_match=args.real_shards_per_match,
        scenario_examples=args.scenario_examples,
        seed=args.seed,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out_markdown.write_text(
        _markdown(report, report_version=args.report_version), encoding="utf-8"
    )
    print(f"json: {args.out_json}")
    print(f"markdown: {args.out_markdown}")
    print(f"shared_context_examples: {report['sampling']['shared_context_examples']}")
    for row in report["global_gap_ranking"][:10]:
        print(
            f"{row['metric']}: gap={row['gap_score']:.3f}, "
            f"pff_mean={row['real']['mean']:.3f}, grf_mean={row['synthetic']['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
