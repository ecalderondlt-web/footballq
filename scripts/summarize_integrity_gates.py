"""Combine integrity-control summaries into one paper-path gate artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_DISCOVERY_FEATURES = [
    "normalized_delta_z",
    "raw_delta_z",
    "pca_delta_z",
    "random_encoder_delta_z",
    "handcrafted_structure_metrics",
    "pca_handcrafted_structure_metrics",
]

RELEVANT_PROBE_CONTRASTS = [
    "raw_plus_td_jepa_vs_raw",
    "raw_plus_td_jepa_zscore_vs_raw_zscore",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def falsification_gate(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the TD falsification gate status."""

    blocking = list(payload.get("blocking_conditions", []))
    status = str(payload.get("scientific_claim_status", "unknown"))
    return {
        "status": status,
        "blocking_conditions": blocking,
        "pass_ratio": payload.get("pass_ratio"),
        "caution_ratio": payload.get("caution_ratio"),
    }


def probe_gate(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the incremental-probe gate status."""

    contrast_rows = []
    for item in payload.get("contrasts", {}).values():
        if item.get("contrast") not in RELEVANT_PROBE_CONTRASTS:
            continue
        improvement = item.get("signed_improvement", {})
        match_improvement = item.get("match_level_signed_improvement", {})
        contrast_rows.append(
            {
                "target": item.get("target"),
                "contrast": item.get("contrast"),
                "metric_name": item.get("metric_name"),
                "all_positive": improvement.get("all_positive"),
                "mean_signed_improvement": improvement.get("mean"),
                "min_signed_improvement": improvement.get("min"),
                "match_level_mean_signed_improvement": match_improvement.get("mean"),
            }
        )
    if not contrast_rows:
        status = "incomplete"
    else:
        status = "diagnostic_only"
    return {
        "status": status,
        "claim_status": payload.get("claim_status"),
        "contrasts": contrast_rows,
        "note": (
            "Current probe targets are geometry/control diagnostics; positive incremental "
            "value is not tactical evidence by itself."
        ),
    }


def discovery_gate(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the discovery-control gate status."""

    features = payload.get("features", {})
    missing = [feature for feature in REQUIRED_DISCOVERY_FEATURES if feature not in features]
    status = "incomplete" if missing else "diagnostic_only"
    primary = features.get("normalized_delta_z", {})
    raw = features.get("raw_delta_z", {})
    random = features.get("random_encoder_delta_z", {})
    return {
        "status": status,
        "missing_required_features": missing,
        "num_features": len(features),
        "primary_entropy_mean": primary.get("cluster_size_entropy", {}).get("mean"),
        "raw_entropy_mean": raw.get("cluster_size_entropy", {}).get("mean"),
        "random_entropy_mean": random.get("cluster_size_entropy", {}).get("mean"),
        "primary_top_match_fraction_max": primary.get(
            "max_cluster_top_match_fraction",
            {},
        ).get("max"),
        "note": (
            "Discovery summaries are controls and nuisance diagnostics; cluster outputs "
            "remain diagnostic until they beat controls and pass blinded enrichment."
        ),
    }


def _next_scientific_action(gates: dict[str, dict[str, Any]]) -> str:
    falsification = gates["falsification"]
    if falsification["status"] != "controls_passed":
        return (
            "Redesign or retrain the representation until falsification controls pass; "
            "use visualization only as blinded diagnostic material."
        )
    if gates["probe_incremental"]["status"] == "incomplete":
        return (
            "Run incremental probe controls comparing raw, z, raw+z, and z-scored "
            "feature views before discovery or visualization."
        )
    if gates["discovery_controls"]["status"] == "incomplete":
        return (
            "Run discovery baselines against raw/PCA/random controls before any "
            "blinded visualization."
        )
    return (
        "Review incremental-probe and discovery-control diagnostics against the "
        "paper gates; proceed only to blinded annotation, not unblinded claims."
    )


def combine_gates(
    falsification: dict[str, Any],
    probe: dict[str, Any],
    discovery: dict[str, Any],
) -> dict[str, Any]:
    """Combine gate summaries into one paper-path status."""

    gates = {
        "falsification": falsification_gate(falsification),
        "probe_incremental": probe_gate(probe),
        "discovery_controls": discovery_gate(discovery),
    }
    blocking = [
        name
        for name, gate in gates.items()
        if gate["status"] in {"blocked", "incomplete", "diagnostic_only"}
    ]
    return {
        "overall_claim_status": "blocked" if blocking else "controls_passed",
        "blocking_gates": blocking,
        "gates": gates,
        "allowed_claims": [
            "The current artifacts exercise reproducibility and diagnostic controls.",
            "Current clusters, residuals, and probe scores remain diagnostics only.",
        ],
        "next_scientific_action": _next_scientific_action(gates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--falsification", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = combine_gates(
        _read_json(args.falsification),
        _read_json(args.probe),
        _read_json(args.discovery),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary_json: {args.out}")
    print(f"overall_claim_status: {summary['overall_claim_status']}")
    print(f"blocking_gates: {', '.join(summary['blocking_gates'])}")


if __name__ == "__main__":
    main()
