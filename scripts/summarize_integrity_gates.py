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

DISCOVERY_CONTROL_FEATURES = [
    "raw_delta_z",
    "pca_delta_z",
    "random_encoder_delta_z",
]
DISCOVERY_SEPARATION_MARGIN = 0.02
MIN_HELDOUT_EXAMPLES_PER_CLUSTER = 5
MAX_DELTA_NORM_TOP_FRACTION = 0.95


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _metric_stat(feature: dict[str, Any], metric: str, stat: str) -> float | None:
    value = feature.get(metric, {})
    if not isinstance(value, dict):
        return None
    return _as_float(value.get(stat))


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
    blocking_conditions = []
    for item in payload.get("contrasts", {}).values():
        if item.get("contrast") not in RELEVANT_PROBE_CONTRASTS:
            continue
        improvement = item.get("signed_improvement", {})
        match_improvement = item.get("match_level_signed_improvement", {})
        target = item.get("target")
        contrast = item.get("contrast")
        row_id = f"{target}:{contrast}"
        seed_all_positive = improvement.get("all_positive")
        match_all_positive = match_improvement.get("all_positive")
        seed_mean = _as_float(improvement.get("mean"))
        seed_min = _as_float(improvement.get("min"))
        match_mean = _as_float(match_improvement.get("mean"))
        match_min = _as_float(match_improvement.get("min"))
        if seed_all_positive is False or seed_mean is None or seed_mean <= 0:
            blocking_conditions.append(f"nonpositive_seed_increment:{row_id}")
        if seed_min is not None and seed_min <= 0:
            blocking_conditions.append(f"negative_seed_increment:{row_id}")
        if (
            match_all_positive is False
            or match_mean is None
            or match_mean <= 0
        ):
            blocking_conditions.append(f"nonpositive_match_increment:{row_id}")
        if match_min is not None and match_min <= 0:
            blocking_conditions.append(f"negative_match_increment:{row_id}")
        contrast_rows.append(
            {
                "target": target,
                "contrast": contrast,
                "metric_name": item.get("metric_name"),
                "all_positive": seed_all_positive,
                "mean_signed_improvement": seed_mean,
                "min_signed_improvement": seed_min,
                "match_level_all_positive": match_all_positive,
                "match_level_mean_signed_improvement": match_mean,
                "match_level_min_signed_improvement": match_min,
            }
        )
    if not contrast_rows:
        status = "incomplete"
    else:
        status = "diagnostic_only"
    return {
        "status": status,
        "claim_status": payload.get("claim_status"),
        "blocking_conditions": sorted(set(blocking_conditions)),
        "positive_contrast_count": sum(
            row["all_positive"] is not False
            and row["match_level_all_positive"] is not False
            and row["mean_signed_improvement"] is not None
            and row["mean_signed_improvement"] > 0
            and row["match_level_mean_signed_improvement"] is not None
            and row["match_level_mean_signed_improvement"] > 0
            for row in contrast_rows
        ),
        "contrast_count": len(contrast_rows),
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
    primary = features.get("normalized_delta_z", {})
    control_rows = []
    for feature_name in DISCOVERY_CONTROL_FEATURES:
        feature = features.get(feature_name, {})
        control_rows.append(
            {
                "feature": feature_name,
                "entropy_mean": _metric_stat(feature, "cluster_size_entropy", "mean"),
                "top_match_fraction_mean": _metric_stat(
                    feature,
                    "max_cluster_top_match_fraction",
                    "mean",
                ),
            }
        )

    primary_entropy_mean = _metric_stat(primary, "cluster_size_entropy", "mean")
    primary_top_match_mean = _metric_stat(
        primary,
        "max_cluster_top_match_fraction",
        "mean",
    )
    primary_top_match_max = _metric_stat(
        primary,
        "max_cluster_top_match_fraction",
        "max",
    )
    min_heldout_examples = _metric_stat(
        primary,
        "min_heldout_examples_per_cluster",
        "min",
    )
    max_delta_norm_top_fraction = _metric_stat(
        primary,
        "max_delta_norm_top_fraction",
        "max",
    )
    control_entropy_values = [
        row["entropy_mean"] for row in control_rows if row["entropy_mean"] is not None
    ]
    control_top_match_values = [
        row["top_match_fraction_mean"]
        for row in control_rows
        if row["top_match_fraction_mean"] is not None
    ]
    best_control_entropy = max(control_entropy_values, default=None)
    best_control_top_match = min(control_top_match_values, default=None)
    entropy_margin_vs_best_control = (
        None
        if primary_entropy_mean is None or best_control_entropy is None
        else primary_entropy_mean - best_control_entropy
    )
    top_match_margin_vs_best_control = (
        None
        if primary_top_match_mean is None or best_control_top_match is None
        else best_control_top_match - primary_top_match_mean
    )
    blocking_conditions = []
    if missing:
        blocking_conditions.extend(f"missing_feature:{feature}" for feature in missing)
    if not missing:
        if (
            entropy_margin_vs_best_control is None
            or entropy_margin_vs_best_control <= DISCOVERY_SEPARATION_MARGIN
        ):
            blocking_conditions.append("latent_entropy_not_separated_from_controls")
        if (
            top_match_margin_vs_best_control is None
            or top_match_margin_vs_best_control <= DISCOVERY_SEPARATION_MARGIN
        ):
            blocking_conditions.append(
                "latent_match_concentration_not_separated_from_controls"
            )
        if (
            min_heldout_examples is not None
            and min_heldout_examples < MIN_HELDOUT_EXAMPLES_PER_CLUSTER
        ):
            blocking_conditions.append("sparse_heldout_clusters")
        if (
            max_delta_norm_top_fraction is not None
            and max_delta_norm_top_fraction >= MAX_DELTA_NORM_TOP_FRACTION
        ):
            blocking_conditions.append("transition_magnitude_concentration")
    status = "incomplete" if missing else "diagnostic_only"
    return {
        "status": status,
        "blocking_conditions": sorted(set(blocking_conditions)),
        "missing_required_features": missing,
        "num_features": len(features),
        "primary_entropy_mean": primary_entropy_mean,
        "best_control_entropy_mean": best_control_entropy,
        "entropy_margin_vs_best_control": entropy_margin_vs_best_control,
        "primary_top_match_fraction_mean": primary_top_match_mean,
        "primary_top_match_fraction_max": primary_top_match_max,
        "best_control_top_match_fraction_mean": best_control_top_match,
        "top_match_margin_vs_best_control": top_match_margin_vs_best_control,
        "min_heldout_examples_per_cluster": min_heldout_examples,
        "max_delta_norm_top_fraction": max_delta_norm_top_fraction,
        "control_comparisons": control_rows,
        "note": (
            "Discovery summaries are controls and nuisance diagnostics; cluster outputs "
            "remain diagnostic until they beat controls and pass blinded enrichment."
        ),
    }


def annotation_gate(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the blinded-annotation gate status."""

    annotation_status = str(payload.get("annotation_status", "incomplete"))
    if annotation_status in {"invalid_package", "invalid_labels"}:
        status = "blocked"
    elif annotation_status == "analyzed":
        status = "diagnostic_only"
    else:
        status = "incomplete"
    enrichment = payload.get("enrichment", {})
    return {
        "status": status,
        "annotation_status": annotation_status,
        "completed_count": payload.get("completed_count"),
        "completion_rate": payload.get("completion_rate"),
        "positive_group_positive_label_rate": enrichment.get(
            "positive_group_positive_label_rate"
        ),
        "control_group_positive_label_rate": enrichment.get(
            "control_group_positive_label_rate"
        ),
        "risk_difference": enrichment.get("risk_difference"),
        "fisher_greater_pvalue": enrichment.get("fisher_greater_pvalue"),
        "note": (
            "Blinded annotation is required before interpretive claims; annotation "
            "summaries remain diagnostic until compared against matched controls."
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
    if gates["probe_incremental"].get("blocking_conditions"):
        return (
            "Resolve mixed incremental probe results or redesign the representation "
            "before treating discovery or annotation as evidence."
        )
    if gates["discovery_controls"]["status"] == "incomplete":
        return (
            "Run discovery baselines against raw/PCA/random controls before any "
            "blinded visualization."
        )
    if gates["discovery_controls"].get("blocking_conditions"):
        return (
            "Improve discovery separation from raw/PCA/random controls before using "
            "blinded annotation for interpretive claims."
        )
    annotation = gates.get("blinded_annotation")
    if annotation is None:
        return (
            "Run blinded annotation against matched controls before any interpretive "
            "claims."
        )
    if annotation["status"] == "incomplete":
        return "Complete blinded annotation and analyze enrichment against matched controls."
    if annotation["status"] == "blocked":
        return "Fix the blinded annotation package before using annotation results."
    return (
        "Review incremental-probe, discovery-control, and blinded-annotation diagnostics "
        "against the paper gates; do not make unblinded tactical claims."
    )


def combine_gates(
    falsification: dict[str, Any],
    probe: dict[str, Any],
    discovery: dict[str, Any],
    annotation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine gate summaries into one paper-path status."""

    gates = {
        "falsification": falsification_gate(falsification),
        "probe_incremental": probe_gate(probe),
        "discovery_controls": discovery_gate(discovery),
    }
    if annotation is not None:
        gates["blinded_annotation"] = annotation_gate(annotation)
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


def blocking_condition_lines(summary: dict[str, Any]) -> list[str]:
    """Return CLI-readable blocking-condition lines from a gate summary."""

    lines = []
    gates = summary.get("gates", {})
    if not isinstance(gates, dict):
        return lines
    for gate_name, gate in gates.items():
        if not isinstance(gate, dict):
            continue
        for condition in gate.get("blocking_conditions", []):
            lines.append(f"blocking_condition[{gate_name}]: {condition}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--falsification", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = combine_gates(
        _read_json(args.falsification),
        _read_json(args.probe),
        _read_json(args.discovery),
        _read_json(args.annotation) if args.annotation is not None else None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary_json: {args.out}")
    print(f"overall_claim_status: {summary['overall_claim_status']}")
    print(f"blocking_gates: {', '.join(summary['blocking_gates'])}")
    print(f"next_scientific_action: {summary['next_scientific_action']}")
    for line in blocking_condition_lines(summary):
        print(line)


if __name__ == "__main__":
    main()
