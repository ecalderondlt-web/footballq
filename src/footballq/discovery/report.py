"""Discovery suite orchestration and report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from footballq.discovery.clustering import cluster_transition_file
from footballq.discovery.enrichment import write_enrichment_outputs
from footballq.discovery.exemplars import write_exemplars
from footballq.discovery.surprise import write_latent_residual_outputs
from footballq.discovery.transitions import (
    build_transition_dataset,
    save_transition_dataset,
    transition_summary,
)


def delta_label(delta_seconds: float) -> str:
    return f"delta_{str(round(float(delta_seconds), 3)).replace('.', 'p')}s"


def _read_csv(path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit is not None else rows


def _choose_k(k_values: list[int]) -> int:
    return 32 if 32 in k_values else sorted(k_values)[len(k_values) // 2]


def _write_report(summary: dict[str, Any], out: Path) -> Path:
    lines = [
        "# Experiment 5: Latent Transition Discovery",
        "",
        f"- transitions: {summary['transition_dataset']['num_examples']}",
        f"- matches: {summary['transition_dataset']['num_matches']}",
        "- delta horizons: "
        + ", ".join(
            f"{value:.3g}s" for value in summary["transition_dataset"]["requested_delta_seconds"]
        ),
        f"- recommended inspection k: {summary['recommended_k']}",
        "",
        "## Cluster Quality",
        "",
        "| delta | k | examples | avg distance | centroid sep | entropy | empty | "
        "centroid margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for delta in summary["deltas"]:
        for cluster in delta["cluster_summary"]["clusters"]:
            q = cluster["quality"]
            lines.append(
                f"| {delta['delta_label']} | {q['k']} | {q['num_examples']} | "
                f"{q['average_within_cluster_distance']:.4f} | "
                f"{q['centroid_separation_mean']:.4f} | {q['cluster_size_entropy']:.3f} | "
                f"{q['empty_cluster_count']} | {q['centroid_margin_proxy']:.3f} |"
            )
    lines.extend(["", "## Top Enriched Associations", ""])
    for delta in summary["deltas"]:
        lines.append(f"### {delta['delta_label']}")
        for item in delta.get("top_enriched", [])[:10]:
            lines.append(
                f"- cluster {item.get('cluster_id')}: {item.get('label')}="
                f"{item.get('value')} score={float(item.get('score') or 0):.3f} "
                f"support={item.get('support')}"
            )
        if not delta.get("top_enriched"):
            lines.append("- no enrichment rows available")
    lines.extend(["", "## Latent Residual", ""])
    for delta in summary["deltas"]:
        residual = delta["latent_residual_summary"]
        lines.append(
            f"- {delta['delta_label']}: score={residual['score_name']} "
            f"p95={residual['high_latent_residual_threshold']:.4f}; "
            "future-ball corr="
            f"{residual.get('correlations', {}).get('future_ball_displacement_corr')}"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Current SkillCorner-derived window metadata often has unknown phase/event labels.",
            "- Possession streams may be sparse; missing labels are preserved as unknown "
            "rather than fabricated.",
            "- Clusters are unsupervised and require human/video interpretation before "
            "tactical naming.",
            "- No visualization, text alignment, counterfactuals, or TD-JEPA fine-tuning "
            "are included here.",
            "",
            "## Recommended Next Step",
            "",
            "Run lightweight top-down rendering for cluster exemplars and high-residual "
            "windows, then begin manual tactical label annotation for recurring latent "
            "transition types.",
        ]
    )
    report_path = out / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_discovery_suite(
    embeddings: str | Path,
    windows: str | Path,
    out: str | Path,
    delta_steps: list[int],
    k_values: list[int],
    fps: float = 10.0,
    feature: str = "normalized_delta_z",
    seed: int = 123,
    max_iter: int = 30,
    fit_sample_size: int = 50000,
    split_manifest_path: str | Path | None = None,
    scientific_mode: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "transition_dataset.pt"
    data = build_transition_dataset(
        embeddings,
        windows,
        out=dataset_path,
        delta_steps=delta_steps,
        fps=fps,
        split_manifest_path=split_manifest_path,
        scientific_mode=scientific_mode,
    )
    transition_summary_payload = transition_summary(data)
    transition_summary_path = out_dir / "transition_dataset_summary.json"
    transition_summary_path.write_text(
        json.dumps(transition_summary_payload, indent=2),
        encoding="utf-8",
    )
    # Save a copy with any feature tensors produced by the builder.
    save_transition_dataset(data, dataset_path)

    chosen_k = _choose_k([int(value) for value in k_values])
    delta_outputs = []
    for delta_seconds in transition_summary_payload["requested_delta_seconds"]:
        label = delta_label(float(delta_seconds))
        delta_dir = out_dir / label
        cluster_summary = cluster_transition_file(
            dataset_path,
            delta_dir,
            [int(value) for value in k_values],
            delta_seconds=float(delta_seconds),
            feature=feature,
            seed=seed,
            max_iter=max_iter,
            fit_sample_size=fit_sample_size,
        )
        chosen = [item for item in cluster_summary["clusters"] if int(item["k"]) == chosen_k][0]
        enrichment = write_enrichment_outputs(
            dataset_path,
            chosen["assignments"],
            delta_dir / f"enrichment_k{chosen_k}.csv",
            delta_dir / "enrichment_summary.json",
        )
        residual = write_latent_residual_outputs(
            dataset_path,
            delta_dir,
            delta_seconds=float(delta_seconds),
            assignments=chosen["assignments"],
        )
        exemplars_path = write_exemplars(
            dataset_path,
            chosen["assignments"],
            delta_dir / f"exemplars_k{chosen_k}.csv",
            seed=seed,
        )
        delta_outputs.append(
            {
                "delta_seconds": float(delta_seconds),
                "delta_label": label,
                "out_dir": str(delta_dir),
                "cluster_summary": cluster_summary,
                "enrichment_summary": enrichment,
                "latent_residual_summary": residual["summary"],
                "exemplars": str(exemplars_path),
                "top_enriched": _read_csv(delta_dir / f"enrichment_k{chosen_k}.csv", limit=20),
            }
        )

    summary = {
        "embeddings": str(embeddings),
        "windows": str(windows),
        "out": str(out_dir),
        "transition_dataset": transition_summary_payload,
        "transition_dataset_path": str(dataset_path),
        "recommended_k": chosen_k,
        "k_values": [int(value) for value in k_values],
        "feature": feature,
        "deltas": delta_outputs,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = _write_report(summary, out_dir)
    summary["summary_json"] = str(summary_path)
    summary["report_md"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
