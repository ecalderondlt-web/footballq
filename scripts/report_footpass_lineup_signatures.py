"""Inventory anonymized FOOTPASS lineups and candidate repeated teams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.io.footpass import (  # noqa: E402
    extract_footpass_lineup_signatures,
    rank_footpass_lineup_matches,
)
from footballq.repro.manifest import build_run_manifest, write_run_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "splits" / "footpass_train48_development_v1.json",
    )
    parser.add_argument("--minimum-overlap", type=int, default=5)
    parser.add_argument("--top-pairs", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signatures = extract_footpass_lineup_signatures(args.h5)
    ranked = rank_footpass_lineup_matches(
        signatures,
        minimum_overlap=args.minimum_overlap,
    )
    payload = {
        "version": 1,
        "dataset": "footpass_train_tactical",
        "claim_status": "identity_candidate_generation_only",
        "method": (
            "first_observed_H1_lineup; inverse-frequency-weighted shirt-number "
            "Jaccard; candidate pairs require external fixture verification"
        ),
        "appearance_count": len(signatures),
        "candidate_pair_count": len(ranked),
        "lineup_signatures": [signature.to_dict() for signature in signatures],
        "ranked_candidate_pairs": ranked[: args.top_pairs],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    manifest = build_run_manifest(
        command=sys.argv,
        config_path=None,
        split_manifest_path=args.split_manifest,
        evaluation_protocol="footpass_lineup_signature_inventory_v1",
        feature_view="shirt_numbers_and_roles_only_no_model_features",
        objective_mode="not_applicable_provenance_recovery",
        dataset_paths={"footpass_tactical_h5": args.h5},
        output_paths={"lineup_signature_report": args.out},
        warnings=[
            "Candidate pairs are not persistent identities until externally verified.",
            "Common shirt-number overlap can create false-positive team matches.",
            "This report is not a model result or tactical-understanding claim.",
        ],
    )
    write_run_manifest(args.out.parent / "run_manifest.json", manifest)
    print(f"appearances: {len(signatures)}")
    print(f"candidate_pairs: {len(ranked)}")
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
