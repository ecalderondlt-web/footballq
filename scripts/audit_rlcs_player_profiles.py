from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from footballq.data.rlcs_player_profiles import audit_profile_stability
from footballq.repro.manifest import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen RLCS V2 profile gate.")
    parser.add_argument("--config", default="configs/rlcs_player_matchup_value_v2.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    profile_cfg = config["profiles"]
    games_path = Path(data_cfg["profile_games"])
    priors_path = Path(data_cfg["profile_priors"])
    games = pd.read_parquet(games_path)
    priors = json.loads(priors_path.read_text(encoding="utf-8"))
    report = audit_profile_stability(
        games,
        priors,
        minimum_games=int(profile_cfg["minimum_support_games"]),
        required_complete_players=int(profile_cfg["required_complete_eligible_players"]),
        minimum_players_per_region=int(profile_cfg["minimum_eligible_players_per_region"]),
        retrieval_auc_minimum=float(profile_cfg["retrieval_auc_minimum"]),
        retrieval_auc_bootstrap_lower_minimum=float(
            profile_cfg["retrieval_auc_bootstrap_lower_minimum"]
        ),
        regional_retrieval_auc_minimum_exclusive=float(
            profile_cfg["regional_retrieval_auc_minimum_exclusive"]
        ),
        median_spearman_minimum=float(profile_cfg["median_core_spearman_minimum"]),
        median_spearman_bootstrap_lower_minimum_exclusive=float(
            profile_cfg["median_core_spearman_bootstrap_lower_minimum_exclusive"]
        ),
        bootstrap_resamples=int(profile_cfg["player_bootstrap_resamples"]),
        bootstrap_seed=int(profile_cfg["player_bootstrap_seed"]),
    )
    report["inputs"] = {
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "profile_games": str(games_path),
        "profile_games_sha256": file_sha256(games_path),
        "profile_priors": str(priors_path),
        "profile_priors_sha256": file_sha256(priors_path),
    }
    destination = Path(data_cfg["profile_audit"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    status = "PASS" if report["all_gates_pass"] else "STOP"
    print(
        f"profile gate: {status}; eligible={report['counts']['eligible_players']}; "
        f"retrieval_auc={report['same_player_retrieval_auc']:.4f}; "
        f"auc_lower={report['player_bootstrap']['retrieval_auc_95pct'][0]:.4f}; "
        f"median_spearman={report['median_core_trait_spearman']:.4f}; "
        f"spearman_lower={report['player_bootstrap']['median_spearman_95pct'][0]:.4f}"
    )
    print(f"report: {destination}")


if __name__ == "__main__":
    main()
