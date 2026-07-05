from scripts.summarize_integrity_gates import combine_gates


def test_combine_gates_blocks_diagnostic_outputs():
    summary = combine_gates(
        {
            "scientific_claim_status": "blocked",
            "blocking_conditions": ["team_swap"],
            "pass_ratio": 1.25,
            "caution_ratio": 1.05,
        },
        {
            "claim_status": "diagnostic_only",
            "contrasts": {
                "target|linear|raw_plus_td_jepa_vs_raw": {
                    "target": "future_ball_global_x_bucket",
                    "contrast": "raw_plus_td_jepa_vs_raw",
                    "metric_name": "macro_f1",
                    "signed_improvement": {
                        "all_positive": True,
                        "mean": 0.1,
                        "min": 0.05,
                    },
                    "match_level_signed_improvement": {"mean": 0.09},
                }
            },
        },
        {
            "features": {
                "normalized_delta_z": {
                    "cluster_size_entropy": {"mean": 0.87},
                    "max_cluster_top_match_fraction": {"max": 0.16},
                },
                "raw_delta_z": {"cluster_size_entropy": {"mean": 0.88}},
                "pca_delta_z": {},
                "random_encoder_delta_z": {"cluster_size_entropy": {"mean": 0.86}},
                "handcrafted_structure_metrics": {},
                "pca_handcrafted_structure_metrics": {},
            }
        },
    )
    assert summary["overall_claim_status"] == "blocked"
    assert summary["gates"]["falsification"]["blocking_conditions"] == ["team_swap"]
    assert summary["gates"]["probe_incremental"]["status"] == "diagnostic_only"
    assert summary["gates"]["discovery_controls"]["missing_required_features"] == []
    assert "falsification controls pass" in summary["next_scientific_action"]


def test_combine_gates_advances_next_action_after_falsification_passes():
    summary = combine_gates(
        {
            "scientific_claim_status": "controls_passed",
            "blocking_conditions": [],
            "pass_ratio": 1.25,
            "caution_ratio": 1.05,
        },
        {
            "claim_status": "diagnostic_only",
            "contrasts": {
                "target|linear|raw_plus_td_jepa_vs_raw": {
                    "target": "future_ball_displacement_m",
                    "contrast": "raw_plus_td_jepa_vs_raw",
                    "metric_name": "rmse",
                    "signed_improvement": {
                        "all_positive": True,
                        "mean": 0.1,
                        "min": 0.05,
                    },
                    "match_level_signed_improvement": {"mean": 0.09},
                }
            },
        },
        {"features": {}},
    )

    assert summary["overall_claim_status"] == "blocked"
    assert summary["blocking_gates"] == ["probe_incremental", "discovery_controls"]
    assert "Run discovery baselines" in summary["next_scientific_action"]


def _passing_falsification_payload():
    return {
        "scientific_claim_status": "controls_passed",
        "blocking_conditions": [],
        "pass_ratio": 1.25,
        "caution_ratio": 1.05,
    }


def _diagnostic_probe_payload():
    return {
        "claim_status": "diagnostic_only",
        "contrasts": {
            "target|linear|raw_plus_td_jepa_vs_raw": {
                "target": "future_ball_displacement_m",
                "contrast": "raw_plus_td_jepa_vs_raw",
                "metric_name": "rmse",
                "signed_improvement": {
                    "all_positive": True,
                    "mean": 0.1,
                    "min": 0.05,
                },
                "match_level_signed_improvement": {"mean": 0.09},
            }
        },
    }


def _complete_discovery_payload():
    return {
        "features": {
            "normalized_delta_z": {
                "cluster_size_entropy": {"mean": 0.87},
                "max_cluster_top_match_fraction": {"max": 0.16},
            },
            "raw_delta_z": {"cluster_size_entropy": {"mean": 0.88}},
            "pca_delta_z": {},
            "random_encoder_delta_z": {"cluster_size_entropy": {"mean": 0.86}},
            "handcrafted_structure_metrics": {},
            "pca_handcrafted_structure_metrics": {},
        }
    }


def test_combine_gates_blocks_incomplete_blinded_annotation():
    summary = combine_gates(
        _passing_falsification_payload(),
        _diagnostic_probe_payload(),
        _complete_discovery_payload(),
        {
            "annotation_status": "incomplete",
            "completed_count": 0,
            "completion_rate": 0.0,
            "enrichment": {},
        },
    )

    assert summary["overall_claim_status"] == "blocked"
    assert summary["gates"]["blinded_annotation"]["status"] == "incomplete"
    assert "Complete blinded annotation" in summary["next_scientific_action"]


def test_combine_gates_keeps_completed_annotation_diagnostic_only():
    summary = combine_gates(
        _passing_falsification_payload(),
        _diagnostic_probe_payload(),
        _complete_discovery_payload(),
        {
            "annotation_status": "analyzed",
            "completed_count": 40,
            "completion_rate": 1.0,
            "enrichment": {
                "positive_group_positive_label_rate": 0.8,
                "control_group_positive_label_rate": 0.4,
                "risk_difference": 0.4,
                "fisher_greater_pvalue": 0.03,
            },
        },
    )

    assert summary["overall_claim_status"] == "blocked"
    assert summary["gates"]["blinded_annotation"]["status"] == "diagnostic_only"
    assert summary["gates"]["blinded_annotation"]["risk_difference"] == 0.4
