from scripts.summarize_integrity_gates import blocking_condition_lines, combine_gates


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
                "cluster_size_entropy": {"mean": 0.91},
                "max_cluster_top_match_fraction": {"mean": 0.1, "max": 0.12},
                "min_heldout_examples_per_cluster": {"min": 12},
                "max_delta_norm_top_fraction": {"max": 0.2},
            },
            "raw_delta_z": {
                "cluster_size_entropy": {"mean": 0.84},
                "max_cluster_top_match_fraction": {"mean": 0.17},
            },
            "pca_delta_z": {
                "cluster_size_entropy": {"mean": 0.83},
                "max_cluster_top_match_fraction": {"mean": 0.18},
            },
            "random_encoder_delta_z": {
                "cluster_size_entropy": {"mean": 0.82},
                "max_cluster_top_match_fraction": {"mean": 0.19},
            },
            "handcrafted_structure_metrics": {},
            "pca_handcrafted_structure_metrics": {},
        }
    }


def _blocked_discovery_payload():
    return {
        "features": {
            "normalized_delta_z": {
                "cluster_size_entropy": {"mean": 0.83},
                "max_cluster_top_match_fraction": {"mean": 0.24, "max": 0.27},
                "min_heldout_examples_per_cluster": {"min": 4},
                "max_delta_norm_top_fraction": {"max": 1.0},
            },
            "raw_delta_z": {
                "cluster_size_entropy": {"mean": 0.84},
                "max_cluster_top_match_fraction": {"mean": 0.23},
            },
            "pca_delta_z": {
                "cluster_size_entropy": {"mean": 0.85},
                "max_cluster_top_match_fraction": {"mean": 0.22},
            },
            "random_encoder_delta_z": {
                "cluster_size_entropy": {"mean": 0.835},
                "max_cluster_top_match_fraction": {"mean": 0.25},
            },
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


def test_combine_gates_records_mixed_probe_blocking_conditions():
    mixed_probe = _diagnostic_probe_payload()
    contrast = mixed_probe["contrasts"]["target|linear|raw_plus_td_jepa_vs_raw"]
    contrast["signed_improvement"] = {
        "all_positive": False,
        "mean": -0.01,
        "min": -0.02,
    }
    contrast["match_level_signed_improvement"] = {
        "all_positive": False,
        "mean": -0.005,
        "min": -0.01,
    }

    summary = combine_gates(
        _passing_falsification_payload(),
        mixed_probe,
        _complete_discovery_payload(),
    )

    conditions = summary["gates"]["probe_incremental"]["blocking_conditions"]
    assert any(condition.startswith("nonpositive_seed_increment") for condition in conditions)
    assert any(condition.startswith("negative_match_increment") for condition in conditions)
    assert "Resolve mixed incremental probe results" in summary["next_scientific_action"]


def test_combine_gates_records_discovery_control_blocking_conditions():
    summary = combine_gates(
        _passing_falsification_payload(),
        _diagnostic_probe_payload(),
        _blocked_discovery_payload(),
    )

    conditions = summary["gates"]["discovery_controls"]["blocking_conditions"]
    assert "latent_entropy_not_separated_from_controls" in conditions
    assert "latent_match_concentration_not_separated_from_controls" in conditions
    assert "sparse_heldout_clusters" in conditions
    assert "transition_magnitude_concentration" in conditions
    assert "Improve discovery separation" in summary["next_scientific_action"]


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


def test_combine_gates_blocks_invalid_annotation_labels():
    summary = combine_gates(
        _passing_falsification_payload(),
        _diagnostic_probe_payload(),
        _complete_discovery_payload(),
        {
            "annotation_status": "invalid_labels",
            "completed_count": 1,
            "completion_rate": 0.025,
            "enrichment": {},
        },
    )

    assert summary["overall_claim_status"] == "blocked"
    assert summary["gates"]["blinded_annotation"]["status"] == "blocked"
    assert "Fix the blinded annotation package" in summary["next_scientific_action"]


def test_blocking_condition_lines_formats_gate_names():
    lines = blocking_condition_lines(
        {
            "gates": {
                "probe_incremental": {
                    "blocking_conditions": ["negative_seed_increment:target:contrast"]
                },
                "discovery_controls": {
                    "blocking_conditions": ["latent_entropy_not_separated_from_controls"]
                },
            }
        }
    )

    assert lines == [
        "blocking_condition[probe_incremental]: negative_seed_increment:target:contrast",
        "blocking_condition[discovery_controls]: latent_entropy_not_separated_from_controls",
    ]
