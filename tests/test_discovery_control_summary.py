from scripts.summarize_discovery_controls import parse_summary_spec, summarize_rows


def test_parse_summary_spec():
    feature, seed, path = parse_summary_spec("normalized_delta_z:7:runs/example.json")
    assert feature == "normalized_delta_z"
    assert seed == "7"
    assert str(path) == "runs\\example.json" or str(path) == "runs/example.json"


def test_summarize_rows_groups_by_feature():
    rows = [
        {
            "feature": "a",
            "seed": "1",
            "average_within_cluster_distance": 1.0,
            "centroid_margin_proxy": 0.1,
            "cluster_size_entropy": 0.8,
            "empty_cluster_count": 0,
            "max_cluster_size_fraction": 0.5,
            "assignment_protocol": "fit_train_assign_all",
        },
        {
            "feature": "a",
            "seed": "2",
            "average_within_cluster_distance": 3.0,
            "centroid_margin_proxy": 0.3,
            "cluster_size_entropy": 0.6,
            "empty_cluster_count": 1,
            "max_cluster_size_fraction": 0.7,
            "assignment_protocol": "fit_train_assign_all",
        },
    ]
    summary = summarize_rows(rows)
    assert summary["a"]["average_within_cluster_distance"]["mean"] == 2.0
    assert summary["a"]["seeds"] == ["1", "2"]
