import json

from scripts.summarize_discovery_controls import (
    parse_summary_spec,
    row_from_cluster_summary,
    summarize_rows,
)


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


def test_row_from_cluster_summary_adds_nuisance_fields(tmp_path):
    clusters_csv = tmp_path / "clusters_k2.csv"
    clusters_csv.write_text(
        "\n".join(
            [
                (
                    "cluster_id,n_examples,fraction,delta_norm_top_fraction,"
                    "train_count,val_count,test_count,match_id_counts"
                ),
                '0,10,0.5,0.2,6,2,2,"{""m1"": 7, ""m2"": 3}"',
                '1,10,0.5,0.0,8,1,1,"{""m1"": 5, ""m2"": 5}"',
            ]
        ),
        encoding="utf-8",
    )
    summary_json = tmp_path / "cluster_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "delta_seconds": 0.2,
                "scientific_mode": True,
                "split_manifest_sha256": "abc",
                "clusters": [
                    {
                        "k": 2,
                        "clusters_csv": str(clusters_csv),
                        "assignment_protocol": "fit_train_assign_all",
                        "quality": {
                            "k": 2,
                            "num_examples": 20,
                            "average_within_cluster_distance": 1.0,
                            "centroid_margin_proxy": 0.1,
                            "cluster_size_entropy": 1.0,
                            "empty_cluster_count": 0,
                            "max_cluster_size": 10,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    row = row_from_cluster_summary("normalized_delta_z", "7", summary_json, k=2)
    assert row["max_cluster_top_match_fraction"] == 0.7
    assert row["mean_cluster_top_match_fraction"] == 0.6
    assert row["max_delta_norm_top_fraction"] == 0.2
    assert row["min_heldout_examples_per_cluster"] == 2.0
