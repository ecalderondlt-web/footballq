import json

from footballq.training.train_statsbomb_event import (
    create_statsbomb_event_model,
    load_statsbomb_event_config,
)


def _manifest():
    return {
        "categorical_feature_names": [
            "event_type",
            "play_pattern",
            "position",
            "subtype",
            "outcome",
        ],
        "continuous_feature_names": [f"feature_{index}" for index in range(17)],
        "freeze_frame_feature_names": [f"feature_{index}" for index in range(6)],
        "categorical_vocabularies": {
            "event_type": {"size": 37},
            "play_pattern": {"size": 11},
            "position": {"size": 28},
            "subtype": {"size": 33},
            "outcome": {"size": 38},
        },
    }


def test_statsbomb_event_config_and_model_feature_views(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  feature_view: event_only\n  d_model: 32\n  n_heads: 4\n  n_layers: 1\n",
        encoding="utf-8",
    )
    config = load_statsbomb_event_config(path)
    model = create_statsbomb_event_model(config, _manifest())

    assert config["_config_path"] == str(path)
    assert model.use_360 is False


def test_statsbomb_event_model_rejects_unknown_feature_view():
    config = {"model": {"feature_view": "fabricated_join"}}
    try:
        create_statsbomb_event_model(config, _manifest())
    except ValueError as exc:
        assert "feature_view" in str(exc)
    else:
        raise AssertionError("Expected invalid StatsBomb feature view to fail.")


def test_statsbomb_smoke_configs_are_two_update_diagnostics():
    for name in (
        "statsbomb_event_encoder_smoke_event_only_v1.yaml",
        "statsbomb_event_encoder_smoke_event_plus_360_v1.yaml",
    ):
        config = load_statsbomb_event_config(f"configs/{name}")
        assert config["training"]["max_train_updates"] == 2
        assert config["training"]["val_max_batches"] == 2
        assert json.loads(json.dumps(config["data"]))["manifest"].endswith("manifest.json")
