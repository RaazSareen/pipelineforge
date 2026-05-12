from pipelineforge.config.settings import load_config


def test_load_config():
    config = load_config()

    assert config["project"]["name"] == "pipelineforge"
