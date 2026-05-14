from pathlib import Path

from pipelineforge.config.settings import load_config
from pipelineforge.core.logging import setup_logger

logger = setup_logger(__name__)


def run_training_pipeline() -> None:
    """
    Execute training workflow.
    """

    config_path = Path("configs/default.toml")

    config = load_config(config_path)

    logger.info(
        "Running training pipeline | model=%s | target=%s",
        config.training.model_type,
        config.data.target_column,
    )

    logger.info("Training pipeline execution completed.")


if __name__ == "__main__":
    run_training_pipeline()
