from pipelineforge.config.settings import load_config
from pipelineforge.utils.logger import get_logger

logger = get_logger(__name__)


def run_training_pipeline() -> None:
    """
    Execute training workflow.
    """
    config = load_config()

    logger.info(
        "Running training pipeline for project: %s",
        config["project"]["name"],
    )

    logger.info("Pipeline execution completed.")


if __name__ == "__main__":
    run_training_pipeline()
