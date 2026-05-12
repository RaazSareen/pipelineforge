from pipelineforge.utils.logger import get_logger

logger = get_logger(__name__)


def run_training_pipeline() -> None:
    logger.info("Starting training pipeline")

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    run_training_pipeline()
