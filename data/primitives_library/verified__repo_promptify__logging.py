import logging


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the promptify logger."""
    promptify_logger = logging.getLogger("promptify")
    if not promptify_logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        promptify_logger.addHandler(stream_handler)
    promptify_logger.setLevel(level)
    return promptify_logger


logger = logging.getLogger("promptify")