import logging

from cover_agent.settings.config_loader import get_settings


class CustomLogger:
    @classmethod
    def get_logger(
        cls,
        name,
        generate_log_files=True,
        file_level=logging.INFO,
        console_level=logging.INFO,
    ):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            if generate_log_files:
                log_file_path = get_settings().get("default").get(
                    "log_file_path", "run.log"
                )
                file_handler = logging.FileHandler(log_file_path, mode="w")
                file_handler.setLevel(file_level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(console_level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            logger.propagate = False

        return logger