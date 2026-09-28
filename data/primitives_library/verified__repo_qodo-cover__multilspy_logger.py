import inspect
import logging
from datetime import datetime

from pydantic import BaseModel


class LogLine(BaseModel):
    time: str
    level: str
    caller_file: str
    caller_name: str
    caller_line: int
    message: str


class MultilspyLogger:
    def __init__(self) -> None:
        self.logger = logging.getLogger("multilspy")
        self.logger.setLevel(logging.INFO)

    def log(
        self, debug_message: str, level: int, sanitized_error_message: str = ""
    ) -> None:
        debug_message = debug_message.replace("'", '"').replace("\n", " ")
        sanitized_error_message = sanitized_error_message.replace("'", '"').replace(
            "\n", " "
        )

        current_frame = inspect.currentframe()
        outer_frames = inspect.getouterframes(current_frame, 2)
        caller = outer_frames[1]

        record = LogLine(
            time=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            level=logging.getLevelName(level),
            caller_file=caller[1].split("/")[-1],
            caller_name=caller[3],
            caller_line=caller[2],
            message=debug_message,
        )

        self.logger.log(level=level, msg=record.json())