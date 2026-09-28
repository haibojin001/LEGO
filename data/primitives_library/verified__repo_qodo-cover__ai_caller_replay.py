import time
from typing import Optional

from cover_agent.custom_logger import CustomLogger
from cover_agent.record_replay_manager import RecordReplayManager
from cover_agent.utils import get_original_caller


class AICallerReplay:
    """Replays previously recorded responses from an LLM."""

    def __init__(
        self,
        source_file: str,
        test_file: str,
        record_replay_manager: Optional[RecordReplayManager] = None,
        logger: Optional[CustomLogger] = None,
        generate_log_files: bool = True,
    ):
        self.source_file = source_file
        self.test_file = test_file
        self.record_replay_manager = (
            record_replay_manager
            if record_replay_manager is not None
            else RecordReplayManager(record_mode=False)
        )
        self.logger = (
            logger
            if logger is not None
            else CustomLogger.get_logger(
                __name__,
                generate_log_files=generate_log_files,
            )
        )

    def call_model(self, prompt: dict, stream=True) -> tuple[str, int, int]:
        caller_name = get_original_caller()
        response = self.record_replay_manager.load_recorded_response(
            self.source_file,
            self.test_file,
            prompt,
            caller_name=caller_name,
        )

        if not response:
            message = (
                "No recorded response found for prompt hash in replay mode. "
                f"Source file: {self.source_file}, Test file: {self.test_file}."
            )
            self.logger.error(message)
            raise KeyError(message)

        content, prompt_tokens, completion_tokens = response
        self.logger.info("▶️  Replaying results from recorded LLM response...")

        if stream:
            self.stream_recorded_llm_response(content)
        else:
            print(content)

        return content, prompt_tokens, completion_tokens

    @staticmethod
    def stream_recorded_llm_response(content: str) -> None:
        for line in content.splitlines():
            if line == "":
                print()
                continue

            leading_spaces = len(line) - len(line.lstrip())
            print(" " * leading_spaces, end="")

            for token in line.lstrip().split():
                print(token, end=" ", flush=True)
                time.sleep(0.01)

            print()