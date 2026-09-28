import hashlib
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from fuzzywuzzy import fuzz

from cover_agent.custom_logger import CustomLogger
from cover_agent.settings.config_loader import get_settings
from cover_agent.utils import truncate_hash


class RecordReplayManager:
    SETTINGS = get_settings().get("default")
    HASH_DISPLAY_LENGTH = SETTINGS.record_replay_hash_display_length

    def __init__(
        self,
        record_mode: bool,
        base_dir: str = SETTINGS.responses_folder,
        logger: Optional[CustomLogger] = None,
        generate_log_files: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.record_mode = record_mode
        self.files_hash = None
        self.logger = logger or CustomLogger.get_logger(
            __name__, generate_log_files=generate_log_files
        )

        self.logger.info(
            f"✨ RecordReplayManager initialized in "
            f"{'Run and Record' if record_mode else 'Run or Replay'} mode."
        )

    def has_response_file(self, source_file: str, test_file: str) -> bool:
        if not source_file or not test_file:
            raise FileNotFoundError(
                "Source file and test file paths must be set to check response file existence"
            )

        response_file = self._get_response_file_path(source_file, test_file)
        exists = response_file.exists()

        if exists:
            self.logger.debug(f"Found recorded LLM response file: {response_file}")
        else:
            self.logger.debug(f"Recorded LLM response file not found: {response_file}")

        return exists

    def load_recorded_response(
        self,
        source_file: str,
        test_file: str,
        prompt: dict[str, Any],
        caller_name: str = "unknown_caller",
        fuzzy_lookup: bool = True,
    ) -> tuple[str, int, int] | None:
        if self.record_mode:
            self.logger.debug("Skipping record loading in record mode.")
            return None

        response_file = self._get_response_file_path(source_file, test_file)
        if not response_file.exists():
            self.logger.debug(
                f"Recorded LLM response file not found: {response_file}."
            )
            return None

        try:
            with open(response_file, "r") as file:
                cached_data = yaml.safe_load(file)

            if caller_name not in cached_data:
                self.logger.info(f"No records found for caller {caller_name}.")
                return None

            caller = f"{caller_name}()"
            prompt_hash = truncate_hash(
                hashlib.sha256(str(prompt).encode()).hexdigest(),
                self.HASH_DISPLAY_LENGTH,
            )
            self.logger.info(
                f"Do a direct hash lookup for prompt hash {prompt_hash} "
                f"under caller {caller}..."
            )

            if prompt_hash in cached_data[caller_name]:
                self.logger.info(
                    f"Record hit for caller {caller_name}() with prompt hash "
                    f"{prompt_hash}."
                )
                entry = cached_data[caller_name][prompt_hash]
                return (
                    entry["response"],
                    entry["prompt_tokens"],
                    entry["completion_tokens"],
                )

            self.logger.info(
                f"No record entry found for prompt hash {prompt_hash} "
                f"under caller {caller}."
            )

            if fuzzy_lookup:
                self.logger.info(
                    f"Trying fuzzy lookup for prompt hash {prompt_hash} "
                    f"under caller {caller}..."
                )
                prompts = {
                    key: value["prompt"]["user"]
                    for key, value in cached_data[caller_name].items()
                }
                fuzzy_prompt_hash = self._find_closest_prompt_match(
                    prompt["user"], prompts
                )

                if fuzzy_prompt_hash:
                    self.logger.info(
                        f"Found fuzzy match for prompt hash {fuzzy_prompt_hash} "
                        f"under caller {caller}."
                    )
                    entry = cached_data[caller_name][fuzzy_prompt_hash]
                    return (
                        entry["response"],
                        entry["prompt_tokens"],
                        entry["completion_tokens"],
                    )

                self.logger.warning(
                    f"No record entry found for prompt hash {fuzzy_prompt_hash} "
                    f"under caller {caller} after fuzzy lookup."
                )
        except Exception as error:
            self.logger.error(
                f"Error loading recorded LLM response {error}", exc_info=True
            )

        return None

    def record_response(
        self,
        source_file: str,
        test_file: str,
        prompt: dict[str, Any],
        response: str,
        prompt_tokens: int,
        completion_tokens: int,
        caller_name: str = "unknown_caller",
    ) -> None:
        if not self.record_mode:
            self.logger.info("Skipping LLM response record in replay mode.")
            return

        response_file = self._get_response_file_path(source_file, test_file)
        self.logger.info(f"Recording LLM response to {response_file}...")

        metadata_key = "metadata"
        files_hash = truncate_hash(
            self._calculate_files_hash(source_file, test_file),
            self.HASH_DISPLAY_LENGTH,
        )
        cached_data: dict[str, Any] = {
            metadata_key: {"files_hash": files_hash}
        }

        if response_file.exists():
            try:
                with open(response_file, "r") as file:
                    loaded_data = yaml.safe_load(file)
                    if isinstance(loaded_data, dict):
                        cached_data.update(
                            {
                                key: value
                                for key, value in loaded_data.items()
                                if key != metadata_key
                            }
                        )
                        self.logger.debug(
                            f"Loaded existing LLM record with "
                            f"{len(cached_data) - 1} entries."
                        )
            except yaml.YAMLError:
                self.logger.warning(
                    f"Invalid YAML in {response_file}, starting fresh."
                )

        prompt_hash = truncate_hash(
            hashlib.sha256(str(prompt).encode()).hexdigest(),
            self.HASH_DISPLAY_LENGTH,
        )
        self.logger.info(
            f"🔴 Recording new LLM response for {caller_name}() "
            f"(prompt hash {prompt_hash})..."
        )

        if caller_name not in cached_data:
            cached_data[caller_name] = {}

        cached_data[caller_name][f"{prompt_hash}"] = {
            "prompt": prompt,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

        os.makedirs(self.base_dir, exist_ok=True)
        with open(response_file, "w") as file:
            yaml.safe_dump(cached_data, file, sort_keys=False)

        self.logger.info(
            f"Successfully recorded LLM response for {caller_name}() "
            f"with prompt hash {prompt_hash}."
        )

    def _calculate_files_hash(self, source_file: str, test_file: str) -> str:
        if self.files_hash is None:
            with open(source_file, "r") as file:
                source_content = file.read()

            with open(test_file, "r") as file:
                test_content = file.read()

            self.files_hash = hashlib.sha256(
                (source_content + test_content).encode()
            ).hexdigest()

        return self.files_hash

    def _get_response_file_path(self, source_file: str, test_file: str) -> Path:
        files_hash = truncate_hash(
            self._calculate_files_hash(source_file, test_file),
            self.HASH_DISPLAY_LENGTH,
        )
        return self.base_dir / f"responses_{files_hash}.yaml"

    def _find_closest_prompt_match(
        self, target_prompt: str, prompts: dict[str, str]
    ) -> Optional[str]:
        best_match = None
        best_score = 0

        for prompt_hash, prompt in prompts.items():
            score = fuzz.ratio(target_prompt, prompt)
            if score >= 90 and score > best_score:
                best_score = score
                best_match = prompt_hash

        return best_match