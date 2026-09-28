from typing import Optional, Tuple

from jinja2 import Environment, StrictUndefined

from cover_agent.agent_completion_abc import AgentCompletionABC
from cover_agent.ai_caller import AICaller
from cover_agent.custom_logger import CustomLogger
from cover_agent.settings.config_loader import get_settings
from cover_agent.utils import load_yaml


class DefaultAgentCompletion(AgentCompletionABC):
    """
    Default completion implementation that renders configured prompt templates
    and submits them through an AICaller.
    """

    def __init__(
        self,
        caller: AICaller,
        logger: Optional[CustomLogger] = None,
        generate_log_files: bool = True,
    ):
        self.caller = caller
        self.logger = logger or CustomLogger.get_logger(
            __name__, generate_log_files=generate_log_files
        )

    def _build_prompt(self, file: str, **kwargs) -> dict:
        environment = Environment(undefined=StrictUndefined)

        try:
            settings = get_settings().get(file)
            if (
                not settings
                or not hasattr(settings, "system")
                or not hasattr(settings, "user")
            ):
                message = (
                    f"Could not find valid system/user prompt settings for: {file}"
                )
                self.logger.error(message)
                raise ValueError(message)

            system_prompt = environment.from_string(settings.system).render(**kwargs)
            user_prompt = environment.from_string(settings.user).render(**kwargs)
        except ValueError:
            raise
        except Exception as error:
            message = f"Error rendering prompt for '{file}': {error}"
            self.logger.error(message)
            raise RuntimeError(message)

        return {"system": system_prompt, "user": user_prompt}

    def generate_tests(
        self,
        source_file_name: str,
        max_tests: int,
        source_file_numbered: str,
        code_coverage_report: str,
        language: str,
        test_file: str,
        test_file_name: str,
        testing_framework: str,
        additional_instructions_text: str = None,
        additional_includes_section: str = None,
        failed_tests_section: str = None,
    ) -> Tuple[str, int, int, str]:
        prompt = self._build_prompt(
            file="test_generation_prompt",
            source_file_name=source_file_name,
            max_tests=max_tests,
            source_file_numbered=source_file_numbered,
            code_coverage_report=code_coverage_report,
            language=language,
            test_file=test_file,
            test_file_name=test_file_name,
            testing_framework=testing_framework,
            additional_instructions_text=additional_instructions_text,
            additional_includes_section=additional_includes_section,
            failed_tests_section=failed_tests_section,
        )
        response, prompt_tokens, completion_tokens = self.caller.call_model(prompt)
        return response, prompt_tokens, completion_tokens, prompt["user"]

    def analyze_test_failure(
        self,
        source_file_name: str,
        source_file: str,
        processed_test_file: str,
        stdout: str,
        stderr: str,
        test_file_name: str,
    ) -> Tuple[str, int, int, str]:
        prompt = self._build_prompt(
            file="analyze_test_run_failure",
            source_file_name=source_file_name,
            source_file=source_file,
            processed_test_file=processed_test_file,
            stdout=stdout,
            stderr=stderr,
            test_file_name=test_file_name,
        )
        response, prompt_tokens, completion_tokens = self.caller.call_model(prompt)
        return response, prompt_tokens, completion_tokens, prompt["user"]

    def analyze_test_insert_line(
        self,
        language: str,
        test_file_numbered: str,
        test_file_name: str,
        additional_instructions_text: str = None,
    ) -> Tuple[str, int, int, str]:
        prompt = self._build_prompt(
            file="analyze_suite_test_insert_line",
            language=language,
            test_file_numbered=test_file_numbered,
            test_file_name=test_file_name,
            additional_instructions_text=additional_instructions_text,
        )
        response, prompt_tokens, completion_tokens = self.caller.call_model(prompt)
        return response, prompt_tokens, completion_tokens, prompt["user"]