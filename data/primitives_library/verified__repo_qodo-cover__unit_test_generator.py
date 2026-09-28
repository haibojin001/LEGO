import json
import os

from typing import Optional

from cover_agent.agent_completion_abc import AgentCompletionABC
from cover_agent.custom_logger import CustomLogger
from cover_agent.file_preprocessor import FilePreprocessor
from cover_agent.settings.config_loader import get_settings
from cover_agent.utils import load_yaml


class UnitTestGenerator:
    def __init__(
        self,
        source_file_path: str,
        test_file_path: str,
        code_coverage_report_path: str,
        test_command: str,
        llm_model: str,
        agent_completion: AgentCompletionABC,
        test_command_dir: str = os.getcwd(),
        included_files: list = None,
        coverage_type="cobertura",
        additional_instructions: str = "",
        use_report_coverage_feature_flag: bool = False,
        project_root: str = "",
        logger: Optional[CustomLogger] = None,
        generate_log_files: bool = True,
    ):
        self.project_root = project_root
        self.source_file_path = source_file_path
        self.test_file_path = test_file_path
        self.code_coverage_report_path = code_coverage_report_path
        self.test_command = test_command
        self.test_command_dir = test_command_dir
        self.included_files = included_files
        self.coverage_type = coverage_type
        self.additional_instructions = additional_instructions
        self.language = self.get_code_language(source_file_path)
        self.use_report_coverage_feature_flag = use_report_coverage_feature_flag
        self.last_coverage_percentages = {}
        self.llm_model = llm_model
        self.agent_completion = agent_completion
        self.generate_log_files = generate_log_files

        self.logger = logger or CustomLogger.get_logger(
            __name__, generate_log_files=self.generate_log_files
        )

        self.preprocessor = FilePreprocessor(self.test_file_path)
        self.total_input_token_count = 0
        self.total_output_token_count = 0
        self.testing_framework = "Unknown"
        self.code_coverage_report = ""

        with open(self.source_file_path, "r") as source_file:
            self.source_code = source_file.read()

        with open(self.test_file_path, "r") as test_file:
            self.test_code = test_file.read()

    def get_code_language(self, source_file_path):
        language_extension_map_org = get_settings().language_extension_map_org
        extension_to_language = {}

        for language, extensions in language_extension_map_org.items():
            for extension in extensions:
                extension_to_language[extension] = language

        extension_s = "." + source_file_path.rsplit(".")[-1]
        language_name = "unknown"

        if extension_s and extension_s in extension_to_language:
            language_name = extension_to_language[extension_s]

        return language_name.lower()

    def check_for_failed_test_runs(self, failed_test_runs):
        if not failed_test_runs:
            failed_test_runs_value = ""
        else:
            failed_test_runs_value = ""
            try:
                for failed_test in failed_test_runs:
                    failed_test_dict = failed_test.get("code", {})
                    if not failed_test_dict:
                        continue

                    code = json.dumps(failed_test_dict)
                    error_message = failed_test.get("error_message", None)
                    failed_test_runs_value += f"Failed Test:\n```\n{code}\n```\n"

                    if error_message:
                        failed_test_runs_value += (
                            f"Test execution error analysis:\n{error_message}\n\n\n"
                        )
                    else:
                        failed_test_runs_value += "\n\n"
            except Exception as error:
                self.logger.error(f"Error processing failed test runs: {error}")
                failed_test_runs_value = ""

        return failed_test_runs_value

    def generate_tests(self, failed_test_runs, language, testing_framework, code_coverage_report):
        failed_test_runs_value = self.check_for_failed_test_runs(failed_test_runs)

        try:
            max_tests_per_run = get_settings().get("default").get(
                "max_tests_per_run", 4
            )

            (
                response,
                prompt_token_count,
                response_token_count,
                self.prompt,
            ) = self.agent_completion.generate_tests(
                source_file_name=os.path.relpath(
                    self.source_file_path, self.project_root
                ),
                max_tests=max_tests_per_run,
                source_file_numbered="\n".join(
                    f"{index + 1} {line}"
                    for index, line in enumerate(self.source_code.split("\n"))
                ),
                code_coverage_report=code_coverage_report,
                test_file_name=os.path.relpath(self.test_file_path, self.project_root),
                test_file=self.test_code,
                language=language,
                testing_framework=testing_framework,
                additional_instructions=self.additional_instructions,
                included_files=self.included_files,
                failed_test_runs=failed_test_runs_value,
            )

            self.total_input_token_count += prompt_token_count
            self.total_output_token_count += response_token_count
            self.testing_framework = testing_framework
            self.code_coverage_report = code_coverage_report

            return load_yaml(response)
        except Exception as error:
            self.logger.error(f"Error generating tests: {error}")
            return {}