import datetime
import json
import logging
import os
from typing import Optional

from diff_cover.diff_cover_tool import main as diff_cover_main
from wandb.sdk.data_types.trace_tree import Trace

from cover_agent.agent_completion_abc import AgentCompletionABC
from cover_agent.coverage_processor import CoverageProcessor
from cover_agent.custom_logger import CustomLogger
from cover_agent.file_preprocessor import FilePreprocessor
from cover_agent.runner import Runner
from cover_agent.settings.config_loader import get_settings
from cover_agent.settings.config_schema import CoverageType
from cover_agent.utils import load_yaml


class UnitTestValidator:
    def __init__(
        self,
        source_file_path: str,
        test_file_path: str,
        code_coverage_report_path: str,
        test_command: str,
        llm_model: str,
        max_run_time_sec: int,
        agent_completion: AgentCompletionABC,
        desired_coverage: int,
        comparison_branch: str,
        coverage_type: CoverageType,
        diff_coverage: bool,
        num_attempts: int,
        test_command_dir: str,
        additional_instructions: str,
        included_files: list,
        use_report_coverage_feature_flag: bool,
        project_root: str = "",
        logger: Optional[CustomLogger] = None,
        generate_log_files: bool = True,
    ):
        self.relevant_line_number_to_insert_imports_after = None
        self.relevant_line_number_to_insert_tests_after = None
        self.test_headers_indentation = None

        self.project_root = project_root
        self.source_file_path = source_file_path
        self.test_file_path = test_file_path
        self.code_coverage_report_path = code_coverage_report_path
        self.test_command = test_command
        self.test_command_dir = test_command_dir
        self.included_files = self.get_included_files(included_files)
        self.coverage_type = coverage_type
        self.desired_coverage = desired_coverage
        self.additional_instructions = additional_instructions
        self.language = self.get_code_language(source_file_path)
        self.use_report_coverage_feature_flag = use_report_coverage_feature_flag
        self.last_coverage_percentages = {}
        self.llm_model = llm_model
        self.diff_coverage = diff_coverage
        self.comparison_branch = comparison_branch
        self.num_attempts = num_attempts
        self.agent_completion = agent_completion
        self.max_run_time_sec = max_run_time_sec
        self.generate_log_files = generate_log_files

        self.logger = logger or CustomLogger.get_logger(
            __name__, generate_log_files=self.generate_log_files
        )

        if self.diff_coverage:
            self.coverage_type = "diff_cover_json"
            self.diff_coverage_report_name = "diff-cover-report.json"
            self.diff_cover_report_path = os.path.join(
                self.test_command_dir, self.diff_coverage_report_name
            )
            self.logger.info(
                f"Diff coverage enabled. Using coverage report: {self.diff_cover_report_path}"
            )
        else:
            self.diff_cover_report_path = ""

        self.preprocessor = FilePreprocessor(self.test_file_path)
        self.failed_test_runs = []
        self.total_input_token_count = 0
        self.total_output_token_count = 0
        self.testing_framework = "Unknown"
        self.code_coverage_report = ""

        with open(self.source_file_path, "r") as source_file:
            self.source_code = source_file.read()

        self.coverage_processor = CoverageProcessor(
            file_path=self.code_coverage_report_path,
            src_file_path=self.source_file_path,
            coverage_type=self.coverage_type,
            use_report_coverage_feature_flag=self.use_report_coverage_feature_flag,
            diff_coverage_report_path=self.diff_cover_report_path,
            generate_log_files=self.generate_log_files,
        )

    def get_coverage(self):
        self.run_coverage()
        return (
            self.failed_test_runs,
            self.language,
            self.testing_framework,
            self.code_coverage_report,
        )

    def get_code_language(self, source_file_path: str) -> str:
        language_extension_map_org = get_settings().language_extension_map_org
        extension_to_language = {}

        for language, extensions in language_extension_map_org.items():
            for extension in extensions:
                extension_to_language[extension] = language

        extension = "." + source_file_path.rsplit(".", 1)[-1]
        language = extension_to_language.get(extension, "unknown")
        return language.lower()

    def get_included_files(self, included_files: list):
        if not included_files:
            return []

        result = []
        for file_path in included_files:
            if isinstance(file_path, dict):
                result.append(file_path)
                continue

            try:
                with open(file_path, "r") as included_file:
                    result.append(
                        {
                            "file_path": file_path,
                            "content": included_file.read(),
                        }
                    )
            except (OSError, UnicodeDecodeError) as error:
                logging.warning("Unable to read included file '%s': %s", file_path, error)

        return result

    def _load_prompt(self):
        candidates = (
            "prompts/initial_test_suite_analysis.yaml",
            "prompts/initial_test_suite_analysis_prompt.yaml",
            "initial_test_suite_analysis.yaml",
        )

        for candidate in candidates:
            try:
                prompt = load_yaml(candidate)
                if prompt:
                    return prompt
            except Exception:
                continue
        return {}

    def _format_prompt(self, template, values):
        if isinstance(template, dict):
            template = (
                template.get("prompt")
                or template.get("system_prompt")
                or template.get("user_prompt")
                or ""
            )

        if not isinstance(template, str):
            return str(template)

        try:
            return template.format(**values)
        except Exception:
            return template

    def _extract_completion_text(self, response):
        if response is None:
            return ""

        if isinstance(response, str):
            return response

        if isinstance(response, bytes):
            return response.decode("utf-8", errors="replace")

        if isinstance(response, dict):
            for key in ("content", "response", "text", "message", "output"):
                if key in response:
                    value = response[key]
                    if isinstance(value, dict):
                        return self._extract_completion_text(value)
                    return str(value)

            choices = response.get("choices")
            if choices:
                return self._extract_completion_text(choices[0])

        if hasattr(response, "content"):
            return str(response.content)

        if hasattr(response, "message"):
            return self._extract_completion_text(response.message)

        if hasattr(response, "choices") and response.choices:
            return self._extract_completion_text(response.choices[0])

        return str(response)

    def _record_token_counts(self, response):
        usage = None
        if isinstance(response, dict):
            usage = response.get("usage")
        elif hasattr(response, "usage"):
            usage = response.usage

        if not usage:
            return

        if not isinstance(usage, dict):
            usage = {
                key: getattr(usage, key)
                for key in (
                    "prompt_tokens",
                    "input_tokens",
                    "completion_tokens",
                    "output_tokens",
                )
                if hasattr(usage, key)
            }

        self.total_input_token_count += int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        )
        self.total_output_token_count += int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )

    def _call_agent(self, prompt):
        methods = ("call", "chat_completion", "complete", "completion")
        last_error = None

        for method_name in methods:
            method = getattr(self.agent_completion, method_name, None)
            if not callable(method):
                continue

            attempts = (
                lambda: method(prompt=prompt, model=self.llm_model),
                lambda: method(prompt, self.llm_model),
                lambda: method(prompt),
            )

            for attempt in attempts:
                try:
                    response = attempt()
                    self._record_token_counts(response)
                    return response
                except TypeError as error:
                    last_error = error
                    continue

        if last_error:
            raise last_error
        raise AttributeError("The supplied agent completion object has no completion method")

    @staticmethod
    def _parse_json_response(response_text):
        if isinstance(response_text, dict):
            return response_text

        text = str(response_text).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except (TypeError, json.JSONDecodeError):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def initial_test_suite_analysis(self):
        prompt_config = self._load_prompt()
        values = {
            "source_file_path": self.source_file_path,
            "test_file_path": self.test_file_path,
            "source_code": self.source_code,
            "language": self.language,
            "project_root": self.project_root,
            "additional_instructions": self.additional_instructions,
        }
        prompt = self._format_prompt(prompt_config, values)

        if not prompt:
            prompt = (
                "Analyze the test suite below and return JSON containing "
                "test_headers_indentation, "
                "relevant_line_number_to_insert_tests_after, "
                "relevant_line_number_to_insert_imports_after, and "
                "testing_framework.\n\n"
                f"Test file: {self.test_file_path}\n"
            )
            try:
                with open(self.test_file_path, "r") as test_file:
                    prompt += test_file.read()
            except OSError:
                pass

        last_error = None
        for attempt in range(max(1, self.num_attempts)):
            try:
                response = self._call_agent(prompt)
                analysis = self._parse_json_response(self._extract_completion_text(response))

                self.test_headers_indentation = analysis.get(
                    "test_headers_indentation",
                    analysis.get("test_header_indentation"),
                )
                self.relevant_line_number_to_insert_tests_after = analysis.get(
                    "relevant_line_number_to_insert_tests_after",
                    analysis.get("line_number_to_insert_tests_after"),
                )
                self.relevant_line_number_to_insert_imports_after = analysis.get(
                    "relevant_line_number_to_insert_imports_after",
                    analysis.get("line_number_to_insert_imports_after"),
                )
                self.testing_framework = analysis.get(
                    "testing_framework",
                    analysis.get("test_framework", self.testing_framework),
                )

                if self.test_headers_indentation is None:
                    raise Exception("Unable to analyze test headers indentation")

                if self.relevant_line_number_to_insert_tests_after is None:
                    raise Exception(
                        "Unable to determine relevant line number to insert new tests"
                    )

                return analysis
            except Exception as error:
                last_error = error
                self.logger.warning(
                    "Initial test suite analysis attempt %s failed: %s",
                    attempt + 1,
                    error,
                )

        raise last_error or Exception("Unable to analyze the initial test suite")

    def _run_command(self):
        constructor_attempts = (
            lambda: Runner(
                self.test_command,
                self.test_command_dir,
                self.max_run_time_sec,
                generate_log_files=self.generate_log_files,
            ),
            lambda: Runner(
                command=self.test_command,
                cwd=self.test_command_dir,
                timeout=self.max_run_time_sec,
                generate_log_files=self.generate_log_files,
            ),
            lambda: Runner(
                self.test_command,
                self.test_command_dir,
                self.max_run_time_sec,
            ),
            lambda: Runner(self.test_command),
        )

        runner = None
        last_error = None
        for constructor in constructor_attempts:
            try:
                runner = constructor()
                break
            except TypeError as error:
                last_error = error

        if runner is None:
            raise last_error

        return runner.run()

    @staticmethod
    def _command_succeeded(result):
        if result is None:
            return True

        if isinstance(result, bool):
            return result

        if isinstance(result, int):
            return result == 0

        if isinstance(result, dict):
            for key in ("exit_code", "returncode", "return_code"):
                if key in result:
                    return result[key] == 0
            if "success" in result:
                return bool(result["success"])
            return True

        if isinstance(result, (tuple, list)) and result:
            first = result[0]
            if isinstance(first, int):
                return first == 0
            if isinstance(first, bool):
                return first

        return True

    def _run_diff_cover(self):
        if not self.diff_coverage:
            return

        args = [
            self.code_coverage_report_path,
            "--compare-branch",
            self.comparison_branch,
            "--json-report",
            self.diff_cover_report_path,
        ]

        previous_directory = os.getcwd()
        try:
            if self.test_command_dir:
                os.chdir(self.test_command_dir)
            try:
                diff_cover_main(args)
            except SystemExit as error:
                if error.code not in (0, None):
                    raise
        finally:
            os.chdir(previous_directory)

    def _process_coverage(self):
        processor = self.coverage_processor

        for method_name in (
            "process_coverage_report",
            "process",
            "parse_coverage_report",
            "get_coverage",
        ):
            method = getattr(processor, method_name, None)
            if not callable(method):
                continue
            try:
                result = method()
                if result is not None:
                    return result
            except TypeError:
                continue

        return None

    def _get_coverage_report(self, processed_result=None):
        processor = self.coverage_processor

        for attribute in (
            "coverage_report",
            "code_coverage_report",
            "report",
        ):
            value = getattr(processor, attribute, None)
            if value not in (None, ""):
                return value

        for method_name in (
            "get_coverage_report",
            "get_report",
            "generate_coverage_report",
        ):
            method = getattr(processor, method_name, None)
            if callable(method):
                value = method()
                if value is not None:
                    return value

        if isinstance(processed_result, str):
            return processed_result
        return ""

    def _get_coverage_percentage(self, processed_result=None):
        processor = self.coverage_processor

        for method_name in (
            "get_coverage_percentage",
            "get_total_coverage_percentage",
            "get_coverage",
        ):
            method = getattr(processor, method_name, None)
            if callable(method):
                try:
                    value = method()
                    if isinstance(value, (int, float)):
                        return value
                except TypeError:
                    continue

        for attribute in ("coverage_percentage", "total_coverage_percentage"):
            value = getattr(processor, attribute, None)
            if isinstance(value, (int, float)):
                return value

        if isinstance(processed_result, (int, float)):
            return processed_result
        return 0

    def run_coverage(self):
        try:
            result = self._run_command()
        except Exception as error:
            self.failed_test_runs.append(str(error))
            self.logger.warning("Test command failed: %s", error)
            return False

        if not self._command_succeeded(result):
            self.failed_test_runs.append(result)
            self.logger.warning("Test command returned a failure result: %s", result)
            return False

        try:
            self._run_diff_cover()
            processed_result = self._process_coverage()
            self.code_coverage_report = self._get_coverage_report(processed_result)
            return True
        except Exception as error:
            self.logger.warning("Unable to process coverage report: %s", error)
            self.code_coverage_report = ""
            return False

    def _insert_test(self, test_code):
        for method_name in (
            "add_test",
            "add_new_test",
            "insert_test",
            "write_test",
        ):
            method = getattr(self.preprocessor, method_name, None)
            if not callable(method):
                continue

            calls = (
                lambda: method(
                    test_code,
                    self.relevant_line_number_to_insert_tests_after,
                    self.test_headers_indentation,
                    self.relevant_line_number_to_insert_imports_after,
                ),
                lambda: method(
                    test_code,
                    self.relevant_line_number_to_insert_tests_after,
                    self.test_headers_indentation,
                ),
                lambda: method(test_code),
            )
            for call in calls:
                try:
                    return call()
                except TypeError:
                    continue

        with open(self.test_file_path, "a") as test_file:
            if os.path.getsize(self.test_file_path) > 0:
                test_file.write("\n")
            test_file.write(str(test_code))
            if not str(test_code).endswith("\n"):
                test_file.write("\n")
        return None

    def _remove_test(self, insertion_result=None):
        for method_name in (
            "remove_last_test",
            "remove_test",
            "revert_test",
            "rollback",
            "restore_original_file",
        ):
            method = getattr(self.preprocessor, method_name, None)
            if not callable(method):
                continue

            try:
                if insertion_result is not None:
                    return method(insertion_result)
                return method()
            except TypeError:
                try:
                    return method()
                except TypeError:
                    continue
        return None

    def validate_test(self, test, test_index=None):
        if isinstance(test, dict):
            test_code = (
                test.get("test_code")
                or test.get("test")
                or test.get("code")
                or test.get("content")
                or ""
            )
        else:
            test_code = getattr(test, "test_code", getattr(test, "code", test))

        if not isinstance(test_code, str):
            test_code = str(test_code)

        trace = None
        try:
            trace = Trace(
                name="validate_test",
                inputs={
                    "source_file_path": self.source_file_path,
                    "test_file_path": self.test_file_path,
                    "test_index": test_index,
                },
                start_time_ms=int(datetime.datetime.now().timestamp() * 1000),
            )
        except Exception:
            trace = None

        insertion_result = self._insert_test(test_code)
        previous_coverage = self.last_coverage_percentages.get(
            self.source_file_path, 0
        )

        successful = self.run_coverage()
        current_coverage = self._get_coverage_percentage()

        is_valid = bool(successful and current_coverage > previous_coverage)
        if is_valid:
            self.last_coverage_percentages[self.source_file_path] = current_coverage
        else:
            self._remove_test(insertion_result)

        if trace is not None:
            try:
                trace.add_output(
                    {
                        "valid": is_valid,
                        "coverage": current_coverage,
                        "previous_coverage": previous_coverage,
                    }
                )
            except Exception:
                pass

        return is_valid