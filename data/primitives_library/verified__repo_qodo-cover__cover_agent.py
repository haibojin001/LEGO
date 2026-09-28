import datetime
import inspect
import os
import shutil
import sys
from typing import Optional

import wandb

from cover_agent.agent_completion_abc import AgentCompletionABC
from cover_agent.ai_caller import AICaller
from cover_agent.ai_caller_replay import AICallerReplay
from cover_agent.custom_logger import CustomLogger
from cover_agent.default_agent_completion import DefaultAgentCompletion
from cover_agent.record_replay_manager import RecordReplayManager
from cover_agent.settings.config_schema import CoverAgentConfig
from cover_agent.unit_test_db import UnitTestDB
from cover_agent.unit_test_generator import UnitTestGenerator
from cover_agent.unit_test_validator import UnitTestValidator


class CoverAgent:
    """
    Coordinates coverage-driven test generation, validation, file handling, and
    optional experiment tracking.
    """

    def __init__(
        self,
        config: CoverAgentConfig,
        agent_completion: AgentCompletionABC = None,
        logger: Optional[CustomLogger] = None,
    ):
        self.config = config
        self.generate_log_files = not config.suppress_log_files

        self.logger = logger or CustomLogger.get_logger(
            __name__, generate_log_files=self.generate_log_files
        )
        if config.suppress_log_files:
            self.logger.info("Suppressed all generated log files.")

        self._validate_paths()
        self._duplicate_test_file()

        if agent_completion:
            self.agent_completion = agent_completion
        else:
            self.ai_caller = self._initialize_ai_caller()
            self.agent_completion = DefaultAgentCompletion(
                caller=self.ai_caller,
                generate_log_files=self.generate_log_files,
            )

        test_command = self.config.test_command
        new_command_line = None

        if (
            hasattr(self.config, "run_each_test_separately")
            and self.config.run_each_test_separately
        ):
            test_file_relative_path = os.path.relpath(
                self.config.test_file_output_path,
                self.config.project_root,
            )

            if "pytest" in test_command:
                try:
                    pytest_position = test_command.index("pytest")
                    separator_position = test_command[pytest_position:].index("--")
                    new_command_line = (
                        f"{test_command[:pytest_position]}pytest "
                        f"{test_file_relative_path} "
                        f"{test_command[pytest_position + separator_position:]}"
                    )
                except ValueError:
                    self.logger.error(
                        "Failed to adapt test command for running a single test: "
                        f"{test_command}"
                    )
            else:
                new_command_line, _, _, _ = (
                    self.agent_completion.adapt_test_command_for_a_single_test_via_ai(
                        test_file_relative_path=test_file_relative_path,
                        test_command=test_command,
                        project_root_dir=self.config.test_command_dir,
                    )
                )

        if new_command_line:
            self.config.test_command_original = test_command
            self.config.test_command = new_command_line
            self.logger.info(
                f"Converting test command: `{test_command}`\n"
                f" to run only a single test: `{new_command_line}`"
            )

        self.test_gen = UnitTestGenerator(
            source_file_path=self.config.source_file_path,
            test_file_path=self.config.test_file_output_path,
            project_root=self.config.project_root,
            code_coverage_report_path=self.config.code_coverage_report_path,
            test_command=self.config.test_command,
            test_command_dir=self.config.test_command_dir,
            included_files=self.config.included_files,
            coverage_type=self.config.coverage_type,
            additional_instructions=self.config.additional_instructions,
            llm_model=self.config.model,
            use_report_coverage_feature_flag=self.config.use_report_coverage_feature_flag,
            agent_completion=self.agent_completion,
            generate_log_files=self.generate_log_files,
        )

        self.test_validator = UnitTestValidator(
            source_file_path=self.config.source_file_path,
            test_file_path=self.config.test_file_output_path,
            project_root=self.config.project_root,
            code_coverage_report_path=self.config.code_coverage_report_path,
            test_command=self.config.test_command,
            test_command_dir=self.config.test_command_dir,
            included_files=self.config.included_files,
            coverage_type=self.config.coverage_type,
            desired_coverage=self.config.desired_coverage,
            additional_instructions=self.config.additional_instructions,
            llm_model=self.config.model,
            use_report_coverage_feature_flag=self.config.use_report_coverage_feature_flag,
            diff_coverage=self.config.diff_coverage,
            comparison_branch=self.config.branch,
            num_attempts=self.config.run_tests_multiple_times,
            agent_completion=self.agent_completion,
            max_run_time_sec=self.config.max_run_time_sec,
            generate_log_files=self.generate_log_files,
        )

    def _initialize_ai_caller(self):
        ai_caller_params = {
            "model": self.config.model,
            "api_base": self.config.api_base,
            "max_tokens": 8192,
            "source_file": self.config.source_file_path,
            "test_file": self.config.test_file_path,
            "record_mode": True,
            "generate_log_files": self.generate_log_files,
        }

        if self.config.record_mode:
            self.logger.info("Initializing AICaller in Record mode...")
            return AICaller(**ai_caller_params)

        try:
            replay_manager = RecordReplayManager(
                record_mode=False,
                generate_log_files=self.generate_log_files,
            )
            replay_manager.source_file = self.config.source_file_path
            replay_manager.test_file = self.config.test_file_path

            if replay_manager.has_response_file(
                source_file=self.config.source_file_path,
                test_file=self.config.test_file_path,
            ):
                self.logger.info(
                    "Initializing AICallerReplay (found recorded responses)..."
                )
                return AICallerReplay(
                    source_file=self.config.source_file_path,
                    test_file=self.config.test_file_path,
                    generate_log_files=self.generate_log_files,
                )
        except Exception as error:
            self.logger.debug(f"Failed to initialize replay mode: {error}")

        self.logger.info(
            "Initializing AICaller without recording (no recorded responses found)"
        )
        ai_caller_params["record_mode"] = False
        return AICaller(**ai_caller_params)

    def _validate_paths(self):
        source_file_path = getattr(self.config, "source_file_path", None)
        project_root = getattr(self.config, "project_root", None)
        test_command_dir = getattr(self.config, "test_command_dir", None)
        test_file_output_path = getattr(self.config, "test_file_output_path", None)

        if not source_file_path:
            raise ValueError("A source file path must be provided.")

        if not os.path.isfile(source_file_path):
            raise FileNotFoundError(f"Source file not found: {source_file_path}")

        if not project_root:
            raise ValueError("A project root path must be provided.")

        if not os.path.isdir(project_root):
            raise FileNotFoundError(f"Project root directory not found: {project_root}")

        if test_command_dir and not os.path.isdir(test_command_dir):
            raise FileNotFoundError(
                f"Test command directory not found: {test_command_dir}"
            )

        if test_file_output_path:
            output_directory = os.path.dirname(
                os.path.abspath(test_file_output_path)
            )
            if output_directory and not os.path.exists(output_directory):
                os.makedirs(output_directory, exist_ok=True)

        self.test_db = self._initialize_test_db()

    def _initialize_test_db(self):
        attempts = (
            (),
            (getattr(self.config, "project_root", None),),
            (getattr(self.config, "test_file_output_path", None),),
        )

        last_error = None
        for args in attempts:
            try:
                if args and args[0] is None:
                    continue
                return UnitTestDB(*args)
            except TypeError as error:
                last_error = error

        if last_error is not None:
            raise last_error
        return UnitTestDB()

    def _duplicate_test_file(self):
        source_test_file = getattr(self.config, "test_file_path", None)
        output_test_file = getattr(self.config, "test_file_output_path", None)

        if not source_test_file or not output_test_file:
            return

        source_absolute = os.path.abspath(source_test_file)
        output_absolute = os.path.abspath(output_test_file)

        if source_absolute == output_absolute:
            return

        if not os.path.exists(source_absolute):
            return

        output_directory = os.path.dirname(output_absolute)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        shutil.copyfile(source_absolute, output_absolute)

    def _call_with_supported_arguments(self, callable_object, values):
        try:
            signature = inspect.signature(callable_object)
        except (TypeError, ValueError):
            return callable_object()

        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        if accepts_var_kwargs:
            return callable_object(**values)

        kwargs = {
            name: value
            for name, value in values.items()
            if name in signature.parameters
        }

        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]

        missing = [
            parameter
            for parameter in required
            if parameter.name not in kwargs
            and parameter.kind != inspect.Parameter.POSITIONAL_ONLY
        ]

        if missing:
            return callable_object()

        return callable_object(**kwargs)

    @staticmethod
    def _extract_coverage(value):
        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, dict):
            for key in (
                "coverage",
                "current_coverage",
                "total_coverage",
                "coverage_percentage",
            ):
                if key in value:
                    extracted = CoverAgent._extract_coverage(value[key])
                    if extracted is not None:
                        return extracted
            return None

        if isinstance(value, (tuple, list)):
            for item in value:
                extracted = CoverAgent._extract_coverage(item)
                if extracted is not None:
                    return extracted

        for attribute in (
            "coverage",
            "current_coverage",
            "total_coverage",
            "coverage_percentage",
        ):
            if hasattr(value, attribute):
                extracted = CoverAgent._extract_coverage(getattr(value, attribute))
                if extracted is not None:
                    return extracted

        return None

    def _get_coverage(self):
        for method_name in (
            "get_coverage",
            "get_current_coverage",
            "validate_test_coverage",
            "validate_coverage",
        ):
            method = getattr(self.test_validator, method_name, None)
            if not callable(method):
                continue

            result = self._call_with_supported_arguments(
                method,
                {
                    "source_file_path": self.config.source_file_path,
                    "test_file_path": self.config.test_file_output_path,
                    "test_command": self.config.test_command,
                },
            )
            coverage = self._extract_coverage(result)
            if coverage is not None:
                return coverage

        return None

    def _log_wandb(self, values):
        if not getattr(self.config, "use_wandb", False):
            return

        try:
            wandb.log(values)
        except Exception as error:
            self.logger.debug(f"Failed to log metrics to Weights & Biases: {error}")

    def _initialize_wandb(self):
        if not getattr(self.config, "use_wandb", False):
            return None

        try:
            config_data = (
                self.config.model_dump()
                if hasattr(self.config, "model_dump")
                else vars(self.config)
            )
            return wandb.init(
                project="qodo-cover",
                config=config_data,
            )
        except Exception as error:
            self.logger.debug(
                f"Failed to initialize Weights & Biases integration: {error}"
            )
            return None

    def run(self):
        self._initialize_wandb()

        desired_coverage = getattr(self.config, "desired_coverage", None)
        max_iterations = getattr(self.config, "max_iterations", 1) or 1
        current_coverage = self._get_coverage()

        if current_coverage is not None:
            self.logger.info(f"Current coverage: {current_coverage}%")
            self._log_wandb(
                {
                    "coverage": current_coverage,
                    "iteration": 0,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

        if (
            desired_coverage is not None
            and current_coverage is not None
            and current_coverage >= desired_coverage
        ):
            self.logger.info(
                f"Desired coverage of {desired_coverage}% has already been achieved."
            )
            return current_coverage

        for iteration in range(1, max_iterations + 1):
            self.logger.info(f"Starting iteration {iteration}/{max_iterations}")

            generator = getattr(self.test_gen, "generate_tests", None)
            if not callable(generator):
                raise AttributeError("UnitTestGenerator does not provide generate_tests.")

            generation_result = self._call_with_supported_arguments(
                generator,
                {
                    "current_coverage": current_coverage,
                    "desired_coverage": desired_coverage,
                    "iteration": iteration,
                    "test_file_path": self.config.test_file_output_path,
                    "source_file_path": self.config.source_file_path,
                },
            )

            validation_result = None
            for method_name in (
                "validate_and_fix_tests",
                "validate_tests",
                "validate_test",
            ):
                validator = getattr(self.test_validator, method_name, None)
                if not callable(validator):
                    continue

                validation_result = self._call_with_supported_arguments(
                    validator,
                    {
                        "generated_tests": generation_result,
                        "test_content": generation_result,
                        "tests": generation_result,
                        "iteration": iteration,
                        "test_file_path": self.config.test_file_output_path,
                        "source_file_path": self.config.source_file_path,
                        "current_coverage": current_coverage,
                        "desired_coverage": desired_coverage,
                    },
                )
                break

            updated_coverage = self._extract_coverage(validation_result)
            if updated_coverage is None:
                updated_coverage = self._get_coverage()

            if updated_coverage is not None:
                current_coverage = updated_coverage
                self.logger.info(
                    f"Coverage after iteration {iteration}: {current_coverage}%"
                )
                self._log_wandb(
                    {
                        "coverage": current_coverage,
                        "iteration": iteration,
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                )

            if (
                desired_coverage is not None
                and current_coverage is not None
                and current_coverage >= desired_coverage
            ):
                self.logger.info(
                    f"Desired coverage of {desired_coverage}% achieved after "
                    f"{iteration} iteration(s)."
                )
                break

        return current_coverage