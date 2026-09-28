import argparse
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from cover_agent.settings.config_loader import get_settings


class CoverageType(Enum):
    LCOV = "lcov"
    COBERTURA = "cobertura"
    JACOCO = "jacoco"


@dataclass
class CoverAgentConfig:
    source_file_path: str
    test_file_path: str
    project_root: str
    test_file_output_path: str
    code_coverage_report_path: str
    test_command: str
    test_command_dir: str
    included_files: list[str] | None
    coverage_type: CoverageType
    report_filepath: str
    desired_coverage: int
    max_iterations: int
    max_run_time_sec: int
    additional_instructions: str
    model: str
    api_base: str
    strict_coverage: bool
    run_tests_multiple_times: int
    log_db_path: str
    branch: str
    use_report_coverage_feature_flag: bool
    diff_coverage: bool
    run_each_test_separately: bool
    record_mode: bool
    suppress_log_files: bool
    max_test_files_allowed_to_analyze: int
    look_for_oldest_unchanged_test_file: bool
    project_language: str
    test_command_original: Optional[str] = None

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "CoverAgentConfig":
        return cls(
            source_file_path=args.source_file_path,
            test_file_path=args.test_file_path,
            project_root=args.project_root,
            test_file_output_path=args.test_file_output_path,
            code_coverage_report_path=args.code_coverage_report_path,
            test_command=args.test_command,
            test_command_dir=args.test_command_dir,
            included_files=args.included_files,
            coverage_type=args.coverage_type,
            report_filepath=args.report_filepath,
            desired_coverage=args.desired_coverage,
            max_iterations=args.max_iterations,
            max_run_time_sec=args.max_run_time_sec,
            additional_instructions=args.additional_instructions,
            model=args.model,
            api_base=args.api_base,
            strict_coverage=args.strict_coverage,
            run_tests_multiple_times=args.run_tests_multiple_times,
            log_db_path=os.getenv("LOG_DB_PATH") or args.log_db_path,
            branch=args.branch,
            use_report_coverage_feature_flag=args.use_report_coverage_feature_flag,
            diff_coverage=args.diff_coverage,
            run_each_test_separately=args.run_each_test_separately,
            record_mode=args.record_mode,
            suppress_log_files=args.suppress_log_files,
            max_test_files_allowed_to_analyze=args.max_test_files_allowed_to_analyze,
            look_for_oldest_unchanged_test_file=args.look_for_oldest_unchanged_test_file,
            project_language=args.project_language,
        )

    @classmethod
    def from_cli_args_with_defaults(
        cls, args: argparse.Namespace
    ) -> "CoverAgentConfig":
        default_config = get_settings().get("default")
        cli_values = vars(args)

        defaults = {
            "source_file_path": default_config.get("source_file_path"),
            "test_file_path": default_config.get("test_file_path"),
            "project_root": default_config.get("project_root"),
            "test_file_output_path": default_config.get("test_file_output_path"),
            "code_coverage_report_path": default_config.get(
                "code_coverage_report_path"
            ),
            "test_command": default_config.get("test_command"),
            "test_command_dir": default_config.get("test_command_dir"),
            "included_files": default_config.get("included_files"),
            "coverage_type": default_config.get("coverage_type"),
            "report_filepath": default_config.get("report_filepath"),
            "desired_coverage": default_config.get("desired_coverage"),
            "max_iterations": default_config.get("max_iterations"),
            "max_run_time_sec": default_config.get("max_run_time_sec"),
            "additional_instructions": default_config.get(
                "additional_instructions"
            ),
            "model": default_config.get("model"),
            "api_base": default_config.get("api_base"),
            "strict_coverage": default_config.get("strict_coverage"),
            "run_tests_multiple_times": default_config.get(
                "run_tests_multiple_times"
            ),
            "log_db_path": default_config.get("log_db_path"),
            "branch": default_config.get("branch"),
            "use_report_coverage_feature_flag": default_config.get(
                "use_report_coverage_feature_flag"
            ),
            "diff_coverage": default_config.get("diff_coverage"),
            "run_each_test_separately": default_config.get(
                "run_each_test_separately"
            ),
            "record_mode": default_config.get("record_mode"),
            "suppress_log_files": default_config.get("suppress_log_files"),
            "max_test_files_allowed_to_analyze": default_config.get(
                "max_test_files_allowed_to_analyze"
            ),
            "look_for_oldest_unchanged_test_file": default_config.get(
                "look_for_oldest_unchanged_test_file"
            ),
            "project_language": default_config.get("project_language"),
        }

        merged = defaults.copy()
        for key, value in cli_values.items():
            if value is not None and hasattr(args, key):
                merged[key] = value

        return cls(**merged)