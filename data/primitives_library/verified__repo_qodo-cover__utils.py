import argparse
import inspect
import logging
import os
import re
from typing import List

import yaml
from dynaconf import Dynaconf
from grep_ast import filename_to_lang

from cover_agent.lsp_logic.utils.utils import is_forbidden_directory
from cover_agent.settings.config_loader import get_settings
from cover_agent.settings.token_handling import TokenEncoder, clip_tokens
from cover_agent.version import __version__


def load_yaml(response_text: str, keys_fix_yaml: List[str] = []) -> dict:
    text = response_text.strip().removeprefix("```yaml").rstrip("`")
    try:
        return yaml.safe_load(text)
    except Exception as exc:
        logging.info(
            "Failed to parse AI prediction: %s. Attempting to fix YAML formatting.",
            exc,
        )
        result = try_fix_yaml(text, keys_fix_yaml=keys_fix_yaml)
        if not result:
            logging.info("Failed to parse AI prediction after fixing YAML formatting.")
        return result


def try_fix_yaml(response_text: str, keys_fix_yaml: List[str] = []) -> dict:
    lines = response_text.split("\n")

    adjusted_lines = list(lines)
    for index, line in enumerate(adjusted_lines):
        for key in keys_fix_yaml:
            if key in line and "|-" not in line:
                adjusted_lines[index] = line.replace(key, f"{key} |-\n        ")

    try:
        parsed = yaml.safe_load("\n".join(adjusted_lines))
        logging.info("Successfully parsed AI prediction after adding |-")
        return parsed
    except Exception:
        pass

    fenced = re.search(r"```(yaml)?[\s\S]*?```", "\n".join(adjusted_lines))
    if fenced is not None:
        try:
            parsed = yaml.safe_load(
                fenced.group().removeprefix("```yaml").rstrip("`")
            )
            logging.info(
                "Successfully parsed AI prediction after extracting yaml snippet"
            )
            return parsed
        except Exception:
            pass

    without_braces = (
        response_text.strip()
        .rstrip()
        .removeprefix("{")
        .removesuffix("}")
        .rstrip(":\n")
    )
    try:
        parsed = yaml.safe_load(without_braces)
        logging.info(
            "Successfully parsed AI prediction after removing curly brackets"
        )
        return parsed
    except Exception:
        pass

    parsed = {}
    for removed_count in range(1, len(lines)):
        candidate = "\n".join(lines[:-removed_count])
        try:
            parsed = yaml.safe_load(candidate)
            if "language" in parsed:
                logging.info(
                    "Successfully parsed AI prediction after removing %s lines",
                    removed_count,
                )
                return parsed
        except Exception:
            pass

    try:
        start = response_text.find("\nlanguage:")
        if start == -1:
            start = response_text.find("language:")
        last_code = response_text.rfind("test_code:")
        end = response_text.find("\n\n", last_code)
        if end == -1:
            end = len(response_text)
        candidate = response_text[start:end].strip()
        parsed = yaml.safe_load(candidate)
        logging.info(
            "Successfully parsed AI prediction when using the language: key as a starting point"
        )
        return parsed
    except Exception:
        pass

    return {}


def get_included_files(
    included_files: list, project_root: str = "", disable_tokens=False
) -> str:
    if not included_files:
        return ""

    contents = []
    relative_names = []

    for path in included_files:
        try:
            with open(path, "r") as source:
                contents.append(source.read())
            relative_names.append(
                os.path.relpath(path, project_root) if project_root else path
            )
        except IOError as exc:
            print(f"Error reading file {path}: {str(exc)}")

    output = ""
    for index, content in enumerate(contents):
        output += (
            f"file_path: `{relative_names[index]}`\n"
            f"content:\n"
            f"```\n{content}\n```\n\n\n"
        )

    output = output.strip()

    if (
        not disable_tokens
        and get_settings().get("include_files.limit_tokens", False)
    ):
        encoder = TokenEncoder.get_token_encoder()
        token_count = len(encoder.encode(output))
        token_limit = get_settings().get("include_files.max_tokens")
        if token_count > token_limit:
            print(
                f"Clipping included files content from {token_count} to "
                f"{token_limit} tokens"
            )
            output = clip_tokens(
                output,
                token_limit,
                num_input_tokens=token_count,
            )

    return output


def parse_args_full_repo(settings: Dynaconf) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Cover Agent v{__version__}")

    log_db_path = os.getenv("LOG_DB_PATH") or settings.get("log_db_path")

    parser.add_argument(
        "--max-test-files-allowed-to-analyze",
        type=int,
        default=settings.get("max_test_files_allowed_to_analyze"),
        help="The maximum number of test files to analyze. Default: %(default)s.",
    )
    parser.add_argument(
        "--look-for-oldest-unchanged-test-file",
        action="store_true",
        help="If set, Cover-Agent will look for the oldest unchanged test file to analyze.",
    )
    parser.add_argument(
        "--project-language",
        required=True,
        default=settings.get("project_language"),
        help="The programming language of the project ([python, javascript, typescript]). Default: %(default)s.",
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="Path to the root of the project.",
    )
    parser.add_argument(
        "--test-folder",
        help="Relative path to the relevant tests folder.",
    )
    parser.add_argument(
        "--test-file",
        help="Relative path to the specific test file we want to extend.",
    )
    parser.add_argument(
        "--run-each-test-separately",
        type=bool,
        default=True,
        help="Run each test separately. Default: True",
    )
    parser.add_argument(
        "--code-coverage-report-path",
        required=True,
        help="Path to the code coverage report file.",
    )
    parser.add_argument(
        "--test-command",
        required=True,
        help="The command to run tests and generate coverage report.",
    )
    parser.add_argument(
        "--test-command-dir",
        default=settings.get("test_command_dir"),
        help="The directory from which to run the test command.",
    )
    parser.add_argument(
        "--coverage-type",
        default=settings.get("coverage_type"),
        help="The type of coverage report. Default: %(default)s.",
    )
    parser.add_argument(
        "--report-file",
        default=settings.get("report_file"),
        help="Path to the markdown report file. Default: %(default)s.",
    )
    parser.add_argument(
        "--desired-coverage",
        type=int,
        default=settings.get("desired_coverage"),
        help="Desired code coverage percentage. Default: %(default)s.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=settings.get("max_iterations"),
        help="Maximum number of iterations. Default: %(default)s.",
    )
    parser.add_argument(
        "--additional-instructions",
        default=settings.get("additional_instructions"),
        help="Additional instructions to provide to the AI.",
    )
    parser.add_argument(
        "--included-files",
        nargs="*",
        default=settings.get("included_files"),
        help="Additional files to include in the prompt.",
    )
    parser.add_argument(
        "--ignore-files",
        nargs="*",
        default=settings.get("ignore_files"),
        help="Files or directories to ignore.",
    )
    parser.add_argument(
        "--log-db-path",
        default=log_db_path,
        help="Path to the log database.",
    )
    parser.add_argument(
        "--branch-name",
        default=settings.get("branch_name"),
        help="The branch name associated with the run.",
    )
    parser.add_argument(
        "--issue-number",
        default=settings.get("issue_number"),
        help="The issue number associated with the run.",
    )
    parser.add_argument(
        "--issue-url",
        default=settings.get("issue_url"),
        help="The issue URL associated with the run.",
    )
    parser.add_argument(
        "--pr-number",
        default=settings.get("pr_number"),
        help="The pull request number associated with the run.",
    )
    parser.add_argument(
        "--pr-url",
        default=settings.get("pr_url"),
        help="The pull request URL associated with the run.",
    )

    return parser.parse_args()