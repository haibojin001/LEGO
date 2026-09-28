"""Check provided directory and sub-directories for missing __init__.py files.

Example:
    python -m devtool.init_py_check framework/py/flwr
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Tuple


def get_init_dir_list_and_warnings(absolute_path: str) -> Tuple[list[str], list[str]]:
    """Search given path and return list of dirs containing __init__.py files."""
    warnings: list[str] = []
    init_dirs: list[str] = []
    ignored_patterns = [
        "__pycache__$",
        ".pytest_cache.*$",
        "dist",
        "flwr.egg-info$",
        ".mypy_cache",
    ]

    for directory, _, filenames in os.walk(absolute_path):
        if any(re.search(pattern, directory) is not None for pattern in ignored_patterns):
            continue

        if any(filename == "__init__.py" for filename in filenames):
            init_dirs.append(directory)
        else:
            warnings.append("- " + directory)

    return warnings, init_dirs


def check_missing_init_files(absolute_path: str) -> list[str]:
    """Search absolute_path and look for missing __init__.py files."""
    warnings, init_dirs = get_init_dir_list_and_warnings(absolute_path)

    if len(warnings) > 0:
        print("Could not find '__init__.py' in the following directories:")
        for warning in warnings:
            print(warning)
        sys.exit(1)

    return init_dirs


def get_all_var_list(init_dir: str) -> Tuple[Path, list[str], list[str]]:
    """Get the __all__ list of a __init__.py file.

    The function returns the path of the '__init__.py' file of the given dir, as well as
    the list itself, and the list of lines corresponding to the list.
    """
    init_file = Path(init_dir) / "__init__.py"
    all_lines: list[str] = []
    all_list: list[str] = []
    capture = False

    for line in init_file.read_text().splitlines():
        stripped_line = line.strip()

        if stripped_line.startswith("__all__"):
            capture = True

        if capture:
            all_lines.append(line)
            if stripped_line.endswith("]"):
                capture = False
                break

    if all_lines:
        all_string = "".join(all_lines)
        all_list = ast.literal_eval(all_string.split("=", 1)[1].strip())

    return init_file, all_list, all_lines


def check_all_init_files(dir_list: list[str]) -> None:
    """Check if __all__ is in alphabetical order in __init__.py files."""
    warnings: list[str] = []

    for init_dir in dir_list:
        init_file, all_list, _ = get_all_var_list(init_dir)

        if all_list and all_list != sorted(all_list):
            warnings.append("- " + str(init_file))

    if len(warnings) > 0:
        print(
            "'__all__' lists in the following '__init__.py' files are "
            "incorrectly sorted:"
        )
        for warning in warnings:
            print(warning)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "Please provide at least one directory path relative "
            "to your current working directory."
        )

    for relative_path in sys.argv[1:]:
        absolute_path = os.path.abspath(os.path.join(os.getcwd(), relative_path))
        init_dirs = check_missing_init_files(absolute_path)
        check_all_init_files(init_dirs)