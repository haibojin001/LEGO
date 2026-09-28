"""Check whether Python source files contain valid copyright notices."""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from devtool.init_py_check import get_init_dir_list_and_warnings

COPYRIGHT_FORMAT = """# Copyright {} Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================="""

COPYRIGHT_PATTERN = re.compile(r"# Copyright (\d{4}) Flower Labs GmbH")


def _get_file_creation_year(filepath: str) -> int:
    result = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%ai", "--", filepath],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )

    if not result.stdout:
        return datetime.now().year

    first_commit_date = result.stdout.splitlines()[-1]
    return int(first_commit_date.split("-")[0])


def _check_copyright(dir_list: list[str]) -> None:
    warning_list: list[str] = []

    for valid_dir in dir_list:
        if "proto" in valid_dir:
            continue

        for py_file in Path(valid_dir).glob("*.py"):
            creation_year = _get_file_creation_year(str(py_file.absolute()))
            match = COPYRIGHT_PATTERN.search(py_file.read_text())
            copyright_year = int(match.group(1)) if match else None

            if copyright_year not in (creation_year, creation_year - 1):
                warning_list.append("- " + str(py_file))

    if warning_list:
        print("Missing or incorrect copyright notice in the following files:")
        for warning in warning_list:
            print(warning)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "Please provide at least one directory path relative "
            "to your current working directory."
        )

    for relative_path in sys.argv[1:]:
        abs_path = os.path.abspath(os.path.join(os.getcwd(), relative_path))
        _, init_dirs = get_init_dir_list_and_warnings(abs_path)
        _check_copyright(init_dirs)