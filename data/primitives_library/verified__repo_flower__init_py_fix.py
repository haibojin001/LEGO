"""Fix provided directory and sub-directories for unsorted __all__ in __init__.py files."""

import os
import sys

import black

from devtool.init_py_check import get_all_var_list, get_init_dir_list_and_warnings


def fix_all_init_files(dir_list: list[str]) -> None:
    """Sort the __all__ variables that are in __init__.py files."""
    warning_list = []

    for init_dir in dir_list:
        init_file, all_list, all_lines = get_all_var_list(init_dir)

        if all_list:
            sorted_all_list = sorted(all_list)
            if all_list != sorted_all_list:
                warning_list.append("- " + str(init_dir))

                old_all_lines = "\n".join(all_lines)
                new_all_lines = (
                    old_all_lines.split("=", 1)[0]
                    + "= "
                    + str(sorted_all_list)[:-1]
                    + ",]"
                )

                new_content = init_file.read_text().replace(
                    old_all_lines, new_all_lines
                )
                init_file.write_text(new_content)

                black.format_file_in_place(
                    init_file,
                    fast=False,
                    mode=black.FileMode(),
                    write_back=black.WriteBack.YES,
                )

    if warning_list:
        print("'__all__' lists in the following '__init__.py' files have been sorted:")
        for warning in warning_list:
            print(warning)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "Please provide at least one directory path relative "
            "to your current working directory."
        )

    for relative_path in sys.argv[1:]:
        abs_path = os.path.abspath(os.path.join(os.getcwd(), relative_path))
        _, init_dirs = get_init_dir_list_and_warnings(abs_path)
        fix_all_init_files(init_dirs)