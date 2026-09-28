import argparse
import re
from pathlib import Path
from typing import Callable

Replacement = str | Callable[[re.Match[str]], str]
ReplacementsByPattern = dict[str, list[tuple[str, Replacement]]]


def _compute_old_version(new_version: str) -> str:
    """Return the preceding minor version for a major.minor version string."""
    major_text, minor_text = new_version.split(".")
    major = int(major_text)
    minor = int(minor_text)

    if minor == 0:
        raise ValueError("Minor version is 0, can't infer previous version.")

    return f"{major}.{minor - 1}"


def _update_python_versions(
    new_full_version: str,
    patch_only: bool = False,
    dry_run: bool = False,
) -> None:
    """Update Python version strings in relevant project files."""
    new_major_minor = ".".join(new_full_version.split(".")[:2])
    replacements: ReplacementsByPattern

    if patch_only:
        print(f"Updating patch version for {new_major_minor} to {new_full_version}")
        version_pattern = re.escape(new_major_minor) + r"\.\d+"

        replacements = {
            "dev/*.sh": [
                (
                    r"(version=\$\{1:-)" + version_pattern + r"(\})",
                    r"\g<1>" + new_full_version + r"\g<2>",
                ),
                (
                    r"(pyenv uninstall -f flower-)" + version_pattern,
                    r"\g<1>" + new_full_version,
                ),
            ],
            "**/*.py": [
                (
                    r'(["\'])' + version_pattern + r'(["\'])',
                    r"\g<1>" + new_full_version + r"\g<2>",
                ),
            ],
            "framework/docs/source/conf.py": [
                (
                    r"(\.\.\s*\|python_full_version\|\s*replace::\s*)"
                    + version_pattern,
                    r"\g<1>" + new_full_version,
                ),
            ],
        }
    else:
        old_version = _compute_old_version(new_major_minor)

        print(f"Determined old version: {old_version}")
        print(
            f"Updating to new version: {new_major_minor} "
            f"(full version: {new_full_version})"
        )

        replacements = {
            ".github/actions/bootstrap/action.yml": [
                (
                    r"^(\s*default:\s*)" + re.escape(old_version) + r"(\s*)$",
                    r"\g<1>" + new_major_minor + r"\g<2>",
                ),
            ],
            ".github/workflows/*.yml": [
                (
                    r"^(\s*python-version:\s*)"
                    + re.escape(old_version)
                    + r"(\s*)$",
                    r"\g<1>" + new_major_minor + r"\g<2>",
                ),
                (
                    r"(['\"]?)" + re.escape(old_version) + r"(['\"]?,?\s*)",
                    lambda match: ""
                    if match.group(2).strip() == ","
                    else "",
                ),
            ],
            "dev/*.sh": [
                (
                    r"(version=\$\{1:-)"
                    + re.escape(old_version)
                    + r"(\.\d+)?(\})",
                    r"\g<1>" + new_full_version + r"\g<3>",
                ),
                (
                    r"(pyenv uninstall -f flower-)"
                    + re.escape(old_version)
                    + r"(\.\d+)?",
                    r"\g<1>" + new_full_version,
                ),
            ],
            "**/pyproject.toml": [
                (
                    r'(python\s*=\s*">=)'
                    + re.escape(old_version)
                    + r'(,\s*<\d+\.\d+")',
                    r"\g<1>" + new_major_minor + r"\g<2>",
                ),
            ],
            "dev/devtool/*.py": [
                (
                    r'(["\'])'
                    + re.escape(old_version)
                    + r'(\.\d+)?(["\'],?)\s*\n?',
                    lambda match: ""
                    if match.group(3) == ","
                    else "",
                ),
            ],
            "**/*.py": [
                (
                    r'(["\'])'
                    + re.escape(old_version)
                    + r'(\.\d+)?(["\'])',
                    r"\g<1>" + new_full_version + r"\g<3>",
                ),
            ],
            "framework/docs/source/conf.py": [
                (
                    r"(\.\.\s*\|python_version\|\s*replace::\s*)"
                    + re.escape(old_version),
                    r"\g<1>" + new_major_minor,
                ),
                (
                    r"(\.\.\s*\|python_full_version\|\s*replace::\s*)"
                    + re.escape(old_version)
                    + r"\.\d+",
                    r"\g<1>" + new_full_version,
                ),
            ],
            "framework/docs/source/*.rst": [
                (
                    r"(`Python\s*"
                    + re.escape(old_version)
                    + r"\s*<https://docs.python.org/"
                    + re.escape(old_version)
                    + r"/>`_)",
                    r"`Python "
                    + new_major_minor
                    + " <https://docs.python.org/"
                    + new_major_minor
                    + "/>`_",
                ),
            ],
            "framework/docs/locales/*/LC_MESSAGES/framework-docs.po": [
                (
                    r"(`Python\s*"
                    + re.escape(old_version)
                    + r"\s*<https://docs.python.org/"
                    + re.escape(old_version)
                    + r"/>`_)",
                    r"`Python "
                    + new_major_minor
                    + " <https://docs.python.org/"
                    + new_major_minor
                    + "/>`_",
                ),
            ],
        }

    for file_pattern, pattern_replacements in replacements.items():
        for file_path in Path().rglob(file_pattern):
            if not file_path.is_file():
                continue

            content = file_path.read_text()
            original_content = content

            for pattern, replacement in pattern_replacements:
                content = re.sub(
                    pattern,
                    replacement,
                    content,
                    flags=re.MULTILINE,
                )

            if content != original_content:
                if dry_run:
                    print(f"Would update {file_path}")
                else:
                    file_path.write_text(content)
                    print(f"Updated {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to update Python versions in the codebase."
    )
    parser.add_argument(
        "new_full_version",
        help="New full Python version to use (e.g., 3.9.22)",
    )
    parser.add_argument(
        "--patch-only",
        action="store_true",
        help="Update only the patch version for matching major.minor versions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without modifying files.",
    )
    arguments = parser.parse_args()

    _update_python_versions(
        new_full_version=arguments.new_full_version,
        patch_only=arguments.patch_only,
        dry_run=arguments.dry_run,
    )