import sys

from os.path import abspath, dirname, exists, join

from dynaconf import Dynaconf


SETTINGS_FILES = [
    "test_generation_prompt.toml",
    "language_extensions.toml",
    "analyze_suite_test_headers_indentation.toml",
    "analyze_suite_test_insert_line.toml",
    "analyze_test_run_failure.toml",
    "analyze_test_against_context.toml",
    "adapt_test_command_for_a_single_test_via_ai.toml",
    "configuration.toml",
]


class SingletonSettings:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SingletonSettings, cls).__new__(
                cls, *args, **kwargs
            )
        return cls._instance

    def __init__(self):
        if hasattr(self, "settings"):
            return

        root_directory = getattr(sys, "_MEIPASS", dirname(abspath(__file__)))
        configuration_paths = [
            join(root_directory, filename) for filename in SETTINGS_FILES
        ]

        for configuration_path in configuration_paths:
            if not exists(configuration_path):
                raise FileNotFoundError(
                    f"Settings file not found: {configuration_path}"
                )

        self.settings = Dynaconf(
            envvar_prefix=False,
            merge_enabled=True,
            settings_files=configuration_paths,
        )


def get_settings() -> Dynaconf:
    return SingletonSettings().settings