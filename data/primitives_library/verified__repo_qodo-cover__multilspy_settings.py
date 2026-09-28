import os
import pathlib


class MultilspySettings:
    """
    Provides settings related to multilspy language servers and caching.
    """

    @staticmethod
    def get_language_server_directory() -> str:
        """Return the directory used to store language server installations."""
        home_directory = pathlib.Path.home()
        multilspy_directory = str(pathlib.PurePath(home_directory, ".multilspy"))
        language_server_directory = str(
            pathlib.PurePath(multilspy_directory, "lsp")
        )
        os.makedirs(language_server_directory, exist_ok=True)
        return language_server_directory

    @staticmethod
    def get_global_cache_directory() -> str:
        """Return the directory used for the global multilspy cache."""
        cache_directory = os.path.join(
            str(pathlib.Path.home()),
            ".multilspy",
            "global_cache",
        )
        os.makedirs(cache_directory, exist_ok=True)
        return cache_directory