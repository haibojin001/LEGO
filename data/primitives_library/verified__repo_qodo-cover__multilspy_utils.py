import gzip
import logging
import os
import platform
import shutil
import subprocess
import uuid
from enum import Enum
from pathlib import Path, PurePath
from typing import Tuple

import requests

from cover_agent.lsp_logic.multilspy.multilspy_exceptions import MultilspyException
from cover_agent.lsp_logic.multilspy.multilspy_logger import MultilspyLogger


class TextUtils:
    """
    Utilities for text operations.
    """

    @staticmethod
    def get_line_col_from_index(text: str, index: int) -> Tuple[int, int]:
        """
        Returns the zero-indexed line and column number of the given index in the given text
        """
        line = 0
        column = 0
        current_index = 0

        while current_index < index:
            if text[current_index] == "\n":
                line += 1
                column = 0
            else:
                column += 1
            current_index += 1

        return line, column

    @staticmethod
    def get_index_from_line_col(text: str, line: int, col: int) -> int:
        """
        Returns the index of the given zero-indexed line and column number in the given text
        """
        index = 0

        while line > 0:
            assert index < len(text), (index, len(text), text)
            if text[index] == "\n":
                line -= 1
            index += 1

        return index + col

    @staticmethod
    def get_updated_position_from_line_and_column_and_edit(
        l: int, c: int, text_to_be_inserted: str
    ) -> Tuple[int, int]:
        """
        Utility function to get the position of the cursor after inserting text at a given line and column.
        """
        newline_count = text_to_be_inserted.count("\n")

        if newline_count > 0:
            l += newline_count
            c = len(text_to_be_inserted.split("\n")[-1])
        else:
            c += len(text_to_be_inserted)

        return l, c


class PathUtils:
    """
    Utilities for platform-agnostic path operations.
    """

    @staticmethod
    def uri_to_path(uri: str) -> str:
        """
        Converts a URI to a file path. Works on both Linux and Windows.

        This method was obtained from https://stackoverflow.com/a/61922504
        """
        try:
            from urllib.parse import unquote, urlparse
            from urllib.request import url2pathname
        except ImportError:
            from urlparse import urlparse
            from urllib import unquote, url2pathname

        parsed_uri = urlparse(uri)
        host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed_uri.netloc)
        return os.path.normpath(
            os.path.join(host, url2pathname(unquote(parsed_uri.path)))
        )


class FileUtils:
    """
    Utility functions for file operations.
    """

    @staticmethod
    def read_file(logger: MultilspyLogger, file_path: str) -> str:
        """
        Reads the file at the given path and returns the contents as a string.
        """
        try:
            for encoding in ["utf-8-sig", "utf-16"]:
                try:
                    with open(file_path, "r", encoding=encoding) as input_file:
                        return input_file.read()
                except UnicodeError:
                    continue
        except Exception as exc:
            logger.log(f"File read '{file_path}' failed: {exc}", logging.ERROR)
            raise MultilspyException("File read failed.") from None

        logger.log(
            f"File read '{file_path}' failed: Unsupported encoding.",
            logging.ERROR,
        )
        raise MultilspyException(
            f"File read '{file_path}' failed: Unsupported encoding."
        ) from None

    @staticmethod
    def download_file(logger: MultilspyLogger, url: str, target_path: str) -> None:
        """
        Downloads the file from the given URL to the given {target_path}
        """
        try:
            response = requests.get(url, stream=True, timeout=60)

            if response.status_code != 200:
                logger.log(
                    f"Error downloading file '{url}': "
                    f"{response.status_code} {response.text}",
                    logging.ERROR,
                )
                raise MultilspyException("Error downoading file.")

            with open(target_path, "wb") as output_file:
                shutil.copyfileobj(response.raw, output_file)
        except Exception as exc:
            logger.log(f"Error downloading file '{url}': {exc}", logging.ERROR)
            raise MultilspyException("Error downoading file.") from None

    @staticmethod
    def download_and_extract_archive(
        logger: MultilspyLogger, url: str, target_path: str, archive_type: str
    ) -> None:
        """
        Downloads the archive from the given URL having format {archive_type} and extracts it to the given {target_path}
        """
        try:
            temporary_files = []
            temporary_file_name = str(
                PurePath(
                    os.path.expanduser("~"),
                    "multilspy_tmp",
                    uuid.uuid4().hex,
                )
            )
            temporary_files.append(temporary_file_name)

            os.makedirs(os.path.dirname(temporary_file_name), exist_ok=True)
            FileUtils.download_file(logger, url, temporary_file_name)

            if archive_type in ["zip", "tar", "gztar", "bztar", "xztar"]:
                assert os.path.isdir(target_path)
                shutil.unpack_archive(
                    temporary_file_name,
                    target_path,
                    archive_type,
                )
            elif archive_type == "zip.gz":
                assert os.path.isdir(target_path)
                uncompressed_file_name = temporary_file_name + ".zip"
                temporary_files.append(uncompressed_file_name)

                with gzip.open(temporary_file_name, "rb") as compressed_file:
                    with open(uncompressed_file_name, "wb") as uncompressed_file:
                        shutil.copyfileobj(compressed_file, uncompressed_file)

                shutil.unpack_archive(uncompressed_file_name, target_path, "zip")
            elif archive_type == "gz":
                with gzip.open(temporary_file_name, "rb") as compressed_file:
                    with open(target_path, "wb") as output_file:
                        shutil.copyfileobj(compressed_file, output_file)
            else:
                logger.log(
                    f"Unknown archive type '{archive_type}' for extraction",
                    logging.ERROR,
                )
                raise MultilspyException(
                    f"Unknown archive type '{archive_type}'"
                )
        except Exception as exc:
            logger.log(
                f"Error extracting archive '{temporary_file_name}' "
                f"obtained from '{url}': {exc}",
                logging.ERROR,
            )
            raise MultilspyException("Error extracting archive.") from exc
        finally:
            for temporary_file_name in temporary_files:
                if os.path.exists(temporary_file_name):
                    Path.unlink(Path(temporary_file_name))


class PlatformId(str, Enum):
    """
    multilspy supported platforms
    """

    WIN_x86 = "win-x86"
    WIN_x64 = "win-x64"
    WIN_arm64 = "win-arm64"
    OSX = "osx"
    OSX_x64 = "osx-x64"
    OSX_arm64 = "osx-arm64"
    LINUX_x86 = "linux-x86"
    LINUX_x64 = "linux-x64"
    LINUX_arm64 = "linux-arm64"
    LINUX_MUSL_x64 = "linux-musl-x64"
    LINUX_MUSL_arm64 = "linux-musl-arm64"
    DARWIN_x64 = "darwin-x64"


class DotnetVersion(str, Enum):
    """
    multilspy supported dotnet versions
    """

    V4 = "4"
    V6 = "6"
    V7 = "7"
    V8 = "8"
    VMONO = "mono"


class PlatformUtils:
    """
    This class provides utilities for platform detection and identification.
    """

    @staticmethod
    def get_platform_id() -> PlatformId:
        """
        Returns the platform id for the current system
        """
        system = platform.system()
        machine = platform.machine()
        bitness = platform.architecture()[0]

        system_map = {
            "Windows": "win",
            "Darwin": "osx",
            "Linux": "linux",
        }
        machine_map = {
            "AMD64": "x64",
            "x86_64": "x64",
            "i386": "x86",
            "i686": "x86",
            "aarch64": "arm64",
            "arm64": "arm64",
        }

        if system in system_map and machine in machine_map:
            platform_id = system_map[system] + "-" + machine_map[machine]

            if system == "Linux" and bitness == "64bit":
                libc = platform.libc_ver()[0]
                if libc != "glibc":
                    platform_id += "-" + libc

            return PlatformId(platform_id)

        raise MultilspyException(
            "Unknown platform: " + system + " " + machine + " " + bitness
        )

    @staticmethod
    def get_dotnet_version() -> DotnetVersion:
        """
        Returns the dotnet version for the current system
        """
        try:
            result = subprocess.run(
                ["dotnet", "--list-runtimes"],
                capture_output=True,
                check=True,
            )

            version = ""
            for line in result.stdout.decode("utf-8").split("\n"):
                if line.startswith("Microsoft.NETCore.App"):
                    version = line.split(" ")[1].split(".")[0]
                    break

            if version == "":
                return DotnetVersion.VMONO

            return DotnetVersion(version)
        except Exception:
            return DotnetVersion.VMONO