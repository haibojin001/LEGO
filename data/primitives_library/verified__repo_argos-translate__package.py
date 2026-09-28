from __future__ import annotations

import copy
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from threading import Lock

import packaging.version

from argostranslate import networking, settings
from argostranslate.tokenizer import BPETokenizer, SentencePieceTokenizer
from argostranslate.utils import error, info, warning


package_lock = Lock()


def _package_data_dir() -> Path:
    package_data_dir = getattr(settings, "package_data_dir", None)
    if package_data_dir is not None:
        return Path(package_data_dir)

    data_dir = getattr(
        settings,
        "data_dir",
        Path.home() / ".local" / "share" / "argos-translate",
    )
    return Path(data_dir) / "packages"


def _package_index_path() -> Path:
    package_index_path = getattr(settings, "package_index_path", None)
    if package_index_path is not None:
        return Path(package_index_path)

    package_index = getattr(settings, "package_index", None)
    if package_index is not None:
        return Path(package_index)

    data_dir = getattr(
        settings,
        "data_dir",
        Path.home() / ".local" / "share" / "argos-translate",
    )
    return Path(data_dir) / "index.json"


def _remote_package_index() -> str:
    return getattr(
        settings,
        "remote_package_index",
        "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json",
    )


def _downloads_dir() -> Path:
    downloads_dir = getattr(settings, "downloads_dir", None)
    if downloads_dir is not None:
        return Path(downloads_dir)

    cache_dir = getattr(settings, "cache_dir", None)
    if cache_dir is not None:
        return Path(cache_dir)

    data_dir = getattr(
        settings,
        "data_dir",
        Path.home() / ".local" / "share" / "argos-translate",
    )
    return Path(data_dir) / "downloads"


class IPackage:
    code: str
    package_path: Path | None
    package_version: str
    argos_version: str
    from_code: str | None
    from_name: str
    from_codes: list
    to_code: str | None
    to_codes: list
    to_name: str
    links: list[str]
    type: str
    languages: list
    dependencies: list
    source_languages: list
    target_languages: list

    def load_metadata_from_json(self, metadata):
        self.code = metadata.get("code")
        self.package_version = metadata.get("package_version", "")
        self.argos_version = metadata.get("argos_version", "")
        self.from_code = metadata.get("from_code")
        self.from_name = metadata.get("from_name", "")
        self.from_codes = metadata.get("from_codes", list())
        self.to_code = metadata.get("to_code")
        self.to_codes = metadata.get("to_codes", list())
        self.to_name = metadata.get("to_name", "")
        self.links = metadata.get("links", list())
        self.type = metadata.get("type", "translate")
        self.languages = metadata.get("languages", list())
        self.dependencies = metadata.get("dependencies", list())
        self.source_languages = metadata.get("source_languages", list())
        self.target_languages = metadata.get("target_languages", list())
        self.target_prefix = metadata.get("target_prefix", "")

        if self.from_code is not None or self.from_name is not None:
            from_language = {}
            if self.from_code is not None:
                from_language["code"] = self.from_code
            if self.from_name is not None:
                from_language["name"] = self.from_name
            self.source_languages.append(from_language)

        if self.to_code is not None or self.to_name is not None:
            to_language = {}
            if self.to_code is not None:
                to_language["code"] = self.to_code
            if self.to_name is not None:
                to_language["name"] = self.to_name
            self.source_languages.append(to_language)

        self.source_languages += copy.deepcopy(self.languages)
        self.target_languages += copy.deepcopy(self.languages)

    def get_readme(self) -> str | None:
        raise NotImplementedError()

    def get_description(self):
        raise NotImplementedError()

    def __eq__(self, other):
        return (
            self.package_version == other.package_version
            and self.argos_version == other.argos_version
            and self.from_code == other.from_code
            and self.from_name == other.from_name
            and self.to_code == other.to_code
            and self.to_name == other.to_name
        )

    def __repr__(self):
        if len(self.from_name) > 0 and len(self.to_name) > 0:
            return "{} -> {}".format(self.from_name, self.to_name)
        elif self.type:
            return self.type
        return ""

    def __str__(self):
        return repr(self).replace("->", "→")


class Package(IPackage):
    def __init__(self, package_path: Path):
        if type(package_path) == str:
            package_path = Path(package_path)

        self.package_path = package_path
        metadata_path = package_path / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                "Error opening package at " + str(metadata_path) + " no metadata.json"
            )

        with open(metadata_path) as metadata_file:
            self.load_metadata_from_json(json.load(metadata_file))

        minisbd_package = package_path / "minisbd"
        stanza_package = package_path / "stanza"
        spacy_package = package_path / "spacy"

        if minisbd_package.exists():
            self.packaged_sbd_path = minisbd_package
        elif stanza_package.exists():
            self.packaged_sbd_path = stanza_package
        elif spacy_package.exists():
            self.packaged_sbd_path = spacy_package
        else:
            self.packaged_sbd_path = None

        sentencepiece_model = package_path / "sentencepiece.model"
        bpe_model = package_path / "bpe.model"

        if sentencepiece_model.exists():
            self.tokenizer = SentencePieceTokenizer(sentencepiece_model)
        elif bpe_model.exists():
            self.tokenizer = BPETokenizer(
                bpe_model,
                self.from_code,
                self.to_code,
            )

    def update(self):
        for available_package in get_available_packages():
            if (
                available_package.from_code == self.from_code
                and available_package.to_code == self.to_code
            ):
                if packaging.version.parse(
                    available_package.package_version
                ) > packaging.version.parse(self.package_version):
                    package_path = available_package.download()
                    uninstall(self)
                    install_from_path(package_path)

    def get_readme(self) -> str | None:
        readme_path = self.package_path / "README.md"
        if not readme_path.exists():
            return None

        with open(readme_path, "r") as readme_file:
            return readme_file.read()

    def get_description(self):
        return self.get_readme()


class AvailablePackage(IPackage):
    def __init__(self, metadata):
        self.package_path = None
        self.load_metadata_from_json(metadata)
        self.metadata = metadata

    def download(self) -> Path:
        if len(self.links) == 0:
            raise ValueError("Package has no download links")

        downloads_dir = _downloads_dir()
        downloads_dir.mkdir(parents=True, exist_ok=True)

        if self.from_code is not None and self.to_code is not None:
            filename = self.from_code + "_" + self.to_code + ".argosmodel"
        elif self.code:
            filename = self.code + ".argosmodel"
        else:
            filename = Path(self.links[0].split("?", 1)[0]).name

        download_path = downloads_dir / filename

        last_error = None
        for link in self.links:
            try:
                urllib.request.urlretrieve(link, download_path)
                return download_path
            except Exception as exception:
                last_error = exception

        if last_error is not None:
            raise last_error
        return download_path

    def install(self):
        for dependency in self.dependencies:
            for available_package in get_available_packages():
                if (
                    available_package.code == dependency
                    or argospm_package_name(available_package) == dependency
                ):
                    available_package.install()
                    break

        install_from_path(self.download())

    def get_readme(self) -> str | None:
        return None

    def get_description(self):
        return self.get_readme()


def install_from_path(path: Path):
    with package_lock:
        if not zipfile.is_zipfile(path):
            raise ValueError("Package file is not a zip file")

        package_data_dir = _package_data_dir()
        package_data_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(path, "r") as package_zip:
            package_zip.extractall(package_data_dir)


def uninstall(package: Package):
    with package_lock:
        shutil.rmtree(package.package_path)


def get_installed_packages() -> list[Package]:
    package_data_dir = _package_data_dir()
    if not package_data_dir.exists():
        return []

    packages = []
    for package_path in package_data_dir.iterdir():
        if not package_path.is_dir():
            continue
        try:
            packages.append(Package(package_path))
        except Exception as exception:
            warning("Error loading package " + str(package_path) + ": " + str(exception))
    return packages


def update_package_index():
    with package_lock:
        try:
            package_index_path = _package_index_path()
            package_index_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                package_index_data = networking.get(_remote_package_index())
            except Exception:
                with urllib.request.urlopen(_remote_package_index()) as response:
                    package_index_data = response.read()

            if hasattr(package_index_data, "read"):
                package_index_data = package_index_data.read()

            if isinstance(package_index_data, str):
                package_index_data = package_index_data.encode()

            with open(package_index_path, "wb") as package_index_file:
                package_index_file.write(package_index_data)

            info("Package index updated")
        except Exception as exception:
            error("Error updating package index: " + str(exception))


def get_available_packages() -> list[AvailablePackage]:
    package_index_path = _package_index_path()

    with open(package_index_path, "r") as package_index_file:
        package_index = json.load(package_index_file)

    if isinstance(package_index, dict):
        package_index = package_index.get("packages", [])

    return [AvailablePackage(metadata) for metadata in package_index]


def argospm_package_name(package: IPackage) -> str:
    if getattr(package, "code", None):
        return package.code

    package_type = getattr(package, "type", "")
    from_code = getattr(package, "from_code", None)
    to_code = getattr(package, "to_code", None)

    if from_code is not None and to_code is not None:
        return "{}-{}_{}".format(package_type, from_code, to_code)

    return package_type