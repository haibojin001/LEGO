import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict

home_dir = Path.home()
if "SNAP" in os.environ:
    home_dir = Path(os.environ["SNAP_USER_DATA"])

data_dir = Path(
    os.getenv("XDG_DATA_HOME", home_dir / ".local" / "share")
) / "argos-translate"
os.makedirs(data_dir, exist_ok=True)

legacy_package_data_dir = Path(
    os.getenv("ARGOS_TRANSLATE_PACKAGE_DIR", data_dir / "packages")
)

config_dir = Path(
    os.getenv("XDG_CONFIG_HOME", home_dir / ".config")
) / "argos-translate"
os.makedirs(config_dir, exist_ok=True)

cache_dir = Path(
    os.getenv("XDG_CACHE_HOME", home_dir / ".local" / "cache")
) / "argos-translate"
os.makedirs(cache_dir, exist_ok=True)

downloads_dir = cache_dir / "downloads"
os.makedirs(downloads_dir, exist_ok=True)

settings_file = config_dir / "settings.json"


def load_settings_dict() -> Dict[str, Any]:
    loaded_settings = {}
    if settings_file.exists():
        try:
            with open(settings_file, "r") as stream:
                loaded_settings = json.load(stream)
            assert isinstance(
                loaded_settings, dict
            ), "settings.json should contain a dictionary"
        except FileNotFoundError as exception:
            print(f"{settings_file} not found : FileNotFoundError {exception}")
        except json.JSONDecodeError as exception:
            print(f"Error decoding {settings_file}: JSONDecodeError {exception}")
    return loaded_settings


def get_setting(key: str, default=None):
    """Return a setting, favoring an environment variable over settings.json."""
    environment_setting = os.getenv(key)
    stored_setting = load_settings_dict().get(key)

    if environment_setting is not None:
        return environment_setting
    if stored_setting is not None:
        return stored_setting
    return default


def set_setting(key: str, value):
    """Store a setting value in settings.json."""
    stored_settings = load_settings_dict()
    stored_settings[key] = value
    with open(settings_file, "w") as stream:
        json.dump(stored_settings, stream, indent=4)


TRUE_VALUES = ["1", "TRUE", "True", "true", 1, True]

debug = get_setting("ARGOS_DEBUG") in TRUE_VALUES
dev_mode = get_setting("ARGOS_DEV_MODE") in TRUE_VALUES

package_index = get_setting(
    "ARGOS_PACKAGE_INDEX",
    default="https://raw.githubusercontent.com/argosopentech/argospm-index/main/",
)

package_data_dir = Path(
    get_setting("ARGOS_PACKAGES_DIR", default=data_dir / "packages")
)
os.makedirs(package_data_dir, exist_ok=True)

downloads_dir = cache_dir / "downloads"
os.makedirs(downloads_dir, exist_ok=True)

if not dev_mode:
    remote_repo = os.getenv(
        "ARGOS_PACKAGE_INDEX",
        default="https://raw.githubusercontent.com/argosopentech/argospm-index/main",
    )
else:
    remote_repo = os.getenv(
        "ARGOS_PACKAGE_INDEX",
        default="https://raw.githubusercontent.com/argosopentech/argospm-index-dev/main",
    )

remote_package_index = package_index + "index.json"
local_package_index = data_dir / "index.json"

experimental_enabled = os.getenv("ARGOS_EXPERIMENTAL_ENABLED") in TRUE_VALUES

device = get_setting("ARGOS_DEVICE_TYPE", "cpu")
inter_threads = int(get_setting("ARGOS_INTER_THREADS", "1"))
intra_threads = int(get_setting("ARGOS_INTRA_THREADS", "0"))
batch_size = int(get_setting("ARGOS_BATCH_SIZE", "32"))
compute_type = get_setting("ARGOS_COMPUTE_TYPE", "auto")
beam_size = int(get_setting("ARGOS_BEAM_SIZE", "4"))


class ModelProvider(Enum):
    OPENNMT = 0
    LIBRETRANSLATE = 1
    OPENAI = 2


model_mapping = {
    "OPENNMT": ModelProvider.OPENNMT,
    "LIBRETRANSLATE": ModelProvider.LIBRETRANSLATE,
    "OPENAI": ModelProvider.OPENAI,
}

model_provider = model_mapping[
    get_setting("ARGOS_MODEL_PROVIDER", default="OPENNMT")
]


class ChunkType(Enum):
    DEFAULT = 0
    ARGOSTRANSLATE = 1
    NONE = 2
    STANZA = 3
    SPACY = 4
    MINISBD = 5


chunk_type_mapping = {
    "DEFAULT": ChunkType.DEFAULT,
    "ARGOSTRANSLATE": ChunkType.ARGOSTRANSLATE,
    "NONE": ChunkType.NONE,
    "STANZA": ChunkType.STANZA,
    "SPACY": ChunkType.SPACY,
    "MINISBD": ChunkType.MINISBD,
}

chunk_type = chunk_type_mapping[
    get_setting("ARGOS_CHUNK_TYPE", default="DEFAULT")
]
if chunk_type == ChunkType.DEFAULT:
    chunk_type = ChunkType.ARGOSTRANSLATE

libretranslate_api_key = get_setting("LIBRETRANSLATE_API_KEY", None)
openai_api_key = get_setting("OPENAI_API_KEY", None)

argos_translate_about_text = (
    "Argos Translate is an open source neural machine "
    + "translation application created by Argos Open "
    + "Technologies, LLC (www.argosopentech.com). "
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

package_dirs = [package_data_dir]
if "SNAP" in os.environ:
    snap_package_dir = Path(os.environ["SNAP"]) / "snap_custom" / "packages"
    if os.path.isdir(snap_package_dir):
        package_dirs.append(snap_package_dir)

    content_snap_packages = (
        Path(os.environ["SNAP"]) / "snap_custom" / "content_snap_packages"
    )
    if os.path.isdir(content_snap_packages):
        for package_dir in content_snap_packages.iterdir():
            if package_dir.is_dir():
                package_dirs.append(package_dir)