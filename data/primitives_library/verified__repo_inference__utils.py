import os
import subprocess
import sys
from typing import Optional


def cuda_count():
    import torch

    return torch.cuda.device_count()


def get_real_path(path: str) -> Optional[str]:
    if os.path.isdir(path):
        entries = os.listdir(path)
        if entries:
            target = os.path.realpath(os.path.join(path, entries[0]))
            if target:
                return os.path.dirname(target)
        return None
    return os.path.realpath(path)


def get_pip_config_args() -> dict[str, str | list[str]]:
    aliases = {
        "global.index-url": "index_url",
        "global.extra-index-url": "extra_index_url",
        "global.trusted-host": "trusted_host",
        "global.find-links": "find_links",
    }

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "config", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        options: dict[str, str | list[str]] = {}

        for row in completed.stdout.splitlines():
            if "=" not in row:
                continue

            name, setting = row.split("=", 1)
            name = name.strip()
            setting = setting.strip().strip("'\"")
            option_name = aliases.get(name)

            if option_name is None or not setting:
                continue

            if option_name in {
                "extra_index_url",
                "find_links",
                "trusted_host",
            }:
                previous = options.get(option_name)
                if previous is None:
                    options[option_name] = setting
                elif isinstance(previous, list):
                    previous.append(setting)
                else:
                    options[option_name] = [previous, setting]
            else:
                options[option_name] = setting

        return options
    except subprocess.CalledProcessError:
        return {}


def make_hashable(obj):
    if isinstance(obj, (tuple, list)):
        return tuple(make_hashable(item) for item in obj)

    if isinstance(obj, dict):
        return tuple(
            sorted((key, make_hashable(value)) for key, value in obj.items())
        )

    if isinstance(obj, set):
        return tuple(sorted(make_hashable(item) for item in obj))

    return obj