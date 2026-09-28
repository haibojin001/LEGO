import os
import subprocess
import sys
from typing import Dict, Optional


class LocalController:
    """Controller for running shell commands and Python snippets on this machine."""

    def run_bash_script(self, code: str, timeout: int = 30) -> Dict:
        try:
            result = subprocess.run(
                ["/bin/bash", "-lc", code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            text_output = (result.stdout or "") + (result.stderr or "")

            print("BASH OUTPUT =======================================")
            print(text_output)
            print("BASH OUTPUT =======================================")

            return {
                "status": "ok" if result.returncode == 0 else "error",
                "returncode": result.returncode,
                "output": text_output,
                "error": "",
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "error",
                "returncode": -1,
                "output": exc.stdout or "",
                "error": f"TimeoutExpired: {str(exc)}",
            }
        except Exception as exc:
            return {
                "status": "error",
                "returncode": -1,
                "output": "",
                "error": str(exc),
            }

    def run_python_script(self, code: str) -> Dict:
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
            )

            print("PYTHON OUTPUT =======================================")
            print(result.stdout or "")
            print("PYTHON OUTPUT =======================================")

            return {
                "status": "ok" if result.returncode == 0 else "error",
                "return_code": result.returncode,
                "output": result.stdout or "",
                "error": result.stderr or "",
            }
        except Exception as exc:
            return {
                "status": "error",
                "return_code": -1,
                "output": "",
                "error": str(exc),
            }


class LocalEnv:
    """Local execution environment exposing a controller."""

    def __init__(self):
        self.controller = LocalController()


def cuda_count():
    import torch

    return torch.cuda.device_count()


def get_real_path(path: str) -> Optional[str]:
    if os.path.isdir(path):
        contents = os.listdir(path)
        if contents:
            resolved = os.path.realpath(os.path.join(path, contents[0]))
            if resolved:
                return os.path.dirname(resolved)
        return None
    return os.path.realpath(path)


def get_pip_config_args() -> dict[str, str | list[str]]:
    name_map = {
        "global.index-url": "index_url",
        "global.extra-index-url": "extra_index_url",
        "global.trusted-host": "trusted_host",
        "global.find-links": "find_links",
    }
    multi_value_options = {
        "extra_index_url",
        "trusted_host",
        "find_links",
    }

    try:
        process = subprocess.run(
            [sys.executable, "-m", "pip", "config", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        result: dict[str, str | list[str]] = {}

        for line in process.stdout.splitlines():
            if "=" not in line:
                continue

            raw_name, raw_value = line.split("=", 1)
            config_name = raw_name.strip()
            config_value = raw_value.strip().strip("'\"")
            argument_name = name_map.get(config_name)

            if argument_name is None or not config_value:
                continue

            if argument_name not in multi_value_options:
                result[argument_name] = config_value
                continue

            existing = result.get(argument_name)
            if existing is None:
                result[argument_name] = config_value
            elif isinstance(existing, list):
                existing.append(config_value)
            else:
                result[argument_name] = [existing, config_value]

        return result
    except subprocess.CalledProcessError:
        return {}


def make_hashable(obj):
    if isinstance(obj, (tuple, list)):
        return tuple(make_hashable(value) for value in obj)

    if isinstance(obj, dict):
        return tuple(
            sorted((key, make_hashable(value)) for key, value in obj.items())
        )

    if isinstance(obj, set):
        return tuple(sorted(make_hashable(value) for value in obj))

    return obj