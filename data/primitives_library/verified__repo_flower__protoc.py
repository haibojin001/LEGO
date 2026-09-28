from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc_tools
from grpc_tools import protoc


class ProtocConfigError(ValueError):
    """Indicates an invalid protoc project configuration."""


class ProtocExecutionError(RuntimeError):
    """Indicates an unsuccessful grpc_tools.protoc invocation."""


@dataclass(frozen=True)
class ProtocOutputs:
    """Directories receiving generated protobuf artifacts."""

    python: Path
    grpc_python: Path
    mypy: Path
    mypy_grpc: Path


@dataclass(frozen=True)
class ProtocConfig:
    """Fully resolved protobuf compiler settings."""

    project_dir: Path
    proto_root: Path
    include_globs: tuple[str, ...]
    include_paths: tuple[Path, ...]
    outputs: ProtocOutputs


def _load_pyproject(project_dir: Path) -> dict[str, Any]:
    path = project_dir / "pyproject.toml"
    if not path.is_file():
        raise ProtocConfigError(f"Missing pyproject.toml: {path}")
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _resolve_dir(project_dir: Path, value: str, *, field_name: str) -> Path:
    directory = (project_dir / value).resolve()
    if not directory.is_dir():
        raise ProtocConfigError(
            f"Configured `{field_name}` directory does not exist: {directory}"
        )
    return directory


def _require_string(config: dict[str, Any], key: str) -> str:
    item = config.get(key)
    if not isinstance(item, str) or not item:
        raise ProtocConfigError(f"Expected `{key}` to be a non-empty string")
    return item


def _require_string_list(config: dict[str, Any], key: str) -> tuple[str, ...]:
    items = config.get(key)
    if not isinstance(items, list) or not items:
        raise ProtocConfigError(f"Expected `{key}` to be a non-empty list")
    if any(not isinstance(item, str) or not item for item in items):
        raise ProtocConfigError(f"Expected `{key}` to contain only non-empty strings")
    return tuple(items)


def _load_outputs(project_dir: Path, config: dict[str, Any]) -> ProtocOutputs:
    values = config.get("outputs")
    if not isinstance(values, dict):
        raise ProtocConfigError("Missing `[tool.devtool.protoc.outputs]` table")

    return ProtocOutputs(
        python=_resolve_dir(
            project_dir,
            _require_string(values, "python"),
            field_name="outputs.python",
        ),
        grpc_python=_resolve_dir(
            project_dir,
            _require_string(values, "grpc_python"),
            field_name="outputs.grpc_python",
        ),
        mypy=_resolve_dir(
            project_dir,
            _require_string(values, "mypy"),
            field_name="outputs.mypy",
        ),
        mypy_grpc=_resolve_dir(
            project_dir,
            _require_string(values, "mypy_grpc"),
            field_name="outputs.mypy_grpc",
        ),
    )


def load_protoc_config(project_dir: Path) -> ProtocConfig:
    """Read and resolve protobuf compiler configuration from pyproject.toml."""
    root = project_dir.resolve()
    document = _load_pyproject(root)

    tool_config = document.get("tool")
    if not isinstance(tool_config, dict):
        raise ProtocConfigError("Missing `[tool.devtool.protoc]` configuration")

    devtool_config = tool_config.get("devtool")
    if not isinstance(devtool_config, dict):
        raise ProtocConfigError("Missing `[tool.devtool.protoc]` configuration")

    settings = devtool_config.get("protoc")
    if not isinstance(settings, dict):
        raise ProtocConfigError("Missing `[tool.devtool.protoc]` configuration")

    proto_root = _resolve_dir(
        root,
        _require_string(settings, "proto_root"),
        field_name="proto_root",
    )
    include_globs = _require_string_list(settings, "include_globs")
    include_paths = tuple(
        _resolve_dir(root, path, field_name="include_paths")
        for path in _require_string_list(settings, "include_paths")
    )

    return ProtocConfig(
        project_dir=root,
        proto_root=proto_root,
        include_globs=include_globs,
        include_paths=include_paths,
        outputs=_load_outputs(root, settings),
    )


def discover_proto_files(
    proto_root: Path, include_globs: tuple[str, ...]
) -> list[Path]:
    """Find matched protobuf files in deterministic order."""
    files = {
        candidate.resolve()
        for pattern in include_globs
        for candidate in proto_root.glob(pattern)
        if candidate.is_file()
    }
    result = sorted(files, key=lambda path: path.as_posix())
    if not result:
        raise ProtocConfigError(
            "No `.proto` files matched the configured include globs under "
            f"{proto_root}"
        )
    return result


def build_protoc_command(config: ProtocConfig, proto_files: list[Path]) -> list[str]:
    """Construct the argument vector used for grpc_tools.protoc."""
    bundled_include = Path(grpc_tools.__path__[0]) / "_proto"
    include_dirs: list[Path] = []

    for directory in (
        bundled_include,
        config.proto_root,
        *config.include_paths,
    ):
        if directory not in include_dirs:
            include_dirs.append(directory)

    arguments = ["grpc_tools.protoc"]
    arguments.extend(f"--proto_path={directory}" for directory in include_dirs)
    arguments.extend(
        (
            f"--python_out={config.outputs.python}",
            f"--grpc_python_out={config.outputs.grpc_python}",
            f"--mypy_out={config.outputs.mypy}",
            f"--mypy_grpc_out={config.outputs.mypy_grpc}",
        )
    )
    arguments.extend(str(path) for path in proto_files)
    return arguments


def compile_project(project_dir: Path) -> None:
    """Compile every configured protobuf source for a project."""
    config = load_protoc_config(project_dir)
    sources = discover_proto_files(config.proto_root, config.include_globs)
    status = protoc.main(build_protoc_command(config, sources))
    if status != 0:
        raise ProtocExecutionError(
            f"`grpc_tools.protoc` failed with exit code {status}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Compile protobufs for a project.")
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project directory containing the pyproject.toml config",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line compiler entrypoint."""
    options = parse_args()
    compile_project(Path(options.project_dir))


if __name__ == "__main__":
    main()