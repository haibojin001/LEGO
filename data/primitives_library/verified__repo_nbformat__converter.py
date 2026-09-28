from __future__ import annotations

from .reader import get_version
from .validator import ValidationError


def convert(nb, to_version):
    """Convert a notebook node object to a specific version.

    Converts notebooks one major version at a time, either upgrading or
    downgrading until the requested version is reached.
    """
    # Import lazily because ``versions`` is initialized by nbformat.__init__
    # after this module is imported.
    from . import versions

    version, _version_minor = get_version(nb)

    if version == to_version:
        return nb

    if to_version in versions:
        if to_version > version:
            step_version = version + 1
            convert_function = versions[step_version].upgrade
        else:
            step_version = version - 1
            convert_function = versions[version].downgrade

        try:
            converted = convert_function(nb)
            if converted.get("nbformat", 1) == version:
                raise ValueError(
                    "Failed to convert notebook from v%d to v%d."
                    % (version, step_version)
                )
        except AttributeError as error:
            raise ValidationError(
                f"Notebook could not be converted from version {version} "
                f"to version {step_version} because it's missing a key: {error}"
            ) from None

        return convert(converted, to_version)

    raise ValueError(
        "Cannot convert notebook to v%d because that version doesn't exist"
        % to_version
    )