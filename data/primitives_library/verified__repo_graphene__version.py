import datetime
import os
import subprocess


def get_complete_version(version=None):
    """Return the configured version tuple, validating supplied versions."""
    if version is None:
        from graphene import VERSION

        return VERSION

    assert len(version) == 5
    assert version[3] in ("alpha", "beta", "rc", "final")
    return version


def get_main_version(version=None):
    """Return the numeric X.Y or X.Y.Z portion of a version."""
    complete = get_complete_version(version)
    count = 2 if complete[2] == 0 else 3
    return ".".join(str(part) for part in complete[:count])


def get_git_changeset():
    """Return the UTC timestamp of the latest Git commit, if available."""
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        process = subprocess.Popen(
            "git log --pretty=format:%ct --quiet -1 HEAD",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            cwd=directory,
            universal_newlines=True,
        )
        value = process.communicate()[0]
        moment = datetime.datetime.utcfromtimestamp(int(value))
    except Exception:
        return None
    return moment.strftime("%Y%m%d%H%M%S")


def get_version(version=None):
    """Return a PEP 440 version string."""
    complete = get_complete_version(version)
    result = get_main_version(complete)

    if complete[3] == "alpha" and complete[4] == 0:
        changeset = get_git_changeset()
        suffix = ".dev%s" % changeset if changeset else ".dev"
    elif complete[3] != "final":
        suffix = {"alpha": "a", "beta": "b", "rc": "rc"}[complete[3]] + str(
            complete[4]
        )
    else:
        suffix = ""

    return str(result + suffix)


def get_docs_version(version=None):
    complete = get_complete_version(version)
    if complete[3] != "final":
        return "dev"
    return "%d.%d" % complete[:2]