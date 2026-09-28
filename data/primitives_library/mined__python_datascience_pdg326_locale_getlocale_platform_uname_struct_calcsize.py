# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg326::locale.getlocale+platform.uname+struct.calcsize
# name: locale_platform_primitive
# summary: Uses locale.getlocale, platform.uname, struct.calcsize across 2 repos
# anchor_symbols: ['locale.getlocale', 'platform.uname', 'struct.calcsize']
# observed in 2 repos: ['alteryx__evalml', 'alteryx__featuretools']...

# --- from alteryx__featuretools::featuretools/utils/utils_info.py::get_sys_info ---
def get_sys_info():
    "Returns system information as a dict"

    blob = []

    try:
        (sysname, nodename, release, version, machine, processor) = platform.uname()
        blob.extend(
            [
                ("python", ".".join(map(str, sys.version_info))),
                ("python-bits", struct.calcsize("P") * 8),
                ("OS", "{sysname}".format(sysname=sysname)),
                ("OS-release", "{release}".format(release=release)),
                ("machine", "{machine}".format(machine=machine)),
                ("processor", "{processor}".format(processor=processor)),
                ("byteorder", "{byteorder}".format(byteorder=sys.byteorder)),
                ("LC_ALL", "{lc}".format(lc=os.environ.get("LC_ALL", "None"))),
                ("LANG", "{lang}".format(lang=os.environ.get("LANG", "None"))),
                ("LOCALE", ".".join(map(str, locale.getlocale()))),
            ],
        )
    except (KeyError, ValueError):
        pass

    return blob

# --- from alteryx__evalml::evalml/utils/cli_utils.py::get_sys_info ---
def get_sys_info():
    """Returns system information.

    Returns:
        List of tuples about system stats.
    """
    blob = []
    try:
        (sysname, nodename, release, version, machine, processor) = platform.uname()
        blob.extend(
            [
                ("python", ".".join(map(str, sys.version_info))),
                ("python-bits", struct.calcsize("P") * 8),
                ("OS", "{sysname}".format(sysname=sysname)),
                ("OS-release", "{release}".format(release=release)),
                ("machine", "{machine}".format(machine=machine)),
                ("processor", "{processor}".format(processor=processor)),
                ("byteorder", "{byteorder}".format(byteorder=sys.byteorder)),
                ("LC_ALL", "{lc}".format(lc=os.environ.get("LC_ALL", "None"))),
                ("LANG", "{lang}".format(lang=os.environ.get("LANG", "None"))),
                ("LOCALE", ".".join(map(str, locale.getlocale()))),
            ],
        )
    except (KeyError, ValueError):
        pass

    return blob
