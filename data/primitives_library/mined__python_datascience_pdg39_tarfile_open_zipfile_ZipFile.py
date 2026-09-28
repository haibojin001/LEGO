# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg39::tarfile.open+zipfile.ZipFile
# name: tarfile_zipfile_primitive
# summary: Uses tarfile.open, zipfile.ZipFile across 3 repos
# anchor_symbols: ['tarfile.open', 'zipfile.ZipFile']
# observed in 3 repos: ['awslabs__gluonts', 'd2l-ai__d2l-en', 'wooey__Wooey']...

# --- from d2l-ai__d2l-en::d2l/mxnet.py::extract ---
def extract(filename, folder=None):
    """Extract a zip/tar file into folder.

    Defined in :numref:`sec_utils`"""
    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)
    assert ext in ('.zip', '.tar', '.gz'), 'Only support zip/tar files.'
    if ext == '.zip':
        fp = zipfile.ZipFile(filename, 'r')
    else:
        fp = tarfile.open(filename, 'r')
    if folder is None:
        folder = base_dir
    fp.extractall(folder)

# --- from d2l-ai__d2l-en::d2l/tensorflow.py::extract ---
def extract(filename, folder=None):
    """Extract a zip/tar file into folder.

    Defined in :numref:`sec_utils`"""
    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)
    assert ext in ('.zip', '.tar', '.gz'), 'Only support zip/tar files.'
    if ext == '.zip':
        fp = zipfile.ZipFile(filename, 'r')
    else:
        fp = tarfile.open(filename, 'r')
    if folder is None:
        folder = base_dir
    fp.extractall(folder)

# --- from awslabs__gluonts::src/gluonts/nursery/tsbench/src/tsbench/utils/filesystem.py::compress_directory ---
def compress_directory(
    directory: Path, target: Path, include: Optional[Set[str]] = None
) -> None:
    """
    Compresses the provided directory into a single `.tar.gz` file.

    Args:
        directory: The directory to compress.
        target: The `.tar.gz` file where the compressed archive should be written.
        include: The filenames to include. If not provided, all files are included.
    """
    with target.open("wb+") as f:
        with tarfile.open(fileobj=f, mode="w:gz") as tar:
            for root, _, files in os.walk(directory):
                for file in files:
                    if include is not None and file not in include:
                        continue
                    name = os.path.join(root, file)
                    tar.add(name, arcname=os.path.relpath(name, directory))

# --- from wooey__Wooey::wooey/backend/utils.py::get_checksum ---
def get_checksum(path=None, buff=None, extra=None):
    import hashlib

    BLOCKSIZE = 65536
    hasher = hashlib.sha1()
    if extra:
        if isinstance(extra, (list, tuple)):
            for i in extra:
                hasher.update(str(i).encode("utf-8"))
        elif isinstance(extra, str):
            hasher.update(extra)
    if buff is not None:
        hasher.update(buff)
    elif path is not None:
        if isinstance(path, str):
            with open(path, "rb") as afile:
                buf = afile.read(BLOCKSIZE)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = afile.read(BLOCKSIZE)
        else:
            start = path.tell()
            path.seek(0)
            buf = path.read(BLOCKSIZE)
            while len(buf) > 0:
                hasher.update(buf)
                buf = path.read(BLOCKSIZE)
            path.seek(start)
    return hasher.hexdigest()

# --- from wooey__Wooey::wooey/backend/utils.py::get_current_scripts ---
def get_current_scripts():
    from ..models import ScriptVersion

    try:
        scripts = ScriptVersion.objects.count()
    except OperationalError:
        # database not initialized yet
        return

    # get the scripts with default version
    scripts = ScriptVersion.objects.select_related("script").filter(
        default_version=True, is_active=True
    )
    # scripts we need to figure out the default version for some reason
    non_default_scripts = ScriptVersion.objects.filter(
        default_version=False, is_active=True
    ).exclude(script__in=[i.script for i in scripts])
    script_versions = defaultdict(list)
    for sv in non_default_scripts:
        try:
            version_string = parse_version(str(sv.script_version))
        except Exception:
            sys.stderr.write(
                "Error converting script version:\n{}".format(traceback.format_exc())
            )
            version_string = sv.script_version
        script_versions[sv.script.script_name].append(
            (version_string, sv.script_iteration, sv)
        )
        [
            script_versions[i].sort(key=itemgetter(0, 1, 2), reverse=True)
            for i in script_versions
        ]
    scripts = [i.script for i in scripts]
    if script_versions:
        for script_version_info in script_versions.values():
            new_scripts = ScriptVersion.objects.select_related("script").filter(
                pk__in=[i[2].pk for i in script_version_info]
            )
            scripts.extend([i.script for i in new_scripts])
    return scripts
