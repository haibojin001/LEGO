# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg118::pathlib.Path+shutil.move+zipfile.ZipFile
# name: pathlib_shutil_primitive
# summary: Uses pathlib.Path, shutil.move, zipfile.ZipFile across 2 repos
# anchor_symbols: ['pathlib.Path', 'shutil.move', 'zipfile.ZipFile']
# observed in 2 repos: ['WecoAI__aideml', 'microsoft__nni']...

# --- from microsoft__nni::test/vso_tools/unpack_dependencies.py::main ---
def main() -> None:
    if sys.platform == 'win32':
        # Strangely user site is not enabled on Windows.
        prefix = Path(sys.prefix)
    else:
        prefix = Path(site.getuserbase())

    print('Extract Python packages to', prefix)
    print('All Python paths:')
    print('\n'.join(sys.path), flush=True)

    extract_all(ZipFile(sys.argv[1]))
    empty_dirs = json.loads(Path('directories.json').read_text())
    symlinks = json.loads(Path('symlinks.json').read_text())
    for dir_ in empty_dirs:
        Path(dir_).mkdir(parents=True, exist_ok=True)
    for link, target in symlinks.items():
        Path(link).symlink_to(target)  # hopefully nobody uses symlink on windows

    move_or_merge(Path('cache/python-dependencies'), prefix)
    shutil.move('cache/nni-manager-dependencies', 'ts/nni_manager/node_modules')
    shutil.move('cache/webui-dependencies', 'ts/webui/node_modules')

# --- from microsoft__nni::test/vso_tools/pack_dependencies.py::main ---
def main() -> None:
    cache = Path('cache')
    cache.mkdir()
    shutil.move('python-packages', 'cache/python-dependencies')
    shutil.move('ts/nni_manager/node_modules', 'cache/nni-manager-dependencies')
    shutil.move('ts/webui/node_modules', 'cache/webui-dependencies')

    archive = ZipFile(sys.argv[1], 'w', ZIP_DEFLATED, compresslevel=9)
    symlinks = {}
    empty_dirs = set()
    for file in sorted(cache.rglob('*')):
        if file.parent.parent == cache or file.parent.name == 'site-packages':
            print('Compress', file, flush=True)
        if file.is_symlink():
            symlinks[str(file)] = os.readlink(file)  # file.readlink() was added in Python 3.9
            continue
        if file.is_dir():
            empty_dirs.add(str(file))
        else:
            archive.write(file)
        empty_dirs.discard(str(file.parent))
    archive.writestr('symlinks.json', json.dumps(symlinks, indent=4))
    archive.writestr('directories.json', json.dumps(list(empty_dirs), indent=4))
    archive.close()

# --- from WecoAI__aideml::aide/utils/__init__.py::extract_archives ---
def extract_archives(path: Path):
    """
    unzips all .zip files within `path` and cleans up task dir

    [TODO] handle nested zips
    """
    for zip_f in path.rglob("*.zip"):
        f_out_dir = zip_f.with_suffix("")

        # special case: the intended output path already exists (maybe data has already been extracted by user)
        if f_out_dir.exists():
            logger.debug(
                f"Skipping {zip_f} as an item with the same name already exists."
            )
            # if it's a file, it's probably exactly the same as in the zip -> remove the zip
            # [TODO] maybe add an extra check to see if zip file content matches the colliding file
            if f_out_dir.is_file() and f_out_dir.suffix != "":
                zip_f.unlink()
            continue

        logger.debug(f"Extracting: {zip_f}")
        f_out_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_f, "r") as zip_ref:
            zip_ref.extractall(f_out_dir)

        # remove any unwanted files
        clean_up_dataset(f_out_dir)

        contents = list(f_out_dir.iterdir())

        # special case: the zip contains a single dir/file with the same name as the zip
        if len(contents) == 1 and contents[0].name == f_out_dir.name:
            sub_item = contents[0]
            # if it's a dir, move its contents to the parent and remove it
            if sub_item.is_dir():
                logger.debug(f"Special handling (child is dir) enabled for: {zip_f}")
                for f in sub_item.rglob("*"):
                    shutil.move(f, f_out_dir)
                sub_item.rmdir()
            # if it's a file, rename it to the parent and remove the parent
            elif sub_item.is_file():
                logger.debug(f"Special handling (child is file) enabled for: {zip_f}")
                sub_item_tmp = sub_item.rename(f_out_dir.with_suffix(".__tmp_rename"))
                f_out_dir.rmdir()
                sub_item_tmp.rename(f_out_dir)

        zip_f.unlink()
