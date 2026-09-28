# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg223::tarfile.open+tempfile.mkdtemp
# name: tarfile_tempfile_primitive
# summary: Uses tarfile.open, tempfile.mkdtemp across 2 repos
# anchor_symbols: ['tarfile.open', 'tempfile.mkdtemp']
# observed in 2 repos: ['InfuseAI__piperider', 'allenai__allennlp']...

# --- from allenai__allennlp::allennlp/models/archival.py::extracted_archive ---
def extracted_archive(resolved_archive_file, cleanup=True):
    tempdir = None
    try:
        tempdir = tempfile.mkdtemp()
        logger.info(f"extracting archive file {resolved_archive_file} to temp dir {tempdir}")
        with tarfile.open(resolved_archive_file, "r:gz") as archive:
            archive.extractall(tempdir)
        yield tempdir
    finally:
        if tempdir is not None and cleanup:
            logger.info(f"removing temporary unarchived model dir at {tempdir}")
            shutil.rmtree(tempdir, ignore_errors=True)

# --- from InfuseAI__piperider::piperider_cli/recipes/utils.py::AbstractRecipeUtils.git_archive ---
def git_archive(self, commit_or_branch):
        def untar(file_path, extract_dir):
            with tarfile.open(file_path, 'r') as tar:
                tar.extractall(extract_dir)

        tmpdirname = Path(tempfile.mkdtemp())
        tar = (tmpdirname / f'{commit_or_branch}.tar').as_posix()

        outs, errs, exit_code = self.execute_command_in_silent(
            rf'git archive --format=tar --output={tar} {commit_or_branch}'
        )
        print(rf'git archive --format=tar --output={tar} {commit_or_branch}')
        if exit_code != 0:
            raise RecipeException(errs)

        project_dir = (tmpdirname / commit_or_branch).as_posix()
        untar(tar, project_dir)

        return project_dir
