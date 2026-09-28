# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg473::os.chdir+os.getcwd+pathlib.Path
# name: os_pathlib_primitive
# summary: Uses os.chdir, os.getcwd, pathlib.Path, subprocess.Popen across 2 repos
# anchor_symbols: ['os.chdir', 'os.getcwd', 'pathlib.Path', 'subprocess.Popen', 'subprocess.check_call']
# observed in 2 repos: ['iterative__mlem', 'nidhaloff__igel']...

# --- from iterative__mlem::tests/core/custom_requirements/test_shell_reqs.py::test_pipe_iter ---
def test_pipe_iter(tmpdir, script_code, executable):
    with subprocess.Popen(executable, stdin=subprocess.PIPE) as proc:
        for line in script_code.splitlines(keepends=True):
            proc.stdin.write(line.encode("utf8"))
        proc.communicate(b"exit()")
        assert proc.returncode == 0
    save("a", os.path.join(tmpdir, "data"))
    subprocess.check_call(["mlem", "apply", "model", "data"], cwd=tmpdir)

    meta = load_meta(os.path.join(tmpdir, "model"), force_type=MlemModel)
    assert len(meta.requirements.__root__) == 1
    assert meta.requirements.to_pip() == [f"numpy=={np.__version__}"]

# --- from nidhaloff__igel::igel/__main__.py::gui ---
def gui():
    """
    Launch the igel gui application.
    PS: you need to have nodejs on your machine
    """
    igel_ui_path = Path(os.getcwd()) / "igel-ui"
    if not Path.exists(igel_ui_path):
        subprocess.check_call(
            ["git"] + ["clone", "https://github.com/nidhaloff/igel-ui.git"]
        )
        logger.info(f"igel UI cloned successfully")

    os.chdir(igel_ui_path)
    logger.info(f"switching to -> {igel_ui_path}")
    logger.info(f"current dir: {os.getcwd()}")
    logger.info(f"make sure you have nodejs installed!!")

    subprocess.Popen(["node", "npm", "install", "open"], shell=True)
    subprocess.Popen(["node", "npm", "install electron", "open"], shell=True)
    logger.info("installing dependencies ...")
    logger.info(f"dependencies installed successfully")
    logger.info(f"node version:")
    subprocess.check_call("node -v", shell=True)
    logger.info(f"npm version:")
    subprocess.check_call("npm -v", shell=True)
    subprocess.check_call("npm i electron", shell=True)
    logger.info("running igel UI...")
    subprocess.check_call("npm start", shell=True)

# --- from nidhaloff__igel::deprecated/old_cli.py::CLI.gui ---
def gui(self, *args, **kwargs):
        igel_ui_path = Path(os.getcwd()) / "igel-ui"
        if not Path.exists(igel_ui_path):
            subprocess.check_call(
                ["git"] + ["clone", "https://github.com/nidhaloff/igel-ui.git"]
            )
            logger.info(f"igel UI cloned successfully")

        os.chdir(igel_ui_path)
        logger.info(f"switching to -> {igel_ui_path}")
        logger.info(f"current dir: {os.getcwd()}")
        logger.info(f"make sure you have nodejs installed!!")

        subprocess.Popen(["node", "npm", "install", "open"], shell=True)
        subprocess.Popen(
            ["node", "npm", "install electron", "open"], shell=True
        )
        logger.info("installing dependencies ...")
        logger.info(f"dependencies installed successfully")
        logger.info(f"node version:")
        subprocess.check_call("node -v", shell=True)
        logger.info(f"npm version:")
        subprocess.check_call("npm -v", shell=True)
        subprocess.check_call("npm i electron", shell=True)
        logger.info("running igel UI...")
        subprocess.check_call("npm start", shell=True)
