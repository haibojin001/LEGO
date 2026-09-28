# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg64::nbconvert.preprocessors.ExecutePreprocessor+nbformat.read+nbformat.write
# name: nbconvert_nbformat_primitive
# summary: Uses nbconvert.preprocessors.ExecutePreprocessor, nbformat.read, nbformat.write across 3 repos
# anchor_symbols: ['nbconvert.preprocessors.ExecutePreprocessor', 'nbformat.read', 'nbformat.write']
# observed in 3 repos: ['bodywork-ml__bodywork-core', 'piskvorky__gensim', 'recommenders-team__recommenders']...

# --- from piskvorky__gensim::docs/notebooks/test_notebooks.py::_notebook_run ---
def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    kernel_name = 'python%d' % sys.version_info[0]
    this_file_directory = os.path.dirname(__file__)
    errors = []
    with tempfile.NamedTemporaryFile(suffix=".ipynb", mode='wt') as fout:
        with smart_open(path, 'rb') as f:
            nb = nbformat.read(f, as_version=4)
            nb.metadata.get('kernelspec', {})['name'] = kernel_name
            ep = ExecutePreprocessor(kernel_name=kernel_name, timeout=10)

            try:
                ep.preprocess(nb, {'metadata': {'path': this_file_directory}})
            except CellExecutionError as e:
                if "SKIP" in e.traceback:
                    print(str(e.traceback).split("\n")[-2])
                else:
                    raise e
            except RuntimeError as e:
                print(e)

            finally:
                nbformat.write(nb, fout)

    return nb, errors

# --- from recommenders-team__recommenders::recommenders/utils/notebook_utils.py::execute_notebook ---
def execute_notebook(
    input_notebook, output_notebook, parameters={}, kernel_name="python3", timeout=5400
):
    """Execute a notebook while passing parameters to it.

    Note:
        Ensure your Jupyter Notebook is set up with parameters that can be
        modified and read. Use Markdown cells to specify parameters that need
        modification and code cells to set parameters that need to be read.

    Args:
        input_notebook (str): Path to the input notebook.
        output_notebook (str): Path to the output notebook
        parameters (dict): Dictionary of parameters to pass to the notebook.
        kernel_name (str): Kernel name.
        timeout (int): Timeout (in seconds) for each cell to execute.
    """

    # Load the Jupyter Notebook
    with open(input_notebook, "r") as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    # Search for and replace parameter values in code cells
    for cell in notebook_content.cells:
        if (
            "tags" in cell.metadata
            and "parameters" in cell.metadata["tags"]
            and cell.cell_type == "code"
        ):
            # Update the cell's source within notebook_content
            cell.source = _update_parameters(cell.source, parameters)

    # Create an execution preprocessor
    execute_preprocessor = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)

    # Execute the notebook
    executed_notebook, _ = execute_preprocessor.preprocess(
        notebook_content, {"metadata": {"path": "./"}}
    )

    # Save the executed notebook
    with open(output_notebook, "w", encoding="utf-8") as executed_notebook_file:
        nbformat.write(executed_notebook, executed_notebook_file)

# --- from bodywork-ml__bodywork-core::src/bodywork/stage_execution.py::run_stage ---
def run_stage(
    stage_name: str,
    repo_url: str,
    repo_branch: str = None,
    cloned_repo_dir: Path = DEFAULT_PROJECT_DIR,
    timeout: int = None,
) -> None:
    """Retrieve latest project code and run the chosen stage.

    :param stage_name: The Bodywork project stage name.
    :param repo_url: Git repository URL.
    :param repo_branch: The Git branch to download, defaults to None.
    :param cloned_repo_dir: The name of the directory int which the
        repository will be cloned, defaults to DEFAULT_PROJECT_DIR.
    :param timeout: The time to wait (in seconds) for the stage
        executable to complete, before terminating the main process.
        Defaults to None.
    :raises BodyworkStageFailure: If the executable script exits with
        a non-zero exit code (i.e. fails).
    """
    _log.info(
        f"Starting stage = {stage_name} from {repo_branch} branch of repo "
        f"at {repo_url}"
    )
    try:
        download_project_code_from_repo(repo_url, repo_branch, cloned_repo_dir)
        config_file_path = cloned_repo_dir / PROJECT_CONFIG_FILENAME
        project_config = BodyworkConfig(config_file_path)
        stage = project_config.stages[stage_name]
        environ["PYTHONPATH"] = str(cloned_repo_dir.absolute())
        if stage.requirements:
            _install_python_requirements(stage.requirements)
        executable_type = _infer_executable_type(stage.executable_module)
        if executable_type is ExecutableType.JUPYTER_NB:
            _log.info(f"Attempting to run notebook = {stage.executable_module_path}")
            notebook = nbformat.read(
                stage.executable_module_path, as_version=nbformat.NO_CONVERT
            )
            nb_runner = ExecutePreprocessor()
            nb_runner.preprocess(
                notebook,
                {"metadata": {"path": stage.executable_module_path.parent}},
            )
        else:
            _log.info(f"Attempting to run module = {stage.executable_module_path}")
            run(
                ["python", stage.executable_module, *stage.args],
                check=True,
                cwd=stage.executable_module_path.parent,
                encoding="utf-8",
                timeout=timeout,
            )
        _log.info(
            f"Successfully ran stage = {stage_name} from {repo_branch} branch of repo "
            f"at {repo_url}"
        )
    except TimeoutExpired:
        msg = f"Timeout exceeded when running {stage.executable_module}"
        raise BodyworkStageFailure(stage_name, msg)
    except Exception as e:
        stage_failure_exception = BodyworkStageFailure(stage_name, e.__repr__())
        _log.error(stage_failure_exception)
        raise stage_failure_exception from e
