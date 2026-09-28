# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg256::examples.tutorial.jupyter.execution.test.utils._execute_notebook+examples.tutorial.jupyter.execution.test.utils._replace_str+nbformat.read
# name: examples_nbformat_primitive
# summary: Uses examples.tutorial.jupyter.execution.test.utils._execute_notebook, examples.tutorial.jupyter.execution.test.utils._replace_str, nbformat.read, nbformat.write across 2 repos
# anchor_symbols: ['examples.tutorial.jupyter.execution.test.utils._execute_notebook', 'examples.tutorial.jupyter.execution.test.utils._replace_str', 'nbformat.read', 'nbformat.write']
# observed in 2 repos: ['NannyML__nannyml', 'modin-project__modin']...

# --- from modin-project__modin::examples/tutorial/jupyter/execution/pandas_on_unidist/test/test_notebooks.py::test_exercise_1 ---
def test_exercise_1():
    modified_notebook_path = os.path.join(local_notebooks_dir, "exercise_1_test.ipynb")
    nb = nbformat.read(
        os.path.join(local_notebooks_dir, "exercise_1.ipynb"),
        as_version=nbformat.NO_CONVERT,
    )

    _replace_str(nb, "import pandas as pd", "import modin.pandas as pd")

    nbformat.write(nb, modified_notebook_path)
    _execute_notebook(modified_notebook_path)

# --- from modin-project__modin::examples/tutorial/jupyter/execution/pandas_on_ray/test/test_notebooks.py::test_exercise_1 ---
def test_exercise_1():
    modified_notebook_path = os.path.join(local_notebooks_dir, "exercise_1_test.ipynb")
    nb = nbformat.read(
        os.path.join(local_notebooks_dir, "exercise_1.ipynb"),
        as_version=nbformat.NO_CONVERT,
    )

    _replace_str(nb, "import pandas as pd", "import modin.pandas as pd")

    nbformat.write(nb, modified_notebook_path)
    _execute_notebook(modified_notebook_path)

# --- from NannyML__nannyml::docs/run_notebooks.py::run_notebook ---
def run_notebook(nb_path):
    nb_path = os.path.abspath(nb_path)
    assert path.endswith('.ipynb')
    nb = nbformat.read(path, as_version=4)
    try:
        cp.preprocess(nb, {'metadata': {'path': os.path.dirname(nb_path)}})
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(nb_path)}})
        postprocess(nb)
    except CellExecutionError:
        print(f'Error executing the notebook "{nb_path}".\n\n')
        raise
    finally:
        nb_out_path = out_dir / Path(nb_path).name
        with open(nb_out_path, mode='w', encoding='utf-8') as f:
            nbformat.write(nb, f)
