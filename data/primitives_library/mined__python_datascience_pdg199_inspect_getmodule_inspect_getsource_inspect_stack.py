# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg199::inspect.getmodule+inspect.getsource+inspect.stack
# name: inspect_pathlib_primitive
# summary: Uses inspect.getmodule, inspect.getsource, inspect.stack, pathlib.Path across 3 repos
# anchor_symbols: ['inspect.getmodule', 'inspect.getsource', 'inspect.stack', 'pathlib.Path']
# observed in 3 repos: ['danijar__handout', 'microsoft__RD-Agent', 'run-house__kubetorch']...

# --- from microsoft__RD-Agent::rdagent/utils/agent/tpl.py::get_caller_dir ---
def get_caller_dir(upshift: int = 0) -> Path:
    # Inspect the calling stack to get the caller's directory
    stack = inspect.stack()
    caller_frame = stack[1 + upshift]
    caller_module = inspect.getmodule(caller_frame[0])
    if caller_module and caller_module.__file__:
        caller_dir = Path(caller_module.__file__).parent
    else:
        caller_dir = DIRNAME
    return caller_dir

# --- from danijar__handout::handout/handout.py::Handout.__init__ ---
def __init__(self, directory, title='Handout'):
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(parents=True, exist_ok=True)
    self._title = title
    self._blocks = collections.defaultdict(list)
    self._pending = []
    # The logger is configured in handout/__init__.py to make it available
    # right after importing the handout package. This allows the user to change
    # the logging level, message format, etc.
    self._logger = logging.getLogger('handout')
    for info in inspect.stack():
      if info.filename == __file__:
        continue
      break
    module = inspect.getmodule(info.frame)
    self._source_name = info.filename
    self._source_text = inspect.getsource(module)
    self._num_images = 0
    self._num_videos = 0
    self._num_figures = 0

# --- from run-house__kubetorch::python_client/kubetorch/resources/callables/utils.py::prepare_notebook_fn ---
def prepare_notebook_fn(fn_pointers, name):
    """Handle a function defined in a notebook by writing it out to a dedicated .py file to be imported
    on the cluster."""
    module_path = Path.cwd() / (f"{name}_fn.py" if name else "sent_fn.py")
    logger.info(
        f"Function is defined in a notebook, writing it out to {str(module_path)} "
        f"to make it importable. Please make sure the function does not rely on any local variables, "
        f"including imports (which should be moved inside the function body). "
        f"This restriction does not apply to functions defined in normal Python files."
    )
    try:
        # Try to pull the frame variable for the function by name
        user_fn_name = fn_pointers[2]
        frame = inspect.stack()[2].frame
        user_fn = frame.f_globals.get(user_fn_name) or frame.f_locals.get(user_fn_name)
        source = inspect.getsource(user_fn).strip() if user_fn else None
        if source is None:
            raise NotebookError(
                f"Failed to load source code for function {user_fn_name}. "
                f"Please ensure the function is defined in the notebook and not relying on local variables."
            )
    except Exception as e:
        raise NotebookError(str(e))

    with module_path.open("w") as f:
        f.write(source)

    return fn_pointers[0], module_path.stem, fn_pointers[2]
