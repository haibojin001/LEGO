# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg338::sys.exc_info+traceback.extract_tb
# name: sys_traceback_primitive
# summary: Uses sys.exc_info, traceback.extract_tb across 3 repos
# anchor_symbols: ['sys.exc_info', 'traceback.extract_tb']
# observed in 3 repos: ['WecoAI__aideml', 'airbnb__knowledge-repo', 'bodywork-ml__bodywork-core']...

# --- from airbnb__knowledge-repo::knowledge_repo/app/models.py::ErrorLog.from_exception ---
def from_exception(cls, e):
        tb = sys.exc_info()[-1]
        filename, linenumber, function, code = traceback.extract_tb(
            sys.exc_info()[-1])[-1]
        filename = os.path.relpath(
            filename, os.path.join(os.path.dirname(__file__), '..'))
        e_args = '; '.join(str(a) for a in e.args)
        return ErrorLog(
            function=function,
            location=f'{filename}:{linenumber}',
            message=f'{e.__class__.__name__}: {e_args}',
            traceback='\n'.join(traceback.format_tb(tb))
        )

# --- from bodywork-ml__bodywork-core::src/bodywork/cli/cli.py::handle_k8s_exceptions.wrapper ---
def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            func(*args, **kwargs)
        except kubernetes.client.rest.ApiException:
            e_type, e_value, e_tb = sys.exc_info()
            exception_origin = traceback.extract_tb(e_tb)[2].name
            print_warn(
                f"Kubernetes API error returned when called from {exception_origin} "
                f"within cli.{func.__name__}: {api_exception_msg(e_value)}"
            )
        except urllib3.exceptions.MaxRetryError:
            e_type, e_value, e_tb = sys.exc_info()
            exception_origin = traceback.extract_tb(e_tb)[2].name
            print_warn(
                f"Failed to connect to the Kubernetes API when called from "
                f"{exception_origin} within cli.{func.__name__}: {e_value}"
            )
        except kubernetes.config.ConfigException as e:
            print_warn(
                f"Cannot load authentication credentials from kubeconfig file when "
                f"calling cli.{func.__name__}: {e}"
            )

# --- from WecoAI__aideml::aide/interpreter.py::exception_summary ---
def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    """Generates a string that summarizes an exception and its stack trace (either in standard python repl or in IPython format)."""
    if format_tb_ipython:
        import IPython.core.ultratb

        # tb_offset = 1 to skip parts of the stack trace in weflow code
        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # skip parts of stack trace in weflow code
        tb_str = "".join(
            [
                line
                for line in tb_lines
                if "aide/" not in line and "importlib" not in line
            ]
        )
        # tb_str = "".join([l for l in tb_lines])

    # replace whole path to file with just filename (to remove agent workspace dir)
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack

# --- from bodywork-ml__bodywork-core::src/bodywork/cli/cli.py::handle_k8s_exceptions ---
def handle_k8s_exceptions(func: Callable[..., None]) -> Callable[..., None]:
    """Decorator for handling k8s API exceptions on the CLI.

    :param func: The inner function to wrap with k8s exception handling.
    :return: The original function wrapped by a function that handles
        k8s API exceptions.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            func(*args, **kwargs)
        except kubernetes.client.rest.ApiException:
            e_type, e_value, e_tb = sys.exc_info()
            exception_origin = traceback.extract_tb(e_tb)[2].name
            print_warn(
                f"Kubernetes API error returned when called from {exception_origin} "
                f"within cli.{func.__name__}: {api_exception_msg(e_value)}"
            )
        except urllib3.exceptions.MaxRetryError:
            e_type, e_value, e_tb = sys.exc_info()
            exception_origin = traceback.extract_tb(e_tb)[2].name
            print_warn(
                f"Failed to connect to the Kubernetes API when called from "
                f"{exception_origin} within cli.{func.__name__}: {e_value}"
            )
        except kubernetes.config.ConfigException as e:
            print_warn(
                f"Cannot load authentication credentials from kubeconfig file when "
                f"calling cli.{func.__name__}: {e}"
            )

    return wrapper
