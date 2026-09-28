# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg202::ast.parse+ast.walk
# name: ast_primitive
# summary: Uses ast.parse, ast.walk across 3 repos
# anchor_symbols: ['ast.parse', 'ast.walk']
# observed in 3 repos: ['OML-Team__open-metric-learning', 'microsoft__RD-Agent', 'ploomber__ploomber']...

# --- from OML-Team__open-metric-learning::tests/test_imports.py::find_imports ---
def find_imports(code: str) -> List[str]:
    code = ast.parse(code)
    imports = set()
    for node in ast.walk(code):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imports.add(node.module)
    return list(imports)

# --- from ploomber__ploomber::src/ploomber/static_analysis/python.py::PythonCallableExtractor.extract_upstream ---
def extract_upstream(self):
        """
        Extract keys requested to an upstream variable (e.g. upstream['key'])
        """
        module = ast.parse(self.code)
        return {
            get_value(node)
            for node in ast.walk(module)
            if isinstance(node, ast.Subscript)
            and get_key_value(node) == "upstream"
            and isinstance(get_constant(node), ast.Str)
        } or None

# --- from ploomber__ploomber::src/ploomber/sources/interact.py::CallableInteractiveDeveloper._reload_fn ---
def _reload_fn(self):
        # force to reload module to get the right information in case the
        # original source code was modified and the function is no longer in
        # the same position
        # NOTE: are there any  problems with this approach?
        # we could also read the file directly and use ast/parso to get the
        # function's information we need
        mod = importlib.reload(inspect.getmodule(self.fn))
        self.fn = getattr(mod, self.fn.__name__)

# --- from microsoft__RD-Agent::rdagent/components/coder/data_science/share/util.py::is_function_called ---
def is_function_called(source_code: str, func_name: str) -> bool:
    """
    Returns True if the function named `func_name` is called in `source_code`.
    """
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # For simple function calls like func()
            if isinstance(node.func, ast.Name) and node.func.id == func_name:
                return True

            # For calls like module.func()
            elif isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
                return True
    return False

# --- from microsoft__RD-Agent::rdagent/components/coder/data_science/share/util.py::extract_first_section_name_from_code ---
def extract_first_section_name_from_code(source_code):
    """
    Extract the first section name from the source code.
    """
    parsed = ast.parse(source_code)
    for node in ast.walk(parsed):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if getattr(call.func, "id", None) == "print" and call.args:
                arg0 = call.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    # Match "Section: ..." pattern
                    m = re.match(r"Section:\s*(.+)", arg0.value)
                    if m:
                        return m.group(1).strip()
    return None

# --- from OML-Team__open-metric-learning::tests/test_outdated_docs.py::check_docstrings_in_file ---
def check_docstrings_in_file(filename: Path) -> None:
    with open(filename, "r") as file:
        content = file.read()

    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)

            if docstring is not None:
                actual_args = set([arg.arg for arg in node.args.args]) - {"self", "cls"}
                docstring_args = set(parse_args_in_docstring(docstring))

                if docstring_args and (actual_args != docstring_args) and ("*_" not in docstring_args):
                    raise ValueError(
                        f"Incorrect docstring for {node.name} in {filename}."
                        f"Actual args are: {actual_args}\n"
                        f"Docstring args are: {docstring_args}"
                    )
