# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg443::jinja2.Environment+jinja2.PackageLoader+pathlib.Path
# name: jinja2_pathlib_primitive
# summary: Uses jinja2.Environment, jinja2.PackageLoader, pathlib.Path across 2 repos
# anchor_symbols: ['jinja2.Environment', 'jinja2.PackageLoader', 'pathlib.Path']
# observed in 2 repos: ['ploomber__ploomber', 'ploomber__sklearn-evaluation']...

# --- from ploomber__sklearn-evaluation::src/sklearn_evaluation/report/util.py::jinja_env ---
def jinja_env():
    env = Environment(
        loader=PackageLoader("sklearn_evaluation", "assets/report"),
    )
    return env

# --- from ploomber__ploomber::tests/placeholders/test_placeholder.py::_package_loader ---
def _package_loader(path_to_test_pkg):
    return Environment(
        loader=PackageLoader("test_pkg", "templates"), undefined=StrictUndefined
    )

# --- from ploomber__ploomber::src/ploomber/scaffold/scaffoldloader.py::ScaffoldLoader.__init__ ---
def __init__(self):
        self.env = Environment(
            loader=PackageLoader("ploomber", str(Path("resources", "ploomber_add"))),
            variable_start_string="[[",
            variable_end_string="]]",
            block_start_string="[%",
            block_end_string="%]",
            undefined=StrictUndefined,
        )
