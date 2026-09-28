# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg453::ast.literal_eval+parso.parse
# name: ast_parso_primitive
# summary: Uses ast.literal_eval, parso.parse across 2 repos
# anchor_symbols: ['ast.literal_eval', 'parso.parse']
# observed in 2 repos: ['ploomber__ploomber', 'ploomber__sklearn-evaluation']...

# --- from ploomber__sklearn-evaluation::src/sklearn_evaluation/nb/NotebookIntrospector.py::parse_injected_parameters_cell ---
def parse_injected_parameters_cell(cells):
    # this is a very simple implementation, for a more robust solution
    # re-implement with ast or parso
    cell = find_cell_with_tag(cells, tag="injected-parameters")

    if not cell:
        return dict()

    children = parso.parse(cell["source"]).children

    statements = [
        _process_stmt(c) for c in children if c.type in {"simple_stmt", "expr_stmt"}
    ]

    return {
        stmt.children[0].value: ast.literal_eval(stmt.children[2].get_code().strip())
        for stmt in statements
        if stmt is not None
    }

# --- from ploomber__ploomber::src/ploomber/static_analysis/pyflakes.py::_get_defined_variables ---
def _get_defined_variables(params_source):
    """
    Return the variables defined in a given source. If a name is defined more
    than once, it uses the last definition. Ignores anything other than
    variable assignments (e.g., function definitions, exceptions)
    """
    used_names = parso.parse(params_source).get_used_names()

    def _get_value(value):
        possible_literal = value.get_definition().children[-1].get_code().strip()

        try:
            # NOTE: this cannot parse dict(a=1, b=2)
            return ast.literal_eval(possible_literal)
        except ValueError:
            return None

    return {
        key: _get_value(value[-1])
        for key, value in used_names.items()
        if value[-1].is_definition() and value[-1].get_definition().type == "expr_stmt"
    }
