# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg852::jinja2.Environment+jinja2.FileSystemLoader+jinja2.select_autoescape
# name: jinja2_primitive
# summary: Uses jinja2.Environment, jinja2.FileSystemLoader, jinja2.select_autoescape across 2 repos
# anchor_symbols: ['jinja2.Environment', 'jinja2.FileSystemLoader', 'jinja2.select_autoescape']
# observed in 2 repos: ['capitalone__datacompy', 'iterative__mlem']...

# --- from iterative__mlem::mlem/utils/templates.py::TemplateModel.generate ---
def generate(self, **additional):
        j2 = Environment(
            loader=FileSystemLoader(self.templates_dir + [self.TEMPLATE_DIR]),
            undefined=StrictUndefined,
            autoescape=select_autoescape(),
        )
        template = j2.get_template(self.TEMPLATE_FILE)
        args = self.prepare_dict()
        args.update(additional)
        return template.render(**args)

# --- from capitalone__datacompy::datacompy/base.py::render ---
def render(template_name: str, **context: Any) -> str:
    """Render a template using Jinja2.

    Parameters
    ----------
    template_name : str
        The name of the template file to render. This can be:
        - A filename in the default templates directory (with or without .j2 extension)
        - A relative path from the default templates directory
        - An absolute path to a template file
    **context : dict
        The context variables to pass to the template

    Returns
    -------
    str
        The rendered template

    Raises
    ------
    FileNotFoundError
        If the template file cannot be found in any of the expected locations
    """
    template_dir, template_file = _resolve_template_path(template_name)

    # Create Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_file)
    return template.render(**context).strip()
