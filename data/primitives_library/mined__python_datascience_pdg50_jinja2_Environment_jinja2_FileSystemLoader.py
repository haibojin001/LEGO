# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg50::jinja2.Environment+jinja2.FileSystemLoader
# name: jinja2_primitive
# summary: Uses jinja2.Environment, jinja2.FileSystemLoader across 9 repos
# anchor_symbols: ['jinja2.Environment', 'jinja2.FileSystemLoader']
# observed in 9 repos: ['InfuseAI__piperider', 'hi-primus__optimus', 'insitro__redun', 'ploomber__ploomber', 'run-house__kubetorch']...

# --- from insitro__redun::redun/console/widgets.py::PageLabel.render ---
def render(self) -> RenderableType:
        return Text(f" Page {self.page} ", style=Style(color="black", bgcolor="green"))

# --- from ploomber__ploomber::tests/placeholders/test_placeholder.py::_filesystem_loader ---
def _filesystem_loader(path_to_test_pkg):
    return Environment(
        loader=FileSystemLoader(str(Path(path_to_test_pkg, "templates"))),
        undefined=StrictUndefined,
    )

# --- from ploomber__ploomber::tests/placeholders/test_placeholder.py::test_strict_templates_raises_error_if_not_strictundefined ---
def test_strict_templates_raises_error_if_not_strictundefined(path_to_assets):
    path = str(path_to_assets / "templates")
    env = Environment(loader=FileSystemLoader(path))

    with pytest.raises(ValueError):
        Placeholder(env.get_template("template.sql"))

# --- from InfuseAI__piperider::piperider_cli/__init__.py::load_jinja_template ---
def load_jinja_template(path: str):
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(searchpath=os.path.dirname(path)))
    _init_jinja_env(env)
    template = env.get_template(os.path.basename(path))

    return template

# --- from sb-ai-lab__LightAutoML::lightautoml/report/report_deco.py::ReportDecoUplift._generate_uplift_subsection ---
def _generate_uplift_subsection(self):
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        uplift_subsection = env.get_template(self._uplift_subsection_path).render(self._uplift_content)
        self._uplift_results.append(uplift_subsection)

# --- from sberbank-ai-lab__LightAutoML::lightautoml/report/report_deco.py::ReportDecoUplift._generate_uplift_subsection ---
def _generate_uplift_subsection(self):
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        uplift_subsection = env.get_template(self._uplift_subsection_path).render(self._uplift_content)
        self._uplift_results.append(uplift_subsection)

# --- from sb-ai-lab__LightAutoML::lightautoml/report/report_deco.py::ReportDeco._generate_inference_section ---
def _generate_inference_section(self):
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        inference_section = env.get_template(self._inference_section_path[self.task]).render(self._inference_content)
        self._model_results.append(inference_section)

# --- from sberbank-ai-lab__LightAutoML::lightautoml/report/report_deco.py::ReportDeco._generate_inference_section ---
def _generate_inference_section(self):
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        inference_section = env.get_template(self._inference_section_path[self.task]).render(self._inference_content)
        self._model_results.append(inference_section)

# --- from sinaptik-ai__pandas-ai::tests/unit_tests/skills/test_shared_template.py::TestSharedTemplate.get_template_environment ---
def get_template_environment(self):
        """Get the Jinja2 template environment."""
        current_dir = Path(__file__).parent
        template_path = (
            current_dir.parent.parent.parent
            / "pandasai"
            / "core"
            / "prompts"
            / "templates"
        )
        return Environment(loader=FileSystemLoader(str(template_path)))

# --- from run-house__kubetorch::python_client/kubetorch/serving/utils.py::_get_rendered_template ---
def _get_rendered_template(template_file: str, template_dir: str, **template_vars) -> str:
    """Helper function to set up and render a template."""
    template_loader = jinja2.FileSystemLoader(searchpath=template_dir)
    template_env = jinja2.Environment(
        loader=template_loader,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        enable_async=False,
        autoescape=False,
    )
    template = template_env.get_template(template_file)
    return template.render(**template_vars)

# --- from run-house__kubetorch::services/kubetorch_controller/core/utils.py::_get_rendered_template ---
def _get_rendered_template(
    template_file: str, template_dir: str, **template_vars
) -> str:
    """Helper function to set up and render a template."""
    template_loader = jinja2.FileSystemLoader(searchpath=template_dir)
    template_env = jinja2.Environment(
        loader=template_loader,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        enable_async=False,
        autoescape=False,
    )
    template = template_env.get_template(template_file)
    return template.render(**template_vars)

# --- from sinaptik-ai__pandas-ai::pandasai/core/prompts/base.py::BasePrompt.__init__ ---
def __init__(self, **kwargs):
        """Initialize the prompt."""
        self.props = kwargs

        if self.template:
            env = Environment()
            self.prompt = env.from_string(self.template)
        elif self.template_path:
            # find path to template file
            current_dir_path = Path(__file__).parent
            path_to_template = os.path.join(current_dir_path, "templates")
            env = Environment(loader=FileSystemLoader(path_to_template))
            self.prompt = env.get_template(self.template_path)

        self._resolved_prompt = None
