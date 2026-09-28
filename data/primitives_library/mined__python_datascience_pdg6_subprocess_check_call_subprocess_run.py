# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg6::subprocess.check_call+subprocess.run
# name: subprocess_primitive
# summary: Uses subprocess.check_call, subprocess.run across 3 repos
# anchor_symbols: ['subprocess.check_call', 'subprocess.run']
# observed in 3 repos: ['explosion__spaCy', 'microsoft__RD-Agent', 'ploomber__ploomber']...

# --- from ploomber__ploomber::tests/conftest.py::git_init ---
def git_init(commit=True):
    if Path("CHANGELOG.md").exists():
        raise ValueError("call git_init in a temporary directory")

    subprocess.run(["git", "init", "-b", "mybranch"])
    subprocess.check_call(["git", "config", "commit.gpgsign", "false"])
    subprocess.check_call(["git", "config", "user.email", "ci@ploomberio"])
    subprocess.check_call(["git", "config", "user.name", "Ploomber"])

    if commit:
        subprocess.run(["git", "add", "--all"])
        subprocess.run(["git", "commit", "-m", "some-commit-message"])

# --- from ploomber__ploomber::tests/env/test_sample_project_expanders.py::test_get_git ---
def test_get_git(tmp_directory, cleanup_env):
    Path("__init__.py").write_text('__version__ = "0.1dev0"')
    Path("env.yaml").write_text('_module: .\ngit: "{{git}}"')

    subprocess.run(["git", "init"])
    subprocess.check_call(["git", "config", "commit.gpgsign", "false"])
    subprocess.run(["git", "config", "user.email", "ci@ploomberio"])
    subprocess.run(["git", "config", "user.name", "Ploomber"])
    subprocess.run(["git", "add", "--all"])
    subprocess.run(["git", "commit", "-m", "first commit"])

    env = Env()
    assert env.git == "master"

# --- from explosion__spaCy::spacy/training/initialize.py::open_file ---
def open_file(loc: Union[str, Path]) -> IO:
    """Handle .gz, .tar.gz or unzipped files"""
    loc = ensure_path(loc)
    if tarfile.is_tarfile(str(loc)):
        return tarfile.open(str(loc), "r:gz")  # type: ignore[return-value]
    elif loc.parts[-1].endswith("gz"):
        return (line.decode("utf8") for line in gzip.open(str(loc), "r"))  # type: ignore[return-value]
    elif loc.parts[-1].endswith("zip"):
        zip_file = zipfile.ZipFile(str(loc))
        names = zip_file.namelist()
        file_ = zip_file.open(names[0])
        return (line.decode("utf8") for line in file_)  # type: ignore[return-value]
    else:
        return loc.open("r", encoding="utf8")

# --- from microsoft__RD-Agent::rdagent/utils/env.py::_prepare_conda_env ---
def _prepare_conda_env(env_name: str, requirements_file: Path, python_version: str = "3.10") -> None:
    """Prepare conda environment with dependencies from requirements.txt.

    Creates the env if it doesn't exist, then installs dependencies.
    Uses a process-level cache to avoid redundant preparation in the same run.

    Args:
        env_name: Conda environment name
        requirements_file: Path to requirements.txt file
        python_version: Python version for the environment
    """
    # 1. Create conda environment if not exists
    result = subprocess.run(f"conda env list | grep -q '^{env_name} '", shell=True)
    if result.returncode != 0:
        print(f"[yellow]Creating conda env '{env_name}' (Python {python_version})...[/yellow]")
        subprocess.check_call(f"conda create -y -n {env_name} python={python_version}", shell=True)
        subprocess.check_call(f"conda run -n {env_name} pip install --upgrade pip", shell=True)

    print(f"[yellow]Installing dependencies from {requirements_file.name}...[/yellow]")
    subprocess.check_call(f"conda run -n {env_name} pip install -r {requirements_file}", shell=True)
    print(f"[green]Conda env '{env_name}' ready[/green]")

    _CONDA_ENV_PREPARED.add(env_name)

# --- from microsoft__RD-Agent::rdagent/utils/env.py::QlibCondaEnv.prepare ---
def prepare(self) -> None:
        """Prepare the conda environment if not already created."""
        try:
            envs = subprocess.run("conda env list", capture_output=True, text=True, shell=True)
            if self.conf.conda_env_name not in envs.stdout:
                print(f"[yellow]Conda env '{self.conf.conda_env_name}' not found, creating...[/yellow]")
                subprocess.check_call(
                    f"conda create -y -n {self.conf.conda_env_name} python=3.10",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install --upgrade pip cython",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install git+https://github.com/microsoft/qlib.git@2fb9380b342556ddb50a4b24e4fe8655d548b2b8",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install catboost xgboost tables torch",
                    shell=True,
                )

        except Exception as e:
            print(f"[red]Failed to prepare conda env: {e}[/red]")
