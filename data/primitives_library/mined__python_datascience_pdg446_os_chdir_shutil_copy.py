# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg446::os.chdir+shutil.copy
# name: os_shutil_primitive
# summary: Uses os.chdir, shutil.copy across 2 repos
# anchor_symbols: ['os.chdir', 'shutil.copy']
# observed in 2 repos: ['ploomber__ploomber', 'pydoit__doit']...

# --- from ploomber__ploomber::tests/spec/test_dagspec.py::test_load_spec_with_custom_name_in_packaged_structure ---
def test_load_spec_with_custom_name_in_packaged_structure(backup_test_pkg):
    os.chdir(Path(backup_test_pkg).parents[1])

    path = Path("src", "test_pkg")
    shutil.copy(path / "pipeline.yaml", path / "pipeline.serve.yaml")

    spec = DAGSpec.find(name="pipeline.serve.yaml")
    assert spec.path == (path / "pipeline.serve.yaml").resolve()

# --- from pydoit__doit::tests/test_doit_cmd.py::TestConfig.test_find_pyproject_toml_config ---
def test_find_pyproject_toml_config(self):
        config_filename = self._test_path('pyproject.toml')
        api_config = {'GLOBAL': {'opty': '10', 'optz': '10'}}

        with tempfile.TemporaryDirectory() as td:
            shutil.copy(config_filename, os.path.join(td, 'pyproject.toml'))
            old_cwd = os.getcwd()
            try:
                os.chdir(td)
                main = doit_cmd.DoitMain(extra_config=api_config)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(1, len(main.config['COMMAND']))
            self.assertIn('bar', main.get_cmds())
            self.assertEqual({'optx': '2', 'opty': '3', 'optz': '10'},
                             main.config['GLOBAL'])
