# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg48::unittest.mock.Mock
# name: unittest_primitive
# summary: Uses unittest.mock.Mock across 15 repos
# anchor_symbols: ['unittest.mock.Mock']
# observed in 15 repos: ['AgnostiqHQ__covalent', 'SforAiDl__KD_Lib', 'cleanlab__cleanlab', 'datafold__data-diff', 'deepchecks__deepchecks']...

# --- from deepchecks__deepchecks::deepchecks/tabular/dataset.py::Dataset._ipython_display_ ---
def _ipython_display_(self):
        display_html(HTML(self.__repr__(fmt='html')))

# --- from underneathall__pinferencia::examples/pytorch/mnist/func_app.py::preprocessing ---
def preprocessing(img_str):
    image = Image.open(BytesIO(base64.b64decode(img_str)))
    tensor = transform(image)
    return torch.stack([tensor]).to(device)

# --- from ploomber__ploomber::tests/cli/test_examples.py::test_cli ---
def test_cli(monkeypatch, argv, kwargs):
    mock = Mock()
    monkeypatch.setattr(examples, "main", mock)

    CliRunner().invoke(cli.cli, argv, catch_exceptions=False)

    mock.assert_called_once_with(**kwargs)

# --- from underneathall__pinferencia::examples/pytorch/mnist/path_app.py::MNISTHandler.predict ---
def predict(self, data):
        image = Image.open(BytesIO(base64.b64decode(data)))
        tensor = self.transform(image)
        input_data = torch.stack([tensor]).to(self.device)
        return self.model(input_data).argmax(1).tolist()[0]

# --- from AgnostiqHQ__covalent::tests/covalent_dispatcher_tests/_cli/groups/db_test.py::test_migration_success ---
def test_migration_success(mocker):
    runner = CliRunner()
    db_mock = Mock()
    mocker.patch.object(DataStore, "factory", lambda: db_mock)
    runner.invoke(migrate, catch_exceptions=False)
    db_mock.run_migrations.assert_called_once()

# --- from deepchecks__deepchecks::tests/base/display_test.py::test_check_failure_display_with_enabled_widgets ---
def test_check_failure_display_with_enabled_widgets():
    # Arrange
    failure = CheckFailure(DummyCheck(), Exception('error message'))
    # Assert
    with patch('deepchecks.core.display.display', Mock(return_value=True)) as mock:
        w = failure.display_check(as_widget=True)
        mock.assert_called_once()

# --- from ottogroup__palladium::palladium/tests/test_R.py::ObjectMixin ---
def ObjectMixin(monkeypatch):
    from palladium.R import ObjectMixin
    r_dict = {}
    r = MagicMock()
    r.__getitem__.side_effect = r_dict.__getitem__
    r.__setitem__.side_effect = r_dict.__setitem__
    r['myfunc'] = Mock()
    r['predict'] = Mock()
    monkeypatch.setattr(ObjectMixin, 'r', r)
    return ObjectMixin

# --- from ottogroup__palladium::palladium/tests/test_julia.py::TestClassificationModel.test_score ---
def test_score(self, model):
        X, y = Mock(), Mock()
        with patch('palladium.julia.accuracy_score') as accuracy_score:
            with patch('palladium.julia.AbstractModel.predict') as predict:
                model.score(X, y)
        accuracy_score.assert_called_with(predict.return_value, y)
        predict.assert_called_with(X)

# --- from AgnostiqHQ__covalent::tests/covalent_dispatcher_tests/_cli/groups/db_test.py::test_migration_with_warning ---
def test_migration_with_warning(mocker):
    """Test the start CLI command invoking migration warning"""
    runner = CliRunner()
    db_mock = Mock()
    db_mock.run_migrations.side_effect = Exception("migration issue")
    mocker.patch.object(DataStore, "factory", lambda: db_mock)
    res = runner.invoke(migrate, catch_exceptions=False)
    assert MIGRATION_WARNING_MSG in res.output

# --- from microsoft__RD-Agent::rdagent/log/ui/llm_st.py::get_folders_sorted ---
def get_folders_sorted(log_path):
    """缓存并返回排序后的文件夹列表，并加入进度打印"""
    with st.spinner("正在加载文件夹列表..."):
        folders = sorted(
            (folder for folder in log_path.iterdir() if folder.is_dir() and list(folder.iterdir())),
            key=lambda folder: folder.stat().st_mtime,
            reverse=True,
        )
        st.write(f"找到 {len(folders)} 个文件夹")
    return [folder.name for folder in folders]

# --- from sb-ai-lab__LightAutoML::lightautoml/addons/interpretation/l2x.py::L2XExplanation.visualize_in_notebook ---
def visualize_in_notebook(self):
        """
        Visualization of important tokens in notebook.
        """
        from IPython.display import HTML
        from IPython.display import display_html

        token_weights = [(escape(x), y) for x, y in zip(self.tokens, self.mask)]
        html_code = draw_html(token_weights, self.task_name, self._hightliting_color, grad_line=False)
        display_html(HTML(html_code))

# --- from sberbank-ai-lab__LightAutoML::lightautoml/addons/interpretation/l2x.py::L2XExplanation.visualize_in_notebook ---
def visualize_in_notebook(self):
        """
        Visualization of important tokens in notebook.
        """
        from IPython.display import HTML
        from IPython.display import display_html

        token_weights = [(escape(x), y) for x, y in zip(self.tokens, self.mask)]
        html_code = draw_html(token_weights, self.task_name, self._hightliting_color, grad_line=False)
        display_html(HTML(html_code))
