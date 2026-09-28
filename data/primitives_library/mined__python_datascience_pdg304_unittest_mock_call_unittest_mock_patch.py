# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg304::unittest.mock.call+unittest.mock.patch
# name: unittest_primitive
# summary: Uses unittest.mock.call, unittest.mock.patch across 10 repos
# anchor_symbols: ['unittest.mock.call', 'unittest.mock.patch']
# observed in 10 repos: ['AgnostiqHQ__covalent', 'WecoAI__aideml', 'freud14__poutyne', 'meteostat__meteostat', 'okfn-brasil__serenata-de-amor']...

# --- from ploomber__ploomber::tests/products/test_file.py::test_download_triggers_client_download ---
def test_download_triggers_client_download(tmp_directory):
    client = Mock()
    product = File("file.txt", client=client)

    product.download()

    client.download.assert_has_calls([call("file.txt"), call(".file.txt.metadata")])

# --- from pykale__pykale::tests/utils/test_download.py::test_retry_download_retries_on_failure ---
def test_retry_download_retries_on_failure():
    fn = MagicMock(side_effect=[RuntimeError("timeout"), RuntimeError("timeout"), None])
    with patch("kale.utils.download.time.sleep") as mock_sleep:
        _retry_download(fn, retries=3, backoff=2)
    assert fn.call_count == 3
    mock_sleep.assert_has_calls([call(1), call(2)])

# --- from ploomber__ploomber::tests/products/test_file.py::test_product_upload_uploads_metadata_and_product ---
def test_product_upload_uploads_metadata_and_product(tmp_directory):
    Path("file.txt").touch()
    Path(".file.txt.metadata").touch()
    client = Mock()
    product = File("file.txt", client=client)

    product.upload()

    client.upload.assert_has_calls(
        [call(Path(".file.txt.metadata")), call(Path("file.txt"))]
    )

# --- from ottogroup__palladium::palladium/tests/test_server.py::TestFitFunctional.test_pass_args ---
def test_pass_args(self, fit, flask_app, args, args_expected):
        with patch('palladium.server.fit_base') as fit_base:
            fit_base.__name__ = 'mock'
            with flask_app.test_request_context(method='POST', data=args):
                fit()
            sleep(0.02)
        assert fit_base.call_args == call(**args_expected)

# --- from ottogroup__palladium::palladium/tests/test_eval.py::TestList.test ---
def test(self, list):
        model_persister = Mock()
        model_persister.list_models.return_value = [{1: 2}]
        model_persister.list_properties.return_value = {5: 6}
        with patch('palladium.eval.pprint') as pprint:
            list(model_persister)
        assert pprint.mock_calls[0] == call([{1: 2}])
        assert pprint.mock_calls[1] == call({5: 6})

# --- from freud14__poutyne::tests/framework/callbacks/test_wandb_logger.py::WandBLoggerTest.test_log_config ---
def test_log_config(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            wandb_patch.init = self.initialize_experiment
            wandb_patch.run = None
            logger = WandBLogger(name=self.a_name)
            logger.log_config_params(self.a_config_params)

            create_experiment_call = [call(self.a_config_params)]
            logger.run.config.update.assert_has_calls(create_experiment_call)

# --- from freud14__poutyne::tests/framework/callbacks/test_periodic.py::PeriodicSaveLambdaTest.test_given_a_fun_save_file_use_function ---
def test_given_a_fun_save_file_use_function(self):
        a_function_mock = MagicMock()

        periodic_save_lambda = PeriodicSaveLambda(filename=self.a_filename, func=a_function_mock)
        a_file_descriptor_mock = MagicMock()
        a_epoch_number = 1
        a_log = {}

        periodic_save_lambda.save_file(a_file_descriptor_mock, a_epoch_number, a_log)
        a_function_mock.assert_has_calls([call(a_file_descriptor_mock, a_epoch_number, a_log)])

# --- from WecoAI__aideml::aide/utils/__init__.py::copytree ---
def copytree(src: Path, dst: Path, use_symlinks=True):
    """
    Copy contents of `src` to `dst`. Unlike shutil.copytree, the dst dir can exist and will be merged.
    If src is a file, only that file will be copied. Optionally uses symlinks instead of copying.

    Args:
        src (Path): source directory
        dst (Path): destination directory
    """
    assert dst.is_dir()

    if src.is_file():
        dest_f = dst / src.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            (dest_f).symlink_to(src)
        else:
            shutil.copyfile(src, dest_f)
        return

    for f in src.iterdir():
        dest_f = dst / f.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            (dest_f).symlink_to(f)
        elif f.is_dir():
            shutil.copytree(f, dest_f)
        else:
            shutil.copyfile(f, dest_f)

# --- from okfn-brasil__serenata-de-amor::rosie/rosie/core/tests/test_core_init.py::TestCore.test_call ---
def test_call(self, mocked_predict, mocked_load):
        mocked_load.return_value = 'model'
        settings = MagicMock()
        settings.UNIQUE_IDS = ['number']
        settings.CLASSIFIERS = {'answer': 42, 'another': 13}
        core = Core(settings, self.adapter)
        core.suspicions = MagicMock()
        core()

        # assert load and predict was called for each classifier
        mocked_load.assert_has_calls((call(42), call(13)), any_order=True)
        mocked_predict.assert_has_calls((
            call('model', 'answer'),
            call('model', 'another')
        ), any_order=True)

        # assert suspicions.xz was created
        expected_path = os.path.join('tmp', 'test', 'suspicions.xz')
        core.suspicions.to_csv.assert_called_once_with(
            expected_path,
            compression='xz',
            encoding='utf-8',
            index=False
        )

# --- from AgnostiqHQ__covalent::tests/covalent_dispatcher_tests/_cli/service_test.py::test_terminate_child_processes ---
def test_terminate_child_processes(mocker):
    from covalent_dispatcher._cli.service import _terminate_child_processes

    psutil_process_mock = mocker.patch(
        "covalent_dispatcher._cli.service.psutil.Process", return_value=MagicMock()
    )
    children_mock = MagicMock()
    psutil_process_mock.return_value.children.return_value = [children_mock]
    wait_procs_mock = mocker.patch("covalent_dispatcher._cli.service.psutil.wait_procs")

    _terminate_child_processes(1)

    psutil_process_mock.assert_has_calls(
        [
            call(1),
            call(children_mock.pid),
        ]
    )
    psutil_process_mock.return_value.children.assert_has_calls(
        [
            call(),
            call(recursive=True),
        ]
    )
    children_mock.send_signal.assert_called_once_with(signal.SIGINT)
    children_mock.kill.assert_called_once_with()
    wait_procs_mock.assert_called_once_with([children_mock])
    children_mock.wait.assert_called_once_with()

# --- from quixio__quix-streams::tests/test_quixstreams/test_internal_producer.py::TestTransactionalInternalProducer.test_retriable_op_error ---
def test_retriable_op_error(self):
        """
        Some specific failure cases from sending offsets or committing a transaction
        are retriable.
        """

        class MockKafkaError(Exception):
            def retriable(self):
                return True

        call_args = [["my", "offsets"], "consumer_metadata", 1]
        error = ConfluentKafkaException(MockKafkaError())

        mock_producer = create_autospec(Producer)
        mock_producer.send_offsets_to_transaction.__name__ = "send_offsets"
        mock_producer.commit_transaction.__name__ = "commit"
        mock_producer.send_offsets_to_transaction.side_effect = [error, None]
        with patch(
            "quixstreams.internal_producer.Producer",
            return_value=mock_producer,
        ):
            producer = InternalProducer(broker_address="xyz", transactional=True)
            producer.commit_transaction(*call_args)

        mock_producer.send_offsets_to_transaction.assert_has_calls(
            [call(*call_args)] * 2
        )
        mock_producer.commit_transaction.assert_called_once()

# --- from meteostat__meteostat::tests/unit/test_network.py::TestNetworkServiceRetry.test_exponential_backoff_between_retries ---
def test_exponential_backoff_between_retries(self):
        """NetworkService should use exponential backoff between retry attempts"""
        service = NetworkService()
        original_retries = config.network_max_retries

        try:
            config.network_max_retries = 3

            with patch("requests.get") as mock_get, patch("time.sleep") as mock_sleep:
                error_response = MagicMock()
                error_response.status_code = 500

                success_response = MagicMock()
                success_response.status_code = 200

                # Fail 3 times, succeed on 4th
                mock_get.side_effect = [
                    error_response,
                    error_response,
                    error_response,
                    success_response,
                ]

                service.get("https://example.com")

                # Backoff: 2^0=1, 2^1=2, 2^2=4
                assert mock_sleep.call_count == 3
                mock_sleep.assert_has_calls([call(1), call(2), call(4)])
        finally:
            config.network_max_retries = original_retries
