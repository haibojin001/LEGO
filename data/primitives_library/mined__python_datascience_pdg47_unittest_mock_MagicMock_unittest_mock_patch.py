# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg47::unittest.mock.MagicMock+unittest.mock.patch
# name: unittest_primitive
# summary: Uses unittest.mock.MagicMock, unittest.mock.patch across 12 repos
# anchor_symbols: ['unittest.mock.MagicMock', 'unittest.mock.patch']
# observed in 12 repos: ['NorskRegnesentral__skweak', 'alteryx__evalml', 'awslabs__gluonts', 'cleanlab__cleanlab', 'meteostat__meteostat']...

# --- from sinaptik-ai__pandas-ai::tests/unit_tests/conftest.py::mock_json_load ---
def mock_json_load():
    mock = MagicMock()

    with patch("json.load", mock):
        yield mock

# --- from awslabs__gluonts::docs/md2ipynb.py::black_cells.apply_black ---
def apply_black(match):
        code = match.group(1)

        formatted = black.format_str(code, mode=black.Mode())

        return "\n".join(["```", formatted.rstrip(), "```"])

# --- from alteryx__evalml::evalml/pipelines/components/transformers/preprocessing/lsa.py::LSA.__init__ ---
def __init__(self, random_seed=0, **kwargs):
        self._lsa_pipeline = make_pipeline(
            TfidfVectorizer(),
            TruncatedSVD(random_state=random_seed),
        )
        self._provenance = {}
        super().__init__(random_seed=random_seed, **kwargs)

# --- from sinaptik-ai__pandas-ai::tests/unit_tests/test_pandasai_init.py::TestPandasAIInit.test_chat_sandbox_passed_to_agent ---
def test_chat_sandbox_passed_to_agent(self, sample_df):
        with patch("pandasai.Agent") as MockAgent:
            sandbox = MagicMock()
            pandasai.chat("Test query", sample_df, sandbox=sandbox)
            MockAgent.assert_called_once_with([sample_df], sandbox=sandbox)

# --- from pykale__pykale::tests/utils/test_download.py::test_retry_download_raises_after_all_retries ---
def test_retry_download_raises_after_all_retries():
    fn = MagicMock(side_effect=RuntimeError("timeout"))
    with patch("kale.utils.download.time.sleep"):
        with pytest.raises(RuntimeError, match="timeout"):
            _retry_download(fn, retries=3, backoff=2)
    assert fn.call_count == 3

# --- from ottogroup__palladium::palladium/tests/test_persistence.py::TestFileAttachments.test_attachment_not_in_pickle ---
def test_attachment_not_in_pickle(self, persister, tmpdir):
        # Attachment data is not pickled as part of the model:
        with open(tmpdir + '/model-1.pkl.gz', 'rb') as fh:
            with gzip.open(fh, 'rb') as f:
                model1 = pickle.load(f)
                assert 'attachments/myatt.txt' not in annotate(model1)

# --- from ottogroup__palladium::palladium/tests/test_persistence.py::TestS3.s3_cls_with_bucket ---
def s3_cls_with_bucket(self, bucket_name, s3_cls, bucket_location):
        import moto, boto3
        with moto.mock_s3():
            conn = boto3.resource('s3')
            location = {'LocationConstraint': bucket_location}
            conn.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)

            yield s3_cls

# --- from alteryx__evalml::evalml/automl/engine/dask_engine.py::DaskEngine.__init__ ---
def __init__(self, cluster=None):
        if cluster is not None and not isinstance(cluster, (LocalCluster)):
            raise TypeError(
                f"Expected dask.distributed.Client, received {type(cluster)}",
            )
        elif cluster is None:
            cluster = LocalCluster(processes=False)
        self.cluster = cluster
        self.client = Client(self.cluster)
        self._data_futures_cache = {}

# --- from awslabs__gluonts::docs/md2ipynb.py::black_cells ---
def black_cells(text):
    CODE_RE = r"```py(?:thon)?\s*\n(.*?)```"

    text = re.sub(r"^%", r"# %-% #", text, flags=re.M)

    def apply_black(match):
        code = match.group(1)

        formatted = black.format_str(code, mode=black.Mode())

        return "\n".join(["```", formatted.rstrip(), "```"])

    formatted = re.sub(CODE_RE, apply_black, text, flags=re.S)
    return re.sub(r"^# %-% #", r"%", formatted, flags=re.M)

# --- from meteostat__meteostat::tests/unit/test_network.py::TestNetworkServiceRetry.test_successful_request_on_first_attempt ---
def test_successful_request_on_first_attempt(self):
        """NetworkService should return response immediately on success"""
        service = NetworkService()

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            response = service.get("https://example.com")

            assert response.status_code == 200
            assert mock_get.call_count == 1

# --- from pykale__pykale::kale/loaddata/usps.py::USPS.load_samples ---
def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels

# --- from quixio__quix-streams::tests/test_quixstreams/test_sinks/test_core/test_quix_ts_datalake_sink.py::TestStreamTimeoutWiring.test_setup_calls_tracker_start ---
def test_setup_calls_tracker_start(self, sink_factory, mock_blob_client):
        """setup() calls tracker.start() AFTER the blob client is healthy."""
        sink = sink_factory()
        sink._timeout = MagicMock()

        with (
            patch(
                "quixstreams.sinks.core.quix_ts_datalake_sink.get_bucket_name",
                return_value="test-bucket",
            ),
            patch(
                "quixstreams.sinks.core.quix_ts_datalake_sink.BlobStorageClient",
                return_value=mock_blob_client,
            ),
        ):
            sink.setup()

        sink._timeout.start.assert_called_once_with()
