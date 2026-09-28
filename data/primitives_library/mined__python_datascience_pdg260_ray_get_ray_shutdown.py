# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg260::ray.get+ray.shutdown
# name: ray_primitive
# summary: Uses ray.get, ray.shutdown across 5 repos
# anchor_symbols: ['ray.get', 'ray.shutdown']
# observed in 5 repos: ['modin-project__modin', 'ruc-datalab__DeepAnalyze', 'run-house__kubetorch', 'sematic-ai__sematic', 'stitchfix__hamilton']...

# --- from stitchfix__hamilton::graph_adapter_tests/h_ray/test_h_ray.py::init ---
def init():
    ray.init()
    yield "initialized"
    ray.shutdown()

# --- from modin-project__modin::modin/tests/experimental/torch/test_dataloader.py::ray_fix ---
def ray_fix():
    ray.init(num_cpus=1)
    yield None
    ray.shutdown()

# --- from sematic-ai__sematic::sematic/examples/summarization_finetune/train_eval.py::TrainingArguments.to_hugging_face ---
def to_hugging_face(self) -> HfTrainingArguments:
        return HfTrainingArguments(**asdict(self))

# --- from ruc-datalab__DeepAnalyze::deepanalyze/SkyRL/skyrl-train/tests/gpu/conftest.py::ray_init_fixture ---
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()
    ray_init_for_tests()
    yield
    # call ray shutdown after a test regardless
    ray.shutdown()

# --- from ruc-datalab__DeepAnalyze::deepanalyze/SkyRL/skyrl-train/tests/cpu/conftest.py::ray_init ---
def ray_init():
    """Initialize Ray once for the entire test session."""
    if not ray.is_initialized():
        ray.init()
    yield
    if ray.is_initialized():
        ray.shutdown()

# --- from modin-project__modin::modin/tests/core/storage_formats/pandas/test_internals.py::test_materialization_hook_serialization ---
def test_materialization_hook_serialization():
    @ray.remote(num_returns=1)
    def f1():
        return [1, 2, 3]

    @ray.remote(num_returns=1)
    def f2(i):
        return i

    hook = MetaList(f1.remote())[2]
    assert ray.get(f2.remote(hook)) == 3

# --- from run-house__kubetorch::python_client/kubetorch/serving/ray_supervisor.py::RayProcess.framework_cleanup ---
def framework_cleanup(self):
        """Clean up Ray state for reloads."""
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray shutdown completed.")
        except ImportError:
            logger.debug("Ray not available for cleanup")
        except Exception as e:
            logger.debug(f"Failed to shutdown Ray: {e}")

# --- from stitchfix__hamilton::hamilton/experimental/h_ray.py::RayGraphAdapter.build_result ---
def build_result(self, **outputs: typing.Dict[str, typing.Any]) -> typing.Any:
        """Builds the result and brings it back to this running process.

        :param outputs: the dictionary of key -> Union[ray object reference | value]
        :return: The type of object returned by self.result_builder.
        """
        if logger.isEnabledFor(logging.DEBUG):
            for k, v in outputs.items():
                logger.debug(f"Got output {k}, with type [{type(v)}].")
        # need to wrap our result builder in a remote call and then pass in what we want to build from.
        remote_combine = ray.remote(self.result_builder.build_result).remote(**outputs)
        result = ray.get(remote_combine)  # this materializes the object locally
        return result

# --- from sematic-ai__sematic::sematic/ee/plugins/external_resource/ray/cluster.py::RayCluster._do_ray_init ---
def _do_ray_init(self) -> "RayCluster":
        """Connect to Ray if not already connected."""
        if ray.is_initialized():
            return self
        try:
            if self._cluster_name is not None:
                logger.info("Connecting to Ray using URI '%s'", self._head_uri)
                ray.init(address=self._head_uri, log_to_driver=self.forward_logs)
            else:
                ray.init(log_to_driver=self.forward_logs)
        except ConnectionError:
            logger.error("Could not connect to Ray...")
            try:
                ray.shutdown()
            except Exception:
                logger.exception(
                    "Error disconnecting from Ray while handling connection error:"
                )
            raise

        logger.info("Initialized connection to Ray for cluster resource %s", self.id)
        return self
