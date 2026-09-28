# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg68::albumentations.Affine+albumentations.ColorJitter+albumentations.Compose
# name: albumentations_primitive
# summary: Uses albumentations.Affine, albumentations.ColorJitter, albumentations.Compose, albumentations.GaussianNoise across 12 repos
# anchor_symbols: ['albumentations.Affine', 'albumentations.ColorJitter', 'albumentations.Compose', 'albumentations.GaussianNoise', 'albumentations.HorizontalFlip', 'albumentations.HueSaturationValue']
# observed in 12 repos: ['Data-Centric-AI-Community__fg-data-profiling', 'JosephLai241__URS', 'OML-Team__open-metric-learning', 'WecoAI__aideml', 'awslabs__gluonts']...

# --- from microsoft__nni::test/algo/nas/graph_converter/test_convert.py::TestConvert.test_vae.VAE.decode ---
def decode(self, z):
                h3 = F.relu(self.fc3(z))
                return torch.sigmoid(self.fc4(h3))

# --- from yzhao062__pyod::pyod/models/so_gaal.py::Discriminator.forward ---
def forward(self, x):
        x = F.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

# --- from yzhao062__pyod::pyod/models/gaal_base.py::create_discriminator.Discriminator.forward ---
def forward(self, x):
            x = F.relu(self.layer1(x))
            x = torch.sigmoid(self.layer2(x))
            return x

# --- from microsoft__RD-Agent::rdagent/scenarios/kaggle/experiment/templates/meta_tpl_deprecated/model/model_nn.py::FeatureInteractionModel.forward ---
def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# --- from awslabs__gluonts::src/gluonts/shell/serve/app.py::batch_inference_invocations.invocations_error_wrapper ---
def invocations_error_wrapper() -> Response:
        try:
            return invocations()
        except Exception:
            return Response(
                json.dumps({"error": traceback.format_exc()}),
                mimetype="application/jsonlines",
            )

# --- from awslabs__gluonts::src/gluonts/shell/serve/app.py::with_timeout ---
def with_timeout(fn, args, timeout):
    queue = mp.Queue()
    process = mp.Process(target=do, args=(fn, args, queue))
    process.start()

    try:
        return queue.get(True, timeout=timeout)
    except QueueEmpty:
        os.kill(process.pid, signal.SIGKILL)
        return None

# --- from recommenders-team__recommenders::recommenders/models/deeprec/models/dkn.py::DKN._init_embedding ---
def _init_embedding(self, file_path):
        """Load pre-trained embeddings as a constant tensor.

        Args:
            file_path (str): the pre-trained embeddings filename.

        Returns:
            object: A constant tensor.
        """
        return tf.constant(np.load(file_path).astype(np.float32))

# --- from JosephLai241__URS::tests/test_utils/test_Tools.py::Login.create_reddit_object ---
def create_reddit_object():
        load_dotenv()

        return praw.Reddit(
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET"),
            user_agent=os.getenv("USER_AGENT"),
            username=os.getenv("USERNAME"),
            password=os.getenv("PASSWORD"),
        )

# --- from microsoft__RD-Agent::rdagent/scenarios/kaggle/experiment/templates/meta_tpl_deprecated/model/model_nn.py::FeatureInteractionModel.__init__ ---
def __init__(self, num_features):
        super(FeatureInteractionModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

# --- from JosephLai241__URS::tests/test_praw_scrapers/test_utils/test_Validation.py::Login.create_reddit_object ---
def create_reddit_object():
        load_dotenv()

        return praw.Reddit(
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET"),
            user_agent=os.getenv("USER_AGENT"),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD"),
        )

# --- from xorbitsai__xorbits::python/xorbits/_mars/learn/ensemble/_bagging.py::_set_random_states ---
def _set_random_states(estimator, random_state=None):
    random_state = sklearn_check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    if to_set:
        estimator.set_params(**to_set)

# --- from ruc-datalab__DeepAnalyze::deepanalyze/SkyRL/skyagent/skyagent/tasks/verifiers/torl/eval.py::call_with_timeout ---
def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()
