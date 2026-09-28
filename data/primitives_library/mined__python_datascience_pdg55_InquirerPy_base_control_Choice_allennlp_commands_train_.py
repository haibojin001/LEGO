# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg55::InquirerPy.base.control.Choice+allennlp.commands.train.train_model+allennlp.common.file_utils.CacheFile
# name: InquirerPy_allennlp_primitive
# summary: Uses InquirerPy.base.control.Choice, allennlp.commands.train.train_model, allennlp.common.file_utils.CacheFile, allennlp.training.util.make_vocab_from_params across 23 repos
# anchor_symbols: ['InquirerPy.base.control.Choice', 'allennlp.commands.train.train_model', 'allennlp.common.file_utils.CacheFile', 'allennlp.training.util.make_vocab_from_params', 'asyncio.create_task', 'asyncio.gather']
# observed in 23 repos: ['HazyResearch__meerkat', 'Lightning-AI__torchmetrics', 'NannyML__nannyml', 'OML-Team__open-metric-learning', 'airbnb__knowledge-repo']...

# --- from iterative__mlem::mlem/core/objects.py::MlemObject.meta_hash ---
def meta_hash(self):
        return hashlib.md5(  # nosec: B324
            safe_dump(self.dict()).encode("utf8")
        ).hexdigest()

# --- from encord-team__encord-active::src/encord_active/lib/metrics/heuristic/img_features.py::SharpnessMetric.execute ---
def execute(self, image: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.Laplacian(image, cv2.CV_64F).var()

# --- from encord-team__encord-active::src/encord_active/lib/metrics/heuristic/img_features.py::BlurMetric.execute ---
def execute(self, image: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return 1 - cv2.Laplacian(image, cv2.CV_64F).var()

# --- from iterative__mlem::mlem/contrib/docker/context.py::DockerModelDirectory.write_configs ---
def write_configs(self):
        with self.fs.open(
            posixpath.join(self.path, SERVER), "w", encoding="utf8"
        ) as f:
            safe_dump(self.server.dict(), f)

# --- from OML-Team__open-metric-learning::oml/losses/arcface.py::ArcFaceLoss._log_accuracy_on_batch ---
def _log_accuracy_on_batch(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._last_logs["accuracy"] = torch.mean((y == torch.argmax(logits, 1)).to(torch.float32))

# --- from airbnb__knowledge-repo::knowledge_repo/post.py::ReferenceCache.__delitem__ ---
def __delitem__(self, key):
        parents = posixpath.dirname(key).split('/')
        cache = self._cache
        for parent in parents:
            if parent:
                cache = cache[parent]
        del cache[posixpath.basename(key)]

# --- from airbnb__knowledge-repo::knowledge_repo/post.py::ReferenceCache.__getitem__ ---
def __getitem__(self, key):
        parents = posixpath.dirname(key).split('/')
        cache = self._cache
        for parent in parents:
            if parent:
                cache = cache[parent]
        return cache[posixpath.basename(key)]

# --- from bodywork-ml__bodywork-core::src/bodywork/cli/cli.py::_delete_deployment ---
def _delete_deployment(
    name: str = Argument(...),
    asynchronous: bool = Option(False, "--async", hidden=True),
):
    if asynchronous:
        delete_workflow_job(BODYWORK_NAMESPACE, name)
    else:
        delete_deployment(name)
    sys.exit(0)

# --- from violit-dev__violit::src/violit/db.py::ViolItDB.all ---
def all(self, model: Type[T]) -> List[T]:
        """
        Return all records.

        Example::

            tasks = app.db.all(Task)
        """
        _check_sqlmodel()
        with Session(self._engine) as session:
            return list(session.exec(select(model)).all())

# --- from bodywork-ml__bodywork-core::src/bodywork/cli/cli.py::_stage ---
def _stage(
    git_url: str = Argument(...),
    git_branch: str = Option("", "--branch"),
    stage_name: str = Argument(...),
    timeout: int = Argument(None),
):
    try:
        run_stage(stage_name, git_url, git_branch, timeout=timeout)
        sys.exit(0)
    except Exception:
        sys.exit(1)

# --- from microsoft__RD-Agent::test/notebook/testfiles/main2.py::main.green_mask ---
def green_mask(img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([35, 51, 41], dtype=np.uint8)
        upper = np.array([85, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = (mask > 0).astype(np.uint8)
        return mask[..., None]

# --- from probcomp__bayeslite::src/bqlfn.py::correlation_cramerphi ---
def correlation_cramerphi(data0, data1):
    # Compute observed chi^2 statistic.
    chi2, n0, n1 = cramerphi_chi2(data0, data1)
    if math.isnan(chi2):
        return float('NaN')
    n = len(data0)
    assert n == len(data1)
    # Compute observed correlation.
    return math.sqrt(chi2 / (n * (min(n0, n1) - 1)))
