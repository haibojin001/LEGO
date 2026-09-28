# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg155::tensorflow.compat.v1.ConfigProto+tensorflow.compat.v1.Session
# name: tensorflow_primitive
# summary: Uses tensorflow.compat.v1.ConfigProto, tensorflow.compat.v1.Session across 2 repos
# anchor_symbols: ['tensorflow.compat.v1.ConfigProto', 'tensorflow.compat.v1.Session']
# observed in 2 repos: ['microsoft__nni', 'tflearn__tflearn']...

# --- from microsoft__nni::nni/algorithms/hpo/ppo_tuner/util.py::make_session ---
def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if config is None:
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
        config.gpu_options.allow_growth = True

    if make_default:
        return tf.InteractiveSession(config=config, graph=graph)
    else:
        return tf.Session(config=config, graph=graph)

# --- from tflearn__tflearn::tflearn/estimators/base.py::BaseEstimator.__init__ ---
def __init__(self, metric=None, log_dir='/tmp/tflearn_logs/',
                 global_step=None, session=None, graph=None, name=None):

        self.name = name

        # Estimator Graph and Session
        self.graph = tf.Graph() if graph is None else graph
        with self.graph.as_default():
            conf = tf.ConfigProto(allow_soft_placement=True)
            self.session = tf.Session(config=conf) if session is None else session
        if global_step is None:
            with self.graph.as_default():
                self.global_step = tf.train.get_or_create_global_step()

        self.metric = validate_func(metric)

        # Estimator Graph Branches
        self._train = GraphBranch()
        self._pred = GraphBranch()
        self._transform = GraphBranch()
        self._eval = GraphBranch()

        # Tensor Utils
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self._is_initialized = False
        self._to_be_restored = False

        # Ops
        self.train_op = None
        self.loss_op = None
