# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg3::IPython.get_ipython+ast.And+ast.BinOp
# name: IPython_ast_primitive
# summary: Uses IPython.get_ipython, ast.And, ast.BinOp, ast.BoolOp across 23 repos
# anchor_symbols: ['IPython.get_ipython', 'ast.And', 'ast.BinOp', 'ast.BoolOp', 'ast.Compare', 'ast.ImportFrom']
# observed in 23 repos: ['BiomedSciAI__causallib', 'CamDavidsonPilon__lifelines', 'Data-Centric-AI-Community__fg-data-profiling', 'HazyResearch__meerkat', 'Lightning-AI__torchmetrics']...

# --- from BiomedSciAI__causallib::causallib/tests/test_rlearner.py::TestRlearner.create_complex_data_for_ate_victor.g ---
def g(x):
            return np.power(np.sin(x), 2)

# --- from microsoft__nni::examples/trials/weight_sharing/ga_squad/util.py::get_variable ---
def get_variable(name, temp_s):
    '''
    Get variable by name.
    '''
    return tf.Variable(tf.zeros(temp_s), name=name)

# --- from microsoft__nni::examples/trials/ga_squad/util.py::get_variable ---
def get_variable(name, temp_s):
    '''
    Get variable by name.
    '''
    return tf.Variable(tf.zeros(temp_s), name=name)

# --- from CamDavidsonPilon__lifelines::lifelines/utils/__init__.py::quiet_log2 ---
def quiet_log2(p):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"divide by zero encountered in log2")
        return np.log2(p)

# --- from xorbitsai__xorbits::python/xorbits/_mars/dataframe/base/eval.py::CollectionVisitor.eval ---
def eval(self, expr, rewrite=True):
        if rewrite:
            expr = self._preparse(expr)
        node = ast.fix_missing_locations(ast.parse(expr))
        return self.visit(node)

# --- from piskvorky__gensim::gensim/utils.py::ignore_deprecation_warning ---
def ignore_deprecation_warning():
    """Contextmanager for ignoring DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield

# --- from modin-project__modin::modin/core/dataframe/algebra/default2pandas/groupby.py::GroupBy._call_groupby ---
def _call_groupby(cls, df, *args, **kwargs):  # noqa: PR01
        """Call .groupby() on passed `df`."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return df.groupby(*args, **kwargs)

# --- from modin-project__modin::modin/tests/pandas/dataframe/test_iter.py::test___repr__does_not_raise_attribute_column_warning ---
def test___repr__does_not_raise_attribute_column_warning():
    # See https://github.com/modin-project/modin/issues/5380
    df = pd.DataFrame([1])
    with warnings.catch_warnings():
        warnings.filterwarnings(action="error", message=SET_DATAFRAME_ATTRIBUTE_WARNING)
        repr(df)

# --- from d2l-ai__d2l-en::d2l/tensorflow.py::LinearRegressionScratch.__init__ ---
def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
        b = tf.zeros(1)
        self.w = tf.Variable(w, trainable=True)
        self.b = tf.Variable(b, trainable=True)

# --- from Data-Centric-AI-Community__fg-data-profiling::src/data_profiling/model/typeset.py::ProfilingTypeSet.__init__ ---
def __init__(self, config: Settings, type_schema: dict = None):
        self.config = config

        types = typeset_types(config)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            super().__init__(types)

        self.type_schema = self._init_type_schema(type_schema or {})

# --- from d2l-ai__d2l-en::d2l/tensorflow.py::corr2d ---
def corr2d(X, K):
    """Compute 2D cross-correlation.

    Defined in :numref:`sec_conv_layer`"""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y

# --- from HazyResearch__meerkat::meerkat/tools/docs.py::DescriptionSection.fix_indentation ---
def fix_indentation(self, docstring: str) -> str:
        # get common leading whitespace from docstring ignoring first line
        lines = docstring.splitlines()
        leading_whitespace = min(
            len(line) - len(line.lstrip()) for line in lines[1:] if line.strip()
        )

        prefix = leading_whitespace * " "
        text = indent(dedent(self.text), prefix)

        return text
