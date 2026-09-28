# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg23::matplotlib.pyplot.show
# name: matplotlib_primitive
# summary: Uses matplotlib.pyplot.show across 17 repos
# anchor_symbols: ['matplotlib.pyplot.show']
# observed in 17 repos: ['DeepWisdom__AutoDL', 'JacksonWuxs__DaPy', 'ankonzoid__artificio', 'annoviko__pyclustering', 'apachecn__python_data_analysis_and_mining_action']...

# --- from lazyprogrammer__machine_learning_examples::ann_class/tf_example.py::init_weights ---
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# --- from lazyprogrammer__machine_learning_examples::bayesian_ml/3/run.py::e_ln_q_gamma ---
def e_ln_q_gamma(a, b):
  return np.log(b) - a - np.log(np.abs(gamma(a))) + (a - 1)*digamma(a)

# --- from shashankvemuri__Finance::portfolio_strategies/sma_trading_strategy.py::get_stock_data ---
def get_stock_data(stock, num_of_years):
    start = dt.date.today() - dt.timedelta(days=365 * num_of_years)
    end = dt.datetime.now()
    return yf.download(stock, start, end, interval='1d')

# --- from shashankvemuri__Finance::machine_learning/technical_indicators_clustering.py::plot_clusters ---
def plot_clusters(data):
    plt.scatter(data['SMA_5'], data['SMA_15'], c=data['Cluster'], cmap='viridis')
    plt.xlabel('SMA 5')
    plt.ylabel('SMA 15')
    plt.title('Stock Data Clusters')
    plt.show()

# --- from devAmoghS__Machine-Learning-with-Python::working_with_data/utils.py::scatter ---
def scatter():
    plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
    plt.scatter(xs, ys2, marker='.', color='gray',  label='ys2')
    plt.xlabel('xs')
    plt.ylabel('ys')
    plt.legend(loc=9)
    plt.show()

# --- from eriklindernoren__ML-From-Scratch::mlfromscratch/supervised_learning/logistic_regression.py::LogisticRegression._initialize_parameters ---
def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

# --- from terrytangyuan__distributed-ml-patterns::code/project/code/multi-worker-distributed-training.py::_get_serve_image_fn ---
def _get_serve_image_fn(model):
    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.string, name='image_bytes')])
    def serve_image_fn(bytes_inputs):
        decoded_images = tf.map_fn(_preprocess, bytes_inputs, dtype=tf.uint8)
        return model(decoded_images)
    return serve_image_fn

# --- from d2l-ai__d2l-en::d2l/tensorflow.py::MultiHeadAttention.transpose_output ---
def transpose_output(self, X):
        """Reverse the operation of transpose_qkv.
    
        Defined in :numref:`sec_multihead-attention`"""
        X = tf.reshape(X, shape=(-1, self.num_heads, X.shape[1], X.shape[2]))
        X = tf.transpose(X, perm=(0, 2, 1, 3))
        return tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))

# --- from eriklindernoren__ML-From-Scratch::mlfromscratch/supervised_learning/multi_class_lda.py::MultiClassLDA.plot_in_2d ---
def plot_in_2d(self, X, y, title=None):
        """ Plot the dataset X and the corresponding labels y in 2D using the LDA
        transformation."""
        X_transformed = self.transform(X, y, n_components=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        plt.scatter(x1, x2, c=y)
        if title: plt.title(title)
        plt.show()

# --- from apachecn__python_data_analysis_and_mining_action::chapter5/code.py::programmer_5 ---
def programmer_5(data_zs, r):
    # 进行数据降维
    tsne = TSNE()
    tsne.fit_transform(data_zs)
    tsne = pd.DataFrame(tsne.embedding_, index=data_zs.index)

    # 不同类别用不同颜色和样式绘图
    d = tsne[r[u'聚类类别'] == 0]
    plt.plot(d[0], d[1], 'r.')
    d = tsne[r[u'聚类类别'] == 1]
    plt.plot(d[0], d[1], 'go')
    d = tsne[r[u'聚类类别'] == 2]
    plt.plot(d[0], d[1], 'b*')
    plt.show()

# --- from google__uncertainty-baselines::experimental/language_structure/psl/psl_model.py::PSLModel._unary_to_binary ---
def _unary_to_binary(predicate: tf.Tensor, transpose: bool) -> tf.Tensor:
    predicate_matrix = tf.repeat(predicate, predicate.shape[-1], axis=-1)
    predicate_matrix = tf.reshape(
        predicate_matrix, [-1, predicate.shape[-1], predicate.shape[-1]])
    if transpose:
      predicate_matrix = tf.transpose(predicate_matrix, perm=[0, 2, 1])
    return predicate_matrix

# --- from devAmoghS__Machine-Learning-with-Python::helpers/gradient_descent.py::plot_estimated_derivative ---
def plot_estimated_derivative():
    def square(x):
        return x * x

    def derivative(x):
        return 2 * x

    def derivative_estimate():
        difference_quotient(square, x, h=0.00001)

    # plot to show they're basically the same
    import matplotlib.pyplot as plt
    x = range(-10, 10)
    plt.plot(x, map(derivative, x), 'rx')  # red  x
    plt.plot(x, map(derivative_estimate, x), 'b+')  # blue +
    plt.show()  # purple *, hopefully
