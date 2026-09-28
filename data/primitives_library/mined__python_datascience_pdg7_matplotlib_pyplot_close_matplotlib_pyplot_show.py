# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg7::matplotlib.pyplot.close+matplotlib.pyplot.show
# name: matplotlib_primitive
# summary: Uses matplotlib.pyplot.close, matplotlib.pyplot.show across 18 repos
# anchor_symbols: ['matplotlib.pyplot.close', 'matplotlib.pyplot.show']
# observed in 18 repos: ['BiomedSciAI__causallib', 'DeepWisdom__AutoDL', 'PIA-Group__BioSPPy', 'annoviko__pyclustering', 'awslabs__gluonts']...

# --- from devAmoghS__Machine-Learning-with-Python::working_with_data/utils.py::plot_histogram ---
def plot_histogram(points, bucket_size, title=""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/Auto_Video/skeleton/data/stratified_sampler.py::get_locs ---
def get_locs(n):
    percent = 1. / n
    locs = [percent * random.random()]
    last = locs[0]
    for i in range(n - 1):
        value = last + percent * random.uniform(0.8, 1.2)  #
        locs.append(value)
        last = value
    return locs

# --- from microsoft__nni::examples/trials/kaggle-tgs-salt/augmentation.py::RandomRotateWithMask.get_angle ---
def get_angle(self):
        if isinstance(self.degrees, collections.Sequence):
            index = int(random.random() * len(self.degrees))
            return self.degrees[index]
        else:
            return random.uniform(-self.degrees, self.degrees)

# --- from shashankvemuri__Finance::stock_data/yf_intraday_data.py::plot_close_prices ---
def plot_close_prices(stock_data, stock_symbol):
    close_prices = stock_data['Close']
    
    # Creating the plot
    fig, ax = plt.subplots()
    ax.plot(close_prices)
    ax.set_title(f'Price for {stock_symbol}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    plt.show()

# --- from devAmoghS__Machine-Learning-with-Python::working_with_data/utils.py::compare_two_distributions ---
def compare_two_distributions():

    random.seed(0)

    uniform = [random.randrange(-100,101) for _ in range(200)]
    normal = [57 * inverse_normal_cdf(random.random())
              for _ in range(200)]

    plot_histogram(uniform, 10, "Uniform Histogram")
    plot_histogram(normal, 10, "Normal Histogram")

# --- from tflearn__tflearn::examples/reinforcement_learning/atari_1step_qlearning.py::AtariEnvironment.get_preprocessed_frame ---
def get_preprocessed_frame(self, observation):
        """
        0) Atari frames: 210 x 160
        1) Get image grayscale
        2) Rescale image 110 x 84
        3) Crop center 84 x 84 (you can crop top/bottom according to the game)
        """
        return resize(rgb2gray(observation), (110, 84))[13:110 - 13, :]

# --- from BiomedSciAI__causallib::causallib/tests/test_plots.py::TestPlots.test_plot_covariate_balance_slope_labeled_correctly ---
def test_plot_covariate_balance_slope_labeled_correctly(self):
        f, ax = plt.subplots()
        axis = self.propensity_evaluation.plot_covariate_balance(kind="slope", ax=ax)
        self.assertIsInstance(axis, matplotlib.axes.Axes)
        self.assertEqual([x.get_xdata() for x in axis.get_lines()][1][0], "unweighted")
        plt.close()

# --- from BiomedSciAI__causallib::causallib/tests/test_plots.py::TestPlots.test_plot_covariate_balance_love_draws_thresh ---
def test_plot_covariate_balance_love_draws_thresh(self):
        thresh = 0.1
        f, ax = plt.subplots()
        axis = self.propensity_evaluation.plot_covariate_balance(
            kind="love", thresh=thresh, ax=ax
        )
        self.assertIsInstance(axis, matplotlib.axes.Axes)
        self.assertEqual(thresh, axis.get_lines()[0].get_xdata()[0])
        plt.close()

# --- from microsoft__nni::docs/source/tutorials/darts.py::plot_double_cells ---
def plot_double_cells(arch_dict):
    image1 = plot_single_cell(arch_dict, 'normal')
    image2 = plot_single_cell(arch_dict, 'reduce')
    height_ratio = max(image1.size[1] / image1.size[0], image2.size[1] / image2.size[0]) 
    _, axs = plt.subplots(1, 2, figsize=(20, 10 * height_ratio))
    axs[0].imshow(image1)
    axs[1].imshow(image2)
    axs[0].axis('off')
    axs[1].axis('off')
    plt.show()

# --- from annoviko__pyclustering::pyclustering/nnet/som.py::som.show_distance_matrix ---
def show_distance_matrix(self):
        """!
        @brief Shows gray visualization of U-matrix (distance matrix).
        
        @see get_distance_matrix()
        
        """
        distance_matrix = self.get_distance_matrix()

        plt.imshow(distance_matrix, cmap=plt.get_cmap('hot'), interpolation='kaiser')
        plt.title("U-Matrix")
        plt.colorbar()
        plt.show()

# --- from mwaskom__seaborn::tests/test_matrix.py::TestHeatmap.test_heatmap_cbar ---
def test_heatmap_cbar(self):

        f = plt.figure()
        mat.heatmap(self.df_norm)
        assert len(f.axes) == 2
        plt.close(f)

        f = plt.figure()
        mat.heatmap(self.df_norm, cbar=False)
        assert len(f.axes) == 1
        plt.close(f)

        f, (ax1, ax2) = plt.subplots(2)
        mat.heatmap(self.df_norm, ax=ax1, cbar_ax=ax2)
        assert len(f.axes) == 2
        plt.close(f)

# --- from tflearn__tflearn::examples/reinforcement_learning/atari_1step_qlearning.py::AtariEnvironment.get_initial_state ---
def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer.
        """
        # Clear the state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack([x_t for i in range(self.action_repeat)], axis=0)

        for i in range(self.action_repeat-1):
            self.state_buffer.append(x_t)
        return s_t
