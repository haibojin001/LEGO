# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg27::collections.defaultdict+math.exp+math.pow
# name: collections_math_primitive
# summary: Uses collections.defaultdict, math.exp, math.pow, math.sqrt across 4 repos
# anchor_symbols: ['collections.defaultdict', 'math.exp', 'math.pow', 'math.sqrt']
# observed in 4 repos: ['ScottfreeLLC__AlphaPy', 'd2l-ai__d2l-en', 'eriklindernoren__ML-From-Scratch', 'serengil__chefboost']...

# --- from eriklindernoren__ML-From-Scratch::mlfromscratch/supervised_learning/naive_bayes.py::NaiveBayes._calculate_likelihood ---
def _calculate_likelihood(self, mean, var, x):
        """ Gaussian likelihood of the data x given mean and var """
        eps = 1e-4 # Added in denominator to prevent division by zero
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent

# --- from eriklindernoren__ML-From-Scratch::mlfromscratch/unsupervised_learning/gaussian_mixture_model.py::GaussianMixtureModel.multivariate_gaussian ---
def multivariate_gaussian(self, X, params):
        """ Likelihood """
        n_features = np.shape(X)[1]
        mean = params["mean"]
        covar = params["cov"]
        determinant = np.linalg.det(covar)
        likelihoods = np.zeros(np.shape(X)[0])
        for i, sample in enumerate(X):
            d = n_features  # dimension
            coeff = (1.0 / (math.pow((2.0 * math.pi), d / 2)
                            * math.sqrt(determinant)))
            exponent = math.exp(-0.5 * (sample - mean).T.dot(np.linalg.pinv(covar)).dot((sample - mean)))
            likelihoods[i] = coeff * exponent

        return likelihoods

# --- from d2l-ai__d2l-en::d2l/mxnet.py::bleu ---
def bleu(pred_seq, label_seq, k):
    """Compute the BLEU.

    Defined in :numref:`sec_seq2seq_training`"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

# --- from d2l-ai__d2l-en::d2l/tensorflow.py::bleu ---
def bleu(pred_seq, label_seq, k):
    """Compute the BLEU.

    Defined in :numref:`sec_seq2seq_training`"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

# --- from ScottfreeLLC__AlphaPy::alphapy/transforms.py::zscore ---
def zscore(vec):
    r"""Calculate the Z-Score.

    Parameters
    ----------
    vec : pandas.Series
        The input array for calculating the Z-Score.

    Returns
    -------
    zscore : float
        The value of the Z-Score.

    References
    ----------
    To calculate the Z-Score, you can find more information here [ZSCORE]_.

    .. [ZSCORE] https://en.wikipedia.org/wiki/Standard_score

    Example
    -------

    >>> vec.rolling(window=20).apply(zscore)

    """
    n1 = np.count_nonzero(vec)
    n2 = len(vec) - n1
    fac1 = float(2 * n1 * n2)
    fac2 = float(n1 + n2)
    rbar = fac1 / fac2 + 1
    sr2num = fac1 * (fac1 - n1 - n2)
    sr2den = math.pow(fac2, 2) * (fac2 - 1)
    sr = math.sqrt(sr2num / sr2den)
    if sr2den and sr:
        zscore = (runs(vec) - rbar) / sr
    else:
        zscore = 0
    return zscore

# --- from serengil__chefboost::chefboost/training/Training.py::findGains ---
def findGains(df: pd.DataFrame, config: dict) -> dict:
    """
    Find entropy and gains of each feature in the data frame
    Args:
        df (pd.DataFrame): (sub) train set as data frame
        config (dict): training configuration
    Returns:
        result (dict): gains with respect to the algorithm
    """
    algorithm = config["algorithm"]
    decision_classes = df["Decision"].unique()

    # -----------------------------

    entropy = 0

    if algorithm in ["ID3", "C4.5"]:
        entropy = calculateEntropy(df, config)

    columns = df.shape[1]
    instances = df.shape[0]

    gains = []

    for i in range(0, columns - 1):
        column_name = df.columns[i]
        column_type = df[column_name].dtypes

        logger.debug(f"{column_name} -> {column_type}")

        if pd.api.types.is_numeric_dtype(column_type):
            df = Preprocess.processContinuousFeatures(algorithm, df, column_name, entropy, config)

        classes = df[column_name].value_counts()

        splitinfo = 0
        if algorithm in ["ID3", "C4.5"]:
            gain = entropy * 1
        else:
            gain = 0

        for j in range(0, len(classes)):
            current_class = classes.keys().tolist()[j]
            logger.debug(f"{column_name} -> {current_class}")

            subdataset = df[df[column_name] == current_class]
            logger.debug(subdataset)

            subset_instances = subdataset.shape[0]
            class_probability = subset_instances / instances

            if algorithm in ["ID3", "C4.5"]:
                subset_entropy = calculateEntropy(subdataset, config)
                gain = gain - class_probability * subset_entropy

            if algorithm == "C4.5":
                splitinfo = splitinfo - class_probability * math.log(class_probability, 2)

            elif algorithm == "CART":  # GINI index
                decision_list = subdataset["Decision"].value_counts().tolist()

                subgini = 1

                for current_decision in decision_list:
                    subgini = subgini - math.pow((current_decision / subset_instances), 2)

                gain = gain + (subset_instances / instances) * subgini

            elif algorithm == "CHAID":
                num_of_decisions = len(decision_classes)

                expected = subset_instances / num_of_decisions

                for d in decision_classes:
                    num_of_d = subdataset[subdataset["Decision"] == d].shape[0]

                    chi_square_of_d = math.sqrt(
                        ((num_of_d - expected) * (num_of_d - expected)) / expected
                    )

                    gain += chi_square_of_d

            elif algorithm == "Regression":
                subset_stdev = subdataset["Decision"].std(ddof=0)
                gain = gain + (subset_instances / instances) * subset_stdev

        # iterating over classes for loop end
        # -------------------------------

        if algorithm == "Regression":
            stdev = df["Decision"].std(ddof=0)
            gain = stdev - gain
        if algorithm == "C4.5":
            if splitinfo == 0:
                splitinfo = 100
                # this can be if data set consists of 2 rows and current column consists
                # of 1 class. still decision can be made (decisions for these 2 rows same).
                # set splitinfo to very large value to make gain ratio very small.
                # in this way, we won't find this column as the most dominant one.
            gain = gain / splitinfo

        # ----------------------------------

        gains.append(gain)

    # -------------------------------------------------

    resp_obj = {}
    resp_obj["gains"] = {}

    for idx, feature in enumerate(df.columns[0:-1]):  # Decision is always the last column
        logger.debug(f"{idx}, {feature}")
        resp_obj["gains"][feature] = gains[idx]

    resp_obj["entropy"] = entropy

    return resp_obj

# --- from serengil__chefboost::chefboost/training/Preprocess.py::processContinuousFeatures ---
def processContinuousFeatures(
    algorithm: str, df: pd.DataFrame, column_name: str, entropy: float, config: dict
) -> pd.DataFrame:
    """
    Find the best split point for numeric features
    Args:
        df (pd.DataFrame): (sub) training dataframe
        column_name (str): current column to process
        entropy (float): calculated entropy
        config (dict): training configuration
    Returns
        df (pd.DataFrame): dataframe with numeric columns updated
            to nominal (e.g. instead of continious age >40 or <=40)
    """
    # if True:
    if df[column_name].nunique() <= 20:
        unique_values = sorted(df[column_name].unique())
    else:
        unique_values = []

        df_mean = df[column_name].mean()
        df_std = df[column_name].std(ddof=0)
        df_min = df[column_name].min()
        df_max = df[column_name].max()

        unique_values.append(df[column_name].min())
        unique_values.append(df[column_name].max())
        unique_values.append(df[column_name].mean())

        scales = list(range(-3, +4, 1))
        for scale in scales:
            if df_mean + scale * df_std > df_min and df_mean + scale * df_std < df_max:
                unique_values.append(df_mean + scale * df_std)

        unique_values.sort()

    logger.debug(f"{column_name} -> {unique_values}")

    subset_gainratios = []
    subset_gains = []
    subset_ginis = []
    subset_red_stdevs = []
    subset_chi_squares = []

    if len(unique_values) == 1:
        winner_threshold = unique_values[0]
        df[column_name] = np.where(
            df[column_name] <= winner_threshold,
            "<=" + str(winner_threshold),
            ">" + str(winner_threshold),
        )
        return df

    for i in range(0, len(unique_values) - 1):
        threshold = unique_values[i]

        subset1 = df[df[column_name] <= threshold]
        subset2 = df[df[column_name] > threshold]

        subset1_rows = subset1.shape[0]
        subset2_rows = subset2.shape[0]
        total_instances = df.shape[0]  # subset1_rows+subset2_rows

        subset1_probability = subset1_rows / total_instances
        subset2_probability = subset2_rows / total_instances

        if algorithm in ["ID3", "C4.5"]:
            threshold_gain = (
                entropy
                - subset1_probability * Training.calculateEntropy(subset1, config)
                - subset2_probability * Training.calculateEntropy(subset2, config)
            )
            subset_gains.append(threshold_gain)

        # C4.5 also need gain in the block above.
        # That's why, instead of else if we used direct if condition here
        if algorithm == "C4.5":
            threshold_splitinfo = -subset1_probability * math.log(
                subset1_probability, 2
            ) - subset2_probability * math.log(subset2_probability, 2)
            gainratio = threshold_gain / threshold_splitinfo
            subset_gainratios.append(gainratio)

        elif algorithm == "CART":
            decision_for_subset1 = subset1["Decision"].value_counts().tolist()
            decision_for_subset2 = subset2["Decision"].value_counts().tolist()

            gini_subset1 = 1
            gini_subset2 = 1

            for current_decision_for_subset1 in decision_for_subset1:
                gini_subset1 = gini_subset1 - math.pow(
                    (current_decision_for_subset1 / subset1_rows), 2
                )

            for current_decision_for_subset2 in decision_for_subset2:
                gini_subset2 = gini_subset2 - math.pow(
                    (current_decision_for_subset2 / subset2_rows), 2
                )

            gini = (subset1_rows / total_instances) * gini_subset1 + (
                subset2_rows / total_instances
            ) * gini_subset2

            subset_ginis.append(gini)

        elif algorithm == "CHAID":
            # subset1 = high, subset2 = normal

            unique_decisions = df["Decision"].unique()  # Yes, No
            num_of_decisions = len(unique_decisions)  # 2

            subset1_expected = subset1.shape[0] / num_of_decisions
            subset2_expected = subset2.shape[0] / num_of_decisions

            chi_square = 0
            for d in unique_decisions:  # Yes, No
                # decision = Yes
                subset1_d = subset1[subset1["Decision"] == d]  # high, yes
                subset2_d = subset2[subset2["Decision"] == d]  # normal, yes

                subset1_d_chi_square = math.sqrt(
                    (
                        (subset1_d.shape[0] - subset1_expected)
                        * (subset1_d.shape[0] - subset1_expected)
                    )
                    / subset1_expected
                )

                subset2_d_chi_square = math.sqrt(
                    (
                        (subset2_d.shape[0] - subset2_expected)
                        * (subset2_d.shape[0] - subset2_expected)
                    )
                    / subset2_expected
                )

                chi_square = chi_square + subset1_d_chi_square + subset2_d_chi_square

            subset_chi_squares.append(chi_square)

        # ----------------------------------
        elif algorithm == "Regression":
            superset_stdev = df["Decision"].std(ddof=0)
            subset1_stdev = subset1["Decision"].std(ddof=0)
            subset2_stdev = subset2["Decision"].std(ddof=0)

            threshold_weighted_stdev = (subset1_rows / total_instances) * subset1_stdev + (
                subset2_rows / total_instances
            ) * subset2_stdev
            threshold_reducted_stdev = superset_stdev - threshold_weighted_stdev
            subset_red_stdevs.append(threshold_reducted_stdev)

        # ----------------------------------

    if algorithm == "C4.5":
        winner_one = subset_gainratios.index(max(subset_gainratios))
    elif (
        algorithm == "ID3"
    ):  # actually, ID3 does not support for continuous features but we can still do it
        winner_one = subset_gains.index(max(subset_gains))
    elif algorithm == "CART":
        winner_one = subset_ginis.index(min(subset_ginis))
    elif algorithm == "CHAID":
        winner_one = subset_chi_squares.index(max(subset_chi_squares))
    elif algorithm == "Regression":
        winner_one = subset_red_stdevs.index(max(subset_red_stdevs))

    winner_threshold = unique_values[winner_one]
    logger.debug(f"{column_name}: {winner_threshold} in {unique_values}")

    logger.debug(f"theshold is {winner_threshold} for {column_name}")
    df[column_name] = np.where(
        df[column_name] <= winner_threshold,
        "<=" + str(winner_threshold),
        ">" + str(winner_threshold),
    )

    return df
