# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg472::sklearn.preprocessing.StandardScaler
# name: sklearn_primitive
# summary: Uses sklearn.preprocessing.StandardScaler across 4 repos
# anchor_symbols: ['sklearn.preprocessing.StandardScaler']
# observed in 4 repos: ['HDI-Project__ATM', 'ScottfreeLLC__AlphaPy', 'firmai__pandasvault', 'nidhaloff__igel']...

# --- from nidhaloff__igel::igel/preprocessing.py::normalize ---
def normalize(x, y=None, method="standard"):
    methods = ("minmax", "standard")

    if method not in methods:
        raise Exception(
            f"Please choose one of the available scaling methods => {methods}"
        )
    logger.info(f"performing a {method} scaling ...")
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    if not y:
        return scaler.fit_transform(X=x)
    else:
        return scaler.fit_transform(X=x, y=y)

# --- from firmai__pandasvault::pandasvault/__init__.py::scaler ---
def scaler(df,scaler=None,train=True, target=None, cols_ignore=None, type="Standard"):

  if cols_ignore:
    hold = df[cols_ignore].copy()
    df = df.drop(cols_ignore,axis=1)
  if target:
    x = df.drop([target],axis=1).values #returns a numpy array
  else:
    x = df.values
  if train:
    if type=="Standard":
      scal = StandardScaler()
    elif type=="MinMax":
      scal = MinMaxScaler()
    scal.fit(x)
    x_scaled = scal.transform(x)
  else:
    x_scaled = scaler.transform(x)
  
  if target:
    df_out = pd.DataFrame(x_scaled, index=df.index, columns=df.drop([target],axis=1).columns)
    df_out[target]= df[target]
  else:
    df_out = pd.DataFrame(x_scaled, index=df.index, columns=df.columns)
  
  df_out = pd.concat((hold,df_out),axis=1)
  if train:
    return df_out, scal
  else:
    return df_out

# --- from ScottfreeLLC__AlphaPy::alphapy/features.py::create_pca_features ---
def create_pca_features(features, model):
    r"""Apply Principal Component Analysis (PCA) to the features.

    Parameters
    ----------
    features : numpy array
        The input features.
    model : alphapy.Model
        The model object with the PCA parameters.

    Returns
    -------
    pfeatures : numpy array
        The PCA features.
    pnames : list
        The PCA feature names.

    References
    ----------
    You can find more information on Principal Component Analysis here [PCA]_.

    .. [PCA] http://scikit-learn.org/stable/modules/decomposition.html#pca

    """

    logger.info("Creating PCA Features")

    # Extract model parameters

    pca_inc = model.specs['pca_inc']
    pca_max = model.specs['pca_max']
    pca_min = model.specs['pca_min']
    pca_whiten = model.specs['pca_whiten']

    # Log model parameters

    logger.info("PCA Minimum   : %d", pca_min)
    logger.info("PCA Maximum   : %d", pca_max)
    logger.info("PCA Increment : %d", pca_inc)
    logger.info("PCA Whitening : %r", pca_whiten)

    # Generate clustering features

    pfeatures = np.zeros((features.shape[0], 1))
    pnames = []
    for i in range(pca_min, pca_max+1, pca_inc):
        logger.info("n_components = %d", i)
        X_pca = PCA(n_components=i, whiten=pca_whiten).fit_transform(features)
        pfeatures = np.column_stack((pfeatures, X_pca))
        pnames.append(USEP.join(['pca', str(i)]))
    pfeatures = np.delete(pfeatures, 0, axis=1)

    # Return new clustering features

    logger.info("PCA Feature Count : %d", pfeatures.shape[1])
    return pfeatures, pnames

# --- from ScottfreeLLC__AlphaPy::alphapy/features.py::create_clusters ---
def create_clusters(features, model):
    r"""Cluster the given features.

    Parameters
    ----------
    features : numpy array
        The features to cluster.
    model : alphapy.Model
        The model object with the clustering parameters.

    Returns
    -------
    cfeatures : numpy array
        The calculated clusters.
    cnames : list
        The cluster feature names.

    References
    ----------
    You can find more information on clustering here [CLUS]_.

    .. [CLUS] http://scikit-learn.org/stable/modules/clustering.html

    """

    logger.info("Creating Clustering Features")

    # Extract model parameters

    cluster_inc = model.specs['cluster_inc']
    cluster_max = model.specs['cluster_max']
    cluster_min = model.specs['cluster_min']
    seed = model.specs['seed']

    # Log model parameters

    logger.info("Cluster Minimum   : %d", cluster_min)
    logger.info("Cluster Maximum   : %d", cluster_max)
    logger.info("Cluster Increment : %d", cluster_inc)

    # Generate clustering features

    cfeatures = np.zeros((features.shape[0], 1))
    cnames = []
    for i in range(cluster_min, cluster_max+1, cluster_inc):
        logger.info("k = %d", i)
        km = MiniBatchKMeans(n_clusters=i, random_state=seed)
        km.fit(features)
        labels = km.predict(features)
        labels = labels.reshape(-1, 1)
        cfeatures = np.column_stack((cfeatures, labels))
        cnames.append(USEP.join(['cluster', str(i)]))
    cfeatures = np.delete(cfeatures, 0, axis=1)

    # Return new clustering features

    logger.info("Clustering Feature Count : %d", cfeatures.shape[1])
    return cfeatures, cnames

# --- from HDI-Project__ATM::atm/classifier.py::Model._make_pipeline ---
def _make_pipeline(self):
        """
        Makes the classifier as well as scaling or dimension reduction steps.
        """
        # create a list of steps, starting with the data encoder
        steps = []

        # create a classifier with specified parameters
        hyperparameters = {k: v for k, v in list(self.params.items())
                           if k not in Model.ATM_KEYS}
        atm_params = {k: v for k, v in list(self.params.items())
                      if k in Model.ATM_KEYS}

        # do special conversions
        hyperparameters = self._special_conversions(hyperparameters)
        classifier = self.class_(**hyperparameters)

        if Model.PCA in atm_params and atm_params[Model.PCA]:
            whiten = (Model.WHITEN in atm_params and atm_params[Model.WHITEN])
            pca_dims = atm_params[Model.PCA_DIMS]
            # PCA dimension in atm_params is a float reprsenting percentages of
            # features to use
            if pca_dims < 1:
                dimensions = int(pca_dims * float(self.num_features))
                logger.info("Using PCA to reduce %d features to %d dimensions"
                            % (self.num_features, dimensions))
                pca = decomposition.PCA(n_components=dimensions, whiten=whiten)
                steps.append(('pca', pca))

        # should we scale the data?
        if atm_params.get(Model.SCALE):
            steps.append(('standard_scale', StandardScaler()))
        elif self.params.get(Model.MINMAX):
            steps.append(('minmax_scale', MinMaxScaler()))

        # add the classifier as the final step in the pipeline
        steps.append((self.method, classifier))
        self.pipeline = Pipeline(steps)

# --- from HDI-Project__ATM::atm/classifier.py::Model._special_conversions ---
def _special_conversions(self, params):
        # TODO: replace this logic with something better
        # create list parameters
        lists = defaultdict(list)
        element_regex = re.compile(r'(.*)\[(\d)\]')
        for name, param in list(params.items()):
            # look for variables of the form "param_name[1]"
            match = element_regex.match(name)
            if match:
                # name of the list parameter
                lname = match.groups()[0]
                # index of the list item
                index = int(match.groups()[1])
                lists[lname].append((index, param))

                # drop the element parameter from our list
                del params[name]

        for lname, items in list(lists.items()):
            # drop the list size parameter
            del params['len(%s)' % lname]

            # sort the list by index
            params[lname] = [val for idx, val in sorted(items)]

        # Gaussian process classifier
        if self.method == "gp":
            if params["kernel"] == "constant":
                params["kernel"] = ConstantKernel()
            elif params["kernel"] == "rbf":
                params["kernel"] = RBF()
            elif params["kernel"] == "matern":
                params["kernel"] = Matern(nu=params["nu"])
                del params["nu"]
            elif params["kernel"] == "rational_quadratic":
                params["kernel"] = RationalQuadratic(length_scale=params["length_scale"],
                                                     alpha=params["alpha"])
                del params["length_scale"]
                del params["alpha"]
            elif params["kernel"] == "exp_sine_squared":
                params["kernel"] = ExpSineSquared(length_scale=params["length_scale"],
                                                  periodicity=params["periodicity"])
                del params["length_scale"]
                del params["periodicity"]

        # return the updated parameter vector
        return params
