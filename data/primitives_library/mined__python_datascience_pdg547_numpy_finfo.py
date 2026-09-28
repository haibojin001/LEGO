# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg547::numpy.finfo
# name: numpy_primitive
# summary: Uses numpy.finfo across 3 repos
# anchor_symbols: ['numpy.finfo']
# observed in 3 repos: ['NannyML__nannyml', 'target__matrixprofile-ts', 'wilsonrljr__sysidentpy']...

# --- from target__matrixprofile-ts::matrixprofile/distanceProfile.py::massDistanceProfile ---
def massDistanceProfile(tsA,idx,m,tsB = None):
    """
    Returns the distance profile of a query within tsA against the time series tsB using the more efficient MASS comparison.

    Parameters
    ----------
    tsA: Time series containing the query for which to calculate the distance profile.
    idx: Starting location of the query within tsA
    m: Length of query.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    """

    selfJoin = False
    if tsB is None:
        selfJoin = True
        tsB = tsA

    query = tsA[idx:(idx+m)]
    n = len(tsB)
    distanceProfile = np.real(np.sqrt(mass(query,tsB).astype(complex)))
    if selfJoin:
        trivialMatchRange = (int(max(0,idx - np.round(m/2,0))),int(min(idx + np.round(m/2+1,0),n)))
        distanceProfile[trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

    #Both the distance profile and corresponding matrix profile index (which should just have the current index)
    return (distanceProfile,np.full(n-m+1,idx,dtype=float))

# --- from target__matrixprofile-ts::matrixprofile/utils.py::slidingDotProduct ---
def slidingDotProduct(query,ts):
    """
    Calculate the dot product between a query and all subsequences of length(query) in the timeseries ts. Note that we use Numpy's rfft method instead of fft.

    Parameters
    ----------
    query: Specific time series query to evaluate.
    ts: Time series to calculate the query's sliding dot product against.
    """

    m = len(query)
    n = len(ts)


    #If length is odd, zero-pad time time series
    ts_add = 0
    if n%2 ==1:
        ts = np.insert(ts,0,0)
        ts_add = 1

    q_add = 0
    #If length is odd, zero-pad query
    if m%2 == 1:
        query = np.insert(query,0,0)
        q_add = 1

    #This reverses the array
    query = query[::-1]


    query = np.pad(query,(0,n-m+ts_add-q_add),'constant')

    #Determine trim length for dot product. Note that zero-padding of the query has no effect on array length, which is solely determined by the longest vector
    trim = m-1+ts_add

    dot_product = fft.irfft(fft.rfft(ts)*fft.rfft(query))


    #Note that we only care about the dot product results from index m-1 onwards, as the first few values aren't true dot products (due to the way the FFT works for dot products)
    return dot_product[trim :]

# --- from NannyML__nannyml::nannyml/performance_estimation/direct_loss_estimation/metrics.py::MAPE._fit ---
def _fit(self, reference_data: pd.DataFrame):
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data, [self.y_true, self.y_pred]
        )
        if empty:
            raise InvalidReferenceDataException(
                f"Cannot fit DLE for {self.display_name}, too many missing values for predictions and targets."
            )

        y_true = reference_data[self.y_true]
        y_pred = reference_data[self.y_pred]

        self._sampling_error_components = mape_sampling_error_components(
            y_true_reference=y_true, y_pred_reference=y_pred
        )

        epsilon = np.finfo(np.float64).eps
        observation_level_metric = abs(y_true - y_pred) / (
            np.maximum(epsilon, abs(y_true))
        )

        self._dee_model = self._train_direct_error_estimation_model(
            X_train=reference_data[self.feature_column_names + [self.y_pred]],
            y_train=observation_level_metric,
            tune_hyperparameters=self.tune_hyperparameters,
            hyperparameter_tuning_config=self.hyperparameter_tuning_config,
            hyperparameters=self.hyperparameters,
            categorical_column_names=self.categorical_column_names,
        )

# --- from wilsonrljr__sysidentpy::sysidentpy/model_structure_selection/sobolev_orthogonal_forward_regression.py::UOFR.sobolev_error_reduction_ratio ---
def sobolev_error_reduction_ratio(
        self,
        psi: np.ndarray,
        y: np.ndarray,
        process_term_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute ERR on the ULS-augmented regression problem.

        Implements Steps 5-7 of the UOFR algorithm: compute ERR significance
        for each term using the augmented ULS matrices and select terms in
        a forward greedy manner.

        The ERR is computed as (Eq. 33 in the paper):
            ERR(phi_k) = <w_k, y>^2 / (<w_k, w_k> * <y, y>)

        where w_k is the orthogonalized regressor.

        Parameters
        ----------
        psi : np.ndarray
            Original regressor matrix of shape (N, num_terms).
        y : np.ndarray
            Output signal of shape (N, 1).
        process_term_number : int
            Maximum number of terms to select.

        Returns
        -------
        err : np.ndarray
            ERR values for each selected term (Eq. 33 computed on ULS problem).
        piv : np.ndarray
            Indices of selected terms in order of selection.
        psi_orthogonal : np.ndarray
            Augmented regressor matrix with selected columns.
        y_augmented : np.ndarray
            Augmented output vector.

        Notes
        -----
        The ERR is computed using the augmented matrices (Y_ULS, Phi_ULS),
        which means term significance is evaluated considering both the
        original fit and the derivative fits. This is the key difference
        from standard OFR: terms that appear significant under L2 may be
        less significant under the Sobolev norm, and vice versa.

        The orthogonalization uses Householder reflections for numerical
        stability, as mentioned in the paper (any orthogonalization method
        is valid, but Householder is preferred for large problems).
        """
        y_target = y[self.max_lag :, 0].reshape(-1, 1)
        y_augmented, psi_augmented = self.augment_uls_terms(
            y_target, psi, self.sobolev_order
        )
        y_augmented = y_augmented.reshape(-1, 1)
        squared_y = np.dot(y_augmented.T, y_augmented)
        squared_y = float(np.maximum(squared_y, np.finfo(np.float64).eps))
        psi_working = psi_augmented.copy()
        y_working = y_augmented.copy()
        num_terms = psi_working.shape[1]
        piv = np.arange(num_terms)
        candidate_err = np.zeros(num_terms)
        err = np.zeros(num_terms)

        for step_idx in np.arange(0, num_terms):
            candidate_err[step_idx:] = _compute_err_slice(
                psi_working,
                y_working,
                step_idx,
                squared_y,
                self.alpha,
                self.eps,
            )

            max_err_idx = np.argmax(candidate_err[step_idx:]) + step_idx
            err[step_idx] = candidate_err[max_err_idx]

            if step_idx == process_term_number:
                break

            if (self.err_tol is not None) and (err.cumsum()[step_idx] >= self.err_tol):
                self.n_terms = step_idx + 1
                process_term_number = step_idx + 1
                break

            psi_working[:, [max_err_idx, step_idx]] = psi_working[
                :, [step_idx, max_err_idx]
            ]
            piv[[max_err_idx, step_idx]] = piv[[step_idx, max_err_idx]]

            reflector = house(psi_working[step_idx:, step_idx])
            row_result = rowhouse(psi_working[step_idx:, step_idx:], reflector)
            y_working[step_idx:] = rowhouse(y_working[step_idx:], reflector)
            psi_working[step_idx:, step_idx:] = np.copy(row_result)

        tmp_piv = piv[0:process_term_number]
        psi_orthogonal = psi_augmented[:, tmp_piv]

        return err, tmp_piv, psi_orthogonal, y_augmented
