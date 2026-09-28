# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg289::numpy.ndim+numpy.shape
# name: numpy_primitive
# summary: Uses numpy.ndim, numpy.shape across 2 repos
# anchor_symbols: ['numpy.ndim', 'numpy.shape']
# observed in 2 repos: ['HunterMcGushion__hyperparameter_hunter', 'tflearn__tflearn']...

# --- from tflearn__tflearn::tflearn/utils.py::prepare_X ---
def prepare_X(X, target_ndim, max_dim=None, min_dim=None, debug_msg="Data"):

    # Validate the dimension
    validate_dim(X, max_dim, min_dim)

    X_ndim = np.ndim(X)
    # Reshape to the desired dimension
    if X_ndim < target_ndim:
        for i in range(target_ndim - X_ndim):
            try:
                X = np.expand_dims(X, axis=0)
            except Exception:
                raise Exception(debug_msg + " shape mismatch (too few dimensions).")
    elif X_ndim > target_ndim:
        for i in range(X_ndim - target_ndim):
            try:
                X = np.reshape(X, newshape=np.shape(X)[:-1])
            except Exception:
                raise Exception(debug_msg +  " shape mismatch (too many dimensions).")
    return X, X_ndim

# --- from HunterMcGushion__hyperparameter_hunter::hyperparameter_hunter/optimization/backends/skopt/engine.py::Optimizer._check_y_is_valid ---
def _check_y_is_valid(self, x, y):
        """Check if the shapes and types of `x` and `y` are consistent. Complains if anything
        is weird about `y`"""
        #################### Per-Second Acquisition Function ####################
        if self.acq_func.endswith("ps"):
            if is_2d_list_like(x):
                if not (np.ndim(y) == 2 and np.shape(y)[1] == 2):
                    raise TypeError("Expected `y` to be a list of (func_val, t)")
            elif is_list_like(x):
                if not (np.ndim(y) == 1 and len(y) == 2):
                    raise TypeError("Expected `y` to be (func_val, t)")

        #################### Standard Acquisition Function ####################
        # If `y` isn't a scalar, we have been handed a batch of points
        elif is_list_like(y) and is_2d_list_like(x):
            for y_value in y:
                if not isinstance(y_value, Number):
                    raise ValueError("Expected `y` to be a list of scalars")
        elif is_list_like(x):
            if not isinstance(y, Number):
                raise ValueError("`func` should return a scalar")
        else:
            raise ValueError(f"Incompatible argument types: `x` ({type(x)}) and `y` ({type(y)})")
