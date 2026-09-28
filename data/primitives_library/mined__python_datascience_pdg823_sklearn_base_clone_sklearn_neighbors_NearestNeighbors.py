# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg823::sklearn.base.clone+sklearn.neighbors.NearestNeighbors
# name: sklearn_primitive
# summary: Uses sklearn.base.clone, sklearn.neighbors.NearestNeighbors across 2 repos
# anchor_symbols: ['sklearn.base.clone', 'sklearn.neighbors.NearestNeighbors']
# observed in 2 repos: ['BiomedSciAI__causallib', 'ZhiningLiu1998__imbalanced-ensemble']...

# --- from BiomedSciAI__causallib::causallib/estimation/matching.py::Matching._instantiate_nearest_neighbors_object ---
def _instantiate_nearest_neighbors_object(self):
        backend = self.knn_backend
        if backend == "sklearn":
            backend_instance = NearestNeighbors(algorithm="auto")
        elif callable(backend):
            backend_instance = backend()
            self.metric = backend_instance.metric
        elif hasattr(backend, "fit") and hasattr(backend, "kneighbors"):
            backend_instance = sk_clone(backend)
            self.metric = backend_instance.metric
        else:
            raise NotImplementedError(
                "`knn_backend` must be either an NearestNeighbors-like object,"
                " a callable returning such an object, or the string \"sklearn\"")
        backend_instance.set_params(**self._get_metric_dict())
        return backend_instance

# --- from ZhiningLiu1998__imbalanced-ensemble::imbens/utils/_validation.py::check_neighbors_object ---
def check_neighbors_object(nn_name, nn_object, additional_neighbor=0):
    """Check the objects is consistent to be a NN.

    Several methods in imbens.sampler relies on NN.
    Only KNeighborsMixin will be accepted. This utility allows for type
    checking and raise if the type is wrong.

    Parameters
    ----------
    nn_name : str
        The name associated to the object to raise an error if needed.

    nn_object : int or KNeighborsMixin,
        The object to be checked.

    additional_neighbor : int, default=0
        Sometimes, some algorithm need an additional neighbors.

    Returns
    -------
    nn_object : KNeighborsMixin
        The k-NN object.
    """
    if isinstance(nn_object, Integral):
        return NearestNeighbors(n_neighbors=nn_object + additional_neighbor)
    elif isinstance(nn_object, KNeighborsMixin):
        return clone(nn_object)
    else:
        raise_isinstance_error(nn_name, [int, KNeighborsMixin], nn_object)
