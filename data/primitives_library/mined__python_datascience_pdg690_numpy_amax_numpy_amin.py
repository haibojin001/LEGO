# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg690::numpy.amax+numpy.amin
# name: numpy_primitive
# summary: Uses numpy.amax, numpy.amin across 2 repos
# anchor_symbols: ['numpy.amax', 'numpy.amin']
# observed in 2 repos: ['DeepWisdom__AutoDL', 'nok__sklearn-porter']...

# --- from nok__sklearn-porter::tests/utils/__init__.py::dataset_generate_x ---
def dataset_generate_x(
    x: np.ndarray, n_samples: Optional[int] = None
) -> np.ndarray:
    """Helper function to create uniform test samples."""
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        msg = 'Two dimensional numpy array is required.'
        raise AssertionError(msg)
    if not n_samples:
        n_samples = PORTER_N_GEN_REGRESSION_TESTS
    return np.random.uniform(
        low=np.amin(x, axis=0),
        high=np.amax(x, axis=0),
        size=(n_samples, len(x[0])),
    )

# --- from DeepWisdom__AutoDL::AutoDL_ingestion_program/data_converter.py::binarization ---
def binarization (array):
	''' Takes a binary-class datafile and turn the max value (positive class) into 1 and the min into 0'''
	array = np.array(array, dtype=float) # conversion needed to use np.inf after
	if len(np.unique(array)) > 2:
		raise ValueError ("The argument must be a binary-class datafile. {} classes detected".format(len(np.unique(array))))
	
	# manipulation which aims at avoid error in data with for example classes '1' and '2'.
	array[array == np.amax(array)] = np.inf
	array[array == np.amin(array)] = 0
	array[array == np.inf] = 1
	return np.array(array, dtype=int)
