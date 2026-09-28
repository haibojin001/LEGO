# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg173::numpy.abs+numpy.divide+numpy.errstate
# name: numpy_primitive
# summary: Uses numpy.abs, numpy.divide, numpy.errstate, numpy.nan_to_num across 2 repos
# anchor_symbols: ['numpy.abs', 'numpy.divide', 'numpy.errstate', 'numpy.nan_to_num', 'numpy.sign', 'numpy.sum']
# observed in 2 repos: ['annoviko__pyclustering', 'mwaskom__seaborn']...

# --- from mwaskom__seaborn::seaborn/_core/scales.py::_make_symlog_transforms.symexp ---
def symexp(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sign(x) * c * (exp(np.abs(x)) - 1)

# --- from mwaskom__seaborn::seaborn/_core/scales.py::_make_symlog_transforms.symlog ---
def symlog(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sign(x) * log(1 + np.abs(np.divide(x, c)))

# --- from annoviko__pyclustering::pyclustering/utils/metric.py::canberra_distance_numpy ---
def canberra_distance_numpy(object1, object2):
    """!
    @brief Calculate Canberra distance between two objects using numpy.

    @param[in] object1 (array_like): The first vector.
    @param[in] object2 (array_like): The second vector.

    @return (float) Canberra distance between two objects.

    """
    with numpy.errstate(divide='ignore', invalid='ignore'):
        result = numpy.divide(numpy.abs(object1 - object2), numpy.abs(object1) + numpy.abs(object2))

    if len(result.shape) > 1:
        return numpy.sum(numpy.nan_to_num(result), axis=1).T
    else:
        return numpy.sum(numpy.nan_to_num(result))

# --- from annoviko__pyclustering::pyclustering/utils/metric.py::chi_square_distance_numpy ---
def chi_square_distance_numpy(object1, object2):
    """!
    @brief Calculate Chi square distance between two vectors using numpy.

    @param[in] object1 (array_like): The first vector.
    @param[in] object2 (array_like): The second vector.

    @return (float) Chi square distance between two objects.

    """
    with numpy.errstate(divide='ignore', invalid='ignore'):
        result = numpy.divide(numpy.power(object1 - object2, 2), numpy.abs(object1) + numpy.abs(object2))

    if len(result.shape) > 1:
        return numpy.sum(numpy.nan_to_num(result), axis=1).T
    else:
        return numpy.sum(numpy.nan_to_num(result))
