# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg275::numpy.greater_equal+numpy.less+numpy.partition
# name: numpy_sklearn_primitive
# summary: Uses numpy.greater_equal, numpy.less, numpy.partition, numpy.where across 2 repos
# anchor_symbols: ['numpy.greater_equal', 'numpy.less', 'numpy.partition', 'numpy.where', 'sklearn.utils.column_or_1d']
# observed in 2 repos: ['yzhao062__combo', 'yzhao062__pyod']...

# --- from yzhao062__pyod::pyod/utils/utility.py::argmaxn ---
def argmaxn(value_list, n, order='desc'):
    """Return the index of top n elements in the list
    if order is set to 'desc', otherwise return the index of n smallest ones.

    Parameters
    ----------
    value_list : list, array, numpy array of shape (n_samples,)
        A list containing all values.

    n : int
        The number of elements to select.

    order : str, optional (default='desc')
        The order to sort {'desc', 'asc'}:

        - 'desc': descending
        - 'asc': ascending

    Returns
    -------
    index_list : numpy array of shape (n,)
        The index of the top n elements.
    """

    value_list = column_or_1d(value_list)
    length = len(value_list)

    # validate the choice of n
    check_parameter(n, 1, length, include_left=True, include_right=True,
                    param_name='n')

    # for the smallest n, flip the value
    if order != 'desc':
        n = length - n

    value_sorted = np.partition(value_list, length - n)
    threshold = value_sorted[int(length - n)]

    if order == 'desc':
        return np.where(np.greater_equal(value_list, threshold))[0]
    else:  # return the index of n smallest elements
        return np.where(np.less(value_list, threshold))[0]

# --- from yzhao062__combo::combo/utils/utility.py::argmaxn ---
def argmaxn(value_list, n, order='desc'):
    """Return the index of top n elements in the list
    if order is set to 'desc', otherwise return the index of n smallest ones.

    Parameters
    ----------
    value_list : list, array, numpy array of shape (n_samples,)
        A list containing all values.

    n : int
        The number of elements to select.

    order : str, optional (default='desc')
        The order to sort {'desc', 'asc'}:

        - 'desc': descending
        - 'asc': ascending

    Returns
    -------
    index_list : numpy array of shape (n,)
        The index of the top n elements.
    """

    value_list = column_or_1d(value_list)
    length = len(value_list)

    # validate the choice of n
    check_parameter(n, 1, length, include_left=True, include_right=True,
                    param_name='n')

    # for the smallest n, flip the value
    if order != 'desc':
        n = length - n

    value_sorted = np.partition(value_list, length - n)
    threshold = value_sorted[int(length - n)]

    if order == 'desc':
        return np.where(np.greater_equal(value_list, threshold))[0]
    else:  # return the index of n smallest elements
        return np.where(np.less(value_list, threshold))[0]
