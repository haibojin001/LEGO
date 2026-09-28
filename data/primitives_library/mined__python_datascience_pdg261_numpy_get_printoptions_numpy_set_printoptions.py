# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg261::numpy.get_printoptions+numpy.set_printoptions
# name: numpy_primitive
# summary: Uses numpy.get_printoptions, numpy.set_printoptions across 3 repos
# anchor_symbols: ['numpy.get_printoptions', 'numpy.set_printoptions']
# observed in 3 repos: ['modin-project__modin', 'yzhao062__combo', 'yzhao062__pyod']...

# --- from modin-project__modin::modin/tests/numpy/test_array.py::change_numpy_print_threshold ---
def change_numpy_print_threshold():
    prev_threshold = numpy.get_printoptions()["threshold"]
    numpy.set_printoptions(threshold=50)
    yield prev_threshold
    numpy.set_printoptions(threshold=prev_threshold)

# --- from modin-project__modin::modin/numpy/arr.py::array.__repr__ ---
def __repr__(self):
        # If we are dealing with a small array, we can just collate all the data on the
        # head node and let numpy handle the logic to get a string representation.
        if self.size <= numpy.get_printoptions()["threshold"]:
            return repr(self._to_numpy())
        arr = self._build_repr_array()
        prev_threshold = numpy.get_printoptions()["threshold"]
        numpy.set_printoptions(threshold=arr.size - 1)
        try:
            repr_str = repr(arr)
        finally:
            numpy.set_printoptions(threshold=prev_threshold)
        return repr_str

# --- from yzhao062__combo::combo/models/sklearn_base.py::_pprint ---
def _pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'
    Parameters
    ----------
    params : dict
        The dictionary to pretty print
    offset : int
        The offset in characters to add at the begin of each line.
    printer : callable
        The function to convert entries to strings, typically
        the builtin str or repr
    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(params.items())):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines

# --- from yzhao062__pyod::pyod/models/sklearn_base.py::_pprint ---
def _pprint(params, offset=0, printer=repr):
    # noinspection PyPep8
    """Pretty print the dictionary 'params'

    See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
    and sklearn/base.py for more information.

    :param params: The dictionary to pretty print
    :type params: dict

    :param offset: The offset in characters to add at the begin of each line.
    :type offset: int

    :param printer: The function to convert entries to strings, typically
        the builtin str or repr
    :type printer: callable

    :return: None
    """

    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(params.items())):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or '\n' in this_repr:
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines
