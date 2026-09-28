# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg618::numpy.issubdtype+numpy.result_type
# name: numpy_primitive
# summary: Uses numpy.issubdtype, numpy.result_type across 2 repos
# anchor_symbols: ['numpy.issubdtype', 'numpy.result_type']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'xorbitsai__xorbits']...

# --- from sb-ai-lab__LightAutoML::lightautoml/dataset/np_pd_dataset.py::NumpyDataset._check_dtype ---
def _check_dtype(self):
        """Check if dtype in ``.set_data`` is ok and cast if not.

        Raises:
            AttributeError: If there is non-numeric type in dataset.

        """
        # dtypes = list(set(map(lambda x: x.dtype, self.roles.values())))
        dtypes = list(set([i.dtype for i in self.roles.values()]))
        self.dtype = np.result_type(*dtypes) if len(dtypes) else None

        for f in self.roles:
            self._roles[f].dtype = self.dtype

        assert np.issubdtype(self.dtype, np.number), "Support only numeric types in numpy dataset."

        if self.data.dtype != self.dtype:
            try:
                self.data = self.data.astype(self.dtype)
            except:
                pass

# --- from xorbitsai__xorbits::python/xorbits/_mars/tensor/statistics/histogram.py::_unsigned_subtract ---
def _unsigned_subtract(a, b):
    """
    Subtract two values where a >= b, and produce an unsigned result

    This is needed when finding the difference between the upper and lower
    bound of an int16 histogram
    """
    # coerce to a single type
    signed_to_unsigned = {
        np.byte: np.ubyte,
        np.short: np.ushort,
        np.intc: np.uintc,
        np.int_: np.uint,
        np.longlong: np.ulonglong,
    }
    dt = np.result_type(a, b)
    try:
        dt = signed_to_unsigned[dt.type]
    except KeyError:  # pragma: no cover
        return np.subtract(a, b, dtype=dt)
    else:
        # we know the inputs are integers, and we are deliberately casting
        # signed to unsigned
        return np.subtract(a, b, casting="unsafe", dtype=dt)

# --- from xorbitsai__xorbits::python/xorbits/_mars/tensor/statistics/histogram.py::_get_bin_edges ---
def _get_bin_edges(op, a, bins, range, weights):
    # parse the overloaded bins argument
    n_equal_bins = None
    bin_edges = None
    first_edge = None
    last_edge = None

    if isinstance(bins, str):
        # when `bins` is str, x.min() and x.max()
        # will be calculated in advance
        bin_name = bins
        if a.size > 0:
            assert range is not None

        raw_range = range
        first_edge, last_edge = _get_outer_edges(a, range)

        if a.size == 0:
            n_equal_bins = 1
        else:
            # Do not call selectors on empty arrays
            selector = _hist_bin_selectors[bin_name](
                op, a, (first_edge, last_edge), raw_range
            )
            yield from selector.check()
            width = selector.get_result()
            if width:
                # Reference: https://github.com/numpy/numpy/blob/v2.1.0/numpy/lib/_histograms_impl.py#L413
                if (
                    np.__version__ >= "2.1.0"
                    and np.issubdtype(a.dtype, np.integer)
                    and width < 1
                ):
                    width = 1
                n_equal_bins = int(
                    np.ceil(_unsigned_subtract(last_edge, first_edge) / width)
                )
            else:
                # Width can be zero for some estimators, e.g. FD when
                # the IQR of the data is zero.
                n_equal_bins = 1

    elif mt.ndim(bins) == 0:
        first_edge, last_edge = _get_outer_edges(a, range)
        n_equal_bins = bins

    else:
        # cannot be Tensor, must be calculated first
        assert mt.ndim(bins) == 1 and not isinstance(bins, TENSOR_TYPE)
        bin_edges = np.asarray(bins)
        if not is_asc_sorted(bin_edges):
            raise ValueError("`bins` must increase monotonically, when an array")

    if n_equal_bins is not None:
        # numpy gh-10322 means that type resolution rules are dependent on array
        # shapes. To avoid this causing problems, we pick a type now and stick
        # with it throughout.
        bin_type = np.result_type(first_edge, last_edge, a)
        if np.issubdtype(bin_type, np.integer):
            bin_type = np.result_type(bin_type, float)

        # bin edges must be computed
        bin_edges = mt.linspace(
            first_edge,
            last_edge,
            n_equal_bins + 1,
            endpoint=True,
            dtype=bin_type,
            gpu=op.gpu,
        )
        return bin_edges, (first_edge, last_edge, n_equal_bins)
    else:
        return mt.tensor(bin_edges), None
