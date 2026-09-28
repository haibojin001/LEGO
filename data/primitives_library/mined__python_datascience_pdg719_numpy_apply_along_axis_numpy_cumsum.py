# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg719::numpy.apply_along_axis+numpy.cumsum
# name: numpy_primitive
# summary: Uses numpy.apply_along_axis, numpy.cumsum across 2 repos
# anchor_symbols: ['numpy.apply_along_axis', 'numpy.cumsum']
# observed in 2 repos: ['target__matrixprofile-ts', 'xorbitsai__xorbits']...

# --- from target__matrixprofile-ts::matrixprofile/fluss.py::fluss ---
def fluss(mpi, m=None):
    """
    Returns the corrected arc curve (CAC) for the matrix profile index (MPI).
    The FLUSS algorithm provides Fast Low-cost Unipotent Semantic Segmantation.

    Parameters
    ----------
    mpi: Matrix profile index accompanying a time series.
    m: Subsequence length that was used to compute the MPI. Note: leaving this empty omits the correction at the head
    and tail of the CAC.
    """
    n = len(mpi)
    nnmark = np.zeros(n)

    # find the number of additional arcs starting to cross over each index
    for i in range(0, n):
        mpi_val = mpi[i]
        small = int(min(i, mpi_val))
        large = int(max(i, mpi_val))
        nnmark[small + 1] = nnmark[small + 1] + 1
        nnmark[large] = nnmark[large] - 1

    # cumulatively sum all crossing arcs at each index
    cross_count = np.cumsum(nnmark)

    # compute ideal arc curve for all indices
    idealized = np.apply_along_axis(lambda i: _idealized_arc_curve(n, i), 0, np.arange(0, n))
    idealized = cross_count / idealized

    # correct the arc curve so that it is between 0 and 1
    idealized[idealized > 1] = 1
    corrected_arc_curve = idealized

    if m:
        corrected_arc_curve[:m] = 1
        corrected_arc_curve[-m:] = 1

    return corrected_arc_curve

# --- from xorbitsai__xorbits::python/xorbits/_mars/tensor/base/unique.py::TensorUnique._execute_agg_reduce ---
def _execute_agg_reduce(cls, ctx, op: "TensorUnique"):
        input_indexes, input_data = zip(*list(op.iter_mapper_data(ctx)))

        inputs = list(zip(*input_data))
        flatten, device_id, xp = as_same_device(
            list(itertools.chain(*inputs)), device=op.device, ret_extra=True
        )
        n_ret = len(inputs[0])
        inputs = [flatten[i * n_ret : (i + 1) * n_ret] for i in range(len(inputs))]

        inputs_iter = iter(inputs)
        unique_arrays = next(inputs_iter)
        indices_arrays = next(inputs_iter) if op.return_index else None
        inverse_arrays = next(inputs_iter) if op.return_inverse else None
        counts_arrays = next(inputs_iter) if op.return_counts else None

        with device(device_id):
            ar = xp.concatenate(unique_arrays, axis=op.axis)
            result_return_inverse = op.return_inverse or op.return_counts
            axis = op.axis
            if ar.size == 0 or ar.shape[axis] == 0:
                # empty array on the axis
                results = [xp.empty(ar.shape)]
                i = 1
                for it in (op.return_index, op.return_inverse, op.return_counts):
                    if it:
                        results.append(xp.empty([], dtype=op.outputs[i].dtype))
                        i += 1
                results = tuple(results)
            else:
                results = xp.unique(
                    ar,
                    return_index=op.return_index,
                    return_inverse=result_return_inverse,
                    axis=axis,
                )
            results = (results,) if not isinstance(results, tuple) else results
            results_iter = iter(results)
            outputs_iter = iter(op.outputs)
            # unique array
            ctx[next(outputs_iter).key] = next(results_iter)

            if op.output_limit == 1:
                return

            # calc indices
            if op.return_index:
                ctx[next(outputs_iter).key] = xp.concatenate(indices_arrays)[
                    next(results_iter)
                ]
            # calc inverse
            try:
                inverse_result = next(results_iter)
                if op.return_inverse:
                    unique_sizes = tuple(ua.shape[op.axis] for ua in unique_arrays)
                    cum_unique_sizes = np.cumsum((0,) + unique_sizes)
                    indices_out_key = next(outputs_iter).key
                    for i, inverse_array in enumerate(inverse_arrays):
                        p = inverse_result[
                            cum_unique_sizes[i] : cum_unique_sizes[i + 1]
                        ]
                        r = xp.empty(inverse_array.shape, dtype=inverse_array.dtype)
                        if inverse_array.size > 0:
                            r[0] = inverse_array[0]
                            r[1] = p[inverse_array[1]]
                        # return unique length and
                        ctx[indices_out_key, (input_indexes[i][op.axis],)] = (
                            results[0].shape[op.axis],
                            r,
                        )
                # calc counts
                if op.return_counts:
                    result_counts = xp.zeros(results[0].shape[op.axis], dtype=int)
                    t = np.stack([inverse_result, np.concatenate(counts_arrays)])

                    def acc(a):
                        i, v = a
                        result_counts[i] += v

                    np.apply_along_axis(acc, 0, t)
                    ctx[next(outputs_iter).key] = xp.asarray(result_counts)
            except StopIteration:
                pass
