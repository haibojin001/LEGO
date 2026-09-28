# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg426::numpy.multiply+numpy.power+numpy.subtract
# name: numpy_primitive
# summary: Uses numpy.multiply, numpy.power, numpy.subtract across 3 repos
# anchor_symbols: ['numpy.multiply', 'numpy.power', 'numpy.subtract']
# observed in 3 repos: ['cerndb__dist-keras', 'deepchecks__deepchecks', 'sfu-db__dataprep']...

# --- from cerndb__dist-keras::distkeras/parameter_servers.py::ExperimentalParameterServer.handle_commit ---
def handle_commit(self, conn, addr):
        # Receive the parameters from the remote node.
        data = recv_data(conn)
        # Extract the data from the dictionary.
        r = data['residual']
        worker_id = data['worker_id']
        stale_cv = data['stale_center_variable']
        with self.mutex:
            diff_cv = np.subtract(self.center_variable, stale_cv)
            d = 1 / (self.inverse_learning_rate * np.power(diff_cv, 2) + 1)
            r = np.multiply(d, r)
            # Update the center variable.
            self.center_variable = self.center_variable + r
        # Increment the number of parameter server updates.
        self.next_update()

# --- from sfu-db__dataprep::dataprep/eda/diff/render.py::format_ov_stats ---
def format_ov_stats(stats: Dict[str, List[Any]]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """
    Render statistics information for distribution grid
    """
    # pylint: disable=too-many-locals
    nrows, ncols, npresent_cells, nrows_wo_dups, mem_use, dtypes_cnt = stats.values()
    ncells = np.multiply(nrows, ncols).tolist()

    data = {
        "Number of Variables": ncols,
        "Number of Rows": nrows,
        "Missing Cells": np.subtract(ncells, npresent_cells).astype(float).tolist(),
        "Missing Cells (%)": np.subtract(1, np.divide(npresent_cells, ncells)).tolist(),
        "Duplicate Rows": np.subtract(nrows, nrows_wo_dups).tolist(),
        "Duplicate Rows (%)": np.subtract(1, np.divide(nrows_wo_dups, nrows)).tolist(),
        "Total Size in Memory": list(map(float, mem_use)),
        "Average Row Size in Memory": np.subtract(mem_use, nrows).tolist(),
    }
    return {k: _format_values(k, v) for k, v in data.items()}, dtypes_cnt  # type: ignore

# --- from deepchecks__deepchecks::deepchecks/utils/correlation_methods.py::correlation_ratio ---
def correlation_ratio(categorical_data: Union[List, np.ndarray, pd.Series],
                      numerical_data: Union[List, np.ndarray, pd.Series],
                      ignore_mask: Union[List[bool], np.ndarray] = None) -> float:
    """
    Calculate the correlation ratio of numerical_variable to categorical_variable.

    Correlation ratio is a symmetric grouping based method that describe the level of correlation between
    a numeric variable and a categorical variable. returns a value in [0,1].
    For more information see https://en.wikipedia.org/wiki/Correlation_ratio

    Parameters
    ----------
    categorical_data: Union[List, np.ndarray, pd.Series]
        A sequence of categorical values encoded as class indices without nulls except possibly at ignored elements
    numerical_data: Union[List, np.ndarray, pd.Series]
        A sequence of numerical values without nulls except possibly at ignored elements
    ignore_mask: Union[List[bool], np.ndarray[bool]] default: None
        A sequence of boolean values indicating which elements to ignore. If None, includes all indexes.

    Returns
    -------
    float
        Representing the correlation ratio between the variables.
    """
    if ignore_mask:
        numerical_data = numerical_data[~np.asarray(ignore_mask)]
        categorical_data = categorical_data[~np.asarray(ignore_mask)]

    cat_num = int(np.max(categorical_data) + 1)
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(cat_num):
        cat_measures = numerical_data[categorical_data == i]
        n_array[i] = cat_measures.shape[0]
        y_avg_array[i] = np.average(cat_measures.astype(float))  # Cast to float to avoid error in python 3.6
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(numerical_data, y_total_avg), 2))
    if denominator == 0:
        eta = 0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta
