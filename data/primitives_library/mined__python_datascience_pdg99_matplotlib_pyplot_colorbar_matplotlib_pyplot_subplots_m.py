# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg99::matplotlib.pyplot.colorbar+matplotlib.pyplot.subplots+matplotlib.pyplot.title
# name: matplotlib_primitive
# summary: Uses matplotlib.pyplot.colorbar, matplotlib.pyplot.subplots, matplotlib.pyplot.title across 2 repos
# anchor_symbols: ['matplotlib.pyplot.colorbar', 'matplotlib.pyplot.subplots', 'matplotlib.pyplot.title']
# observed in 2 repos: ['piskvorky__gensim', 'ploomber__sklearn-evaluation']...

# --- from piskvorky__gensim::docs/src/auto_examples/howtos/run_compare_lda.py::plot_difference_matplotlib ---
def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """Helper function to plot difference between models.

    Uses matplotlib as the backend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='RdBu_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)

# --- from piskvorky__gensim::docs/src/gallery/howtos/run_compare_lda.py::plot_difference_matplotlib ---
def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """Helper function to plot difference between models.

    Uses matplotlib as the backend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='RdBu_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)

# --- from ploomber__sklearn-evaluation::src/sklearn_evaluation/plot/grid_search.py::_grid_search_double ---
def _grid_search_double(grid_scores, change, subset, cmap, ax, sort):
    if ax is None:
        _, ax = plt.subplots()

    # check that the two different parameters were passed
    if len(set(change)) == 1:
        raise ValueError("You need to pass two different parameters")

    # if a value in subset was passed, use it to filter the groups
    if subset is not None:
        groups = _group_by(grid_scores, _get_params_value(subset.keys()))
        keys = _mapping_to_tuple_pairs(subset)
        groups = {k: v for k, v in _sorted_map_iter(groups, sort) if k in keys}
        grid_scores = _flatten_list(groups.values())
        if not groups:
            raise ValueError(
                (
                    "Your subset didn't match any data"
                    " verify that the values are correct."
                )
            )

    # group by every possible combination in change
    matrix_elements = _group_by(grid_scores, _get_params_value(change))

    for k, v in matrix_elements.items():
        if len(v) > 1:
            raise ValueError(
                (
                    "More than one result matched your criteria."
                    " Make sure you specify parameters using change"
                    " and subset so only one group matches."
                )
            )

    # on each group there must be only one element, get it
    matrix_elements = {k: v[0] for k, v in matrix_elements.items()}

    # get the unique values for each element
    # and sort the results to make sure the matrix axis
    # is showed in increasing order
    row_names = sorted(
        set([t[0] for t in matrix_elements.keys()]), key=lambda x: (x[1] is None, x[1])
    )
    col_names = sorted(
        set([t[1] for t in matrix_elements.keys()]), key=lambda x: (x[1] is None, x[1])
    )

    # size of the matrix
    cols = len(col_names)
    rows = len(row_names)

    # map values to coordinates to populate the marix
    x_coord = {k: v for k, v in zip(col_names, range(cols))}
    y_coord = {k: v for k, v in zip(row_names, range(rows))}

    # replace keys in matrix_elements with their corresponding indexes
    # improve variable naming
    m = {(x_coord[k[1]], y_coord[k[0]]): v for k, v in matrix_elements.items()}

    matrix = np.zeros((rows, cols))

    for (j, i), v in m.items():
        matrix[i][j] = v.mean_validation_score

    # ticks for the axis
    row_labels = ["{}={}".format(*x) for x in row_names]
    col_labels = ["{}={}".format(*y) for y in col_names]

    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)

    # set text on cells
    for (x, y), v in m.items():
        label = "{:.3}".format(v.mean_validation_score)
        ax.text(x, y, label, horizontalalignment="center", verticalalignment="center")

    ax.set_xticks(range(cols))
    ax.set_xticklabels(col_labels, rotation=45)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(row_labels)
    plt.colorbar(im, ax=ax)
    ax.get_figure().tight_layout()
    return ax
