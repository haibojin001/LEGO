# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg661::matplotlib.image.imread+matplotlib.pyplot.axis+matplotlib.pyplot.imshow
# name: matplotlib_primitive
# summary: Uses matplotlib.image.imread, matplotlib.pyplot.axis, matplotlib.pyplot.imshow, matplotlib.pyplot.show across 3 repos
# anchor_symbols: ['matplotlib.image.imread', 'matplotlib.pyplot.axis', 'matplotlib.pyplot.imshow', 'matplotlib.pyplot.show']
# observed in 3 repos: ['ZhiningLiu1998__imbalanced-ensemble', 'alegonz__baikal', 'devAmoghS__Machine-Learning-with-Python']...

# --- from ZhiningLiu1998__imbalanced-ensemble::imbens/utils/_plot.py::plot_online_figure ---
def plot_online_figure(url: str = None):  # pragma: no cover
    '''Plot an online figure'''
    figure = mpimg.imread(url)
    plt.axis('off')
    plt.imshow(figure)
    plt.tight_layout()

# --- from devAmoghS__Machine-Learning-with-Python::k_means_clustering/utils.py::recolor_image ---
def recolor_image(input_file, k=5):
    img = mpimg.imread(input_file)
    pixels = [pixel for row in img for pixel in row]
    clusterer = KMeans(k)
    clusterer.train(pixels)  # this might take a while

    def recolor(pixel):
        cluster = clusterer.classify(pixel)  # index of the closest cluster
        return clusterer.means[clusterer]  # mean of the closest cluster

    new_img = [[recolor(pixel) for pixel in row] for row in img]
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()

# --- from alegonz__baikal::baikal/plot.py::plot_model ---
def plot_model(
    model: Model,
    filename: Optional[str] = None,
    show: bool = False,
    expand_nested: bool = False,
    prog: str = "dot",
    **dot_kwargs
) -> pydot.Dot:
    """Plot a model to file and/or display it.

    This function requires pydot and graphviz. It also requires matplotlib
    to display plots to the screen.

    Parameters
    ----------
    model
        The model to plot.

    filename
        Filename (optional).

    show
        Whether to display the plot in the screen or not. Requires matplotlib.

    expand_nested
        Whether to expand any nested models or not (display the nested model as
        a single step).

    prog
        Program to use to process the dot file into a graph.

    dot_kwargs
        Keyword arguments to pydot.Dot.

    Returns
    -------
    dot_graph
        Dot graph of the given model. It can be used to generate an image for plotting.

    Examples
    --------
    ::

        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from baikal.plot import plot_model

        dot_graph = plot_model(model, include_targets=True, expand_nested=False)
        png = dot_graph.create(format="png", prog=prog)
        img = mpimg.imread(io.BytesIO(png))
        plt.imshow(img, aspect="equal")
        plt.axis("off")
        plt.show()

    """

    dot_transformer = _DotTransformer(expand_nested, **dot_kwargs)
    dot_graph = dot_transformer.transform(model)

    # save plot
    if filename:
        basename, ext = os.path.splitext(filename)
        with open(filename, "wb") as fh:
            fh.write(dot_graph.create(format=ext.lstrip(".").lower(), prog=prog))

    # display graph via matplotlib
    if show:  # pragma: no cover
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        png = dot_graph.create(format="png", prog=prog)
        img = mpimg.imread(io.BytesIO(png))
        plt.imshow(img, aspect="equal")
        plt.axis("off")
        plt.show()

    return dot_graph
