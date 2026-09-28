# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg276::matplotlib.pyplot.axis+matplotlib.pyplot.figure+matplotlib.pyplot.legend
# name: matplotlib_primitive
# summary: Uses matplotlib.pyplot.axis, matplotlib.pyplot.figure, matplotlib.pyplot.legend, matplotlib.pyplot.savefig across 2 repos
# anchor_symbols: ['matplotlib.pyplot.axis', 'matplotlib.pyplot.figure', 'matplotlib.pyplot.legend', 'matplotlib.pyplot.savefig', 'matplotlib.pyplot.scatter', 'matplotlib.pyplot.show', 'matplotlib.pyplot.suptitle', 'matplotlib.pyplot.title', 'matplotlib.pyplot.xticks', 'matplotlib.pyplot.yticks']
# observed in 2 repos: ['dmbee__seglearn', 'yzhao062__pyod']...

# --- from dmbee__seglearn::examples/plot_lda.py::plot_embedding ---
def plot_embedding(emb, y, y_labels):
    # plot a 2D feature map embedding
    x_min, x_max = np.min(emb, 0), np.max(emb, 0)
    emb = (emb - x_min) / (x_max - x_min)

    NC = len(y_labels)
    markers = ['.', '+', 'x', '|', '_', '*', 'o']

    fig = plt.figure()
    fig.set_size_inches(6, 6)

    for c in range(NC):
        i = y == c
        plt.scatter(emb[i, 0], emb[i, 1], marker=markers[c], label=y_labels[c])

    plt.xticks([]), plt.yticks([])
    plt.legend()
    plt.tight_layout()

# --- from yzhao062__pyod::pyod/utils/example.py::data_visualize._plot ---
def _plot(X_inliers, X_outliers, inlier_color='blue',
              outlier_color='orange'):
        """Internal method to add subplot of inliers and outliers.

        Parameters
        ----------
        X_inliers : numpy array of shape (n_samples, n_features)
            Outliers.

        X_outliers : numpy array of shape (n_samples, n_features)
            Inliers.

        sub_plot_title : str
            Subplot title.

        inlier_color : str, optional (default='blue')
            The color of inliers.

        outlier_color : str, optional (default='orange')
            The color of outliers.

        """
        plt.axis("equal")
        plt.scatter(X_inliers[:, 0], X_inliers[:, 1], label='inliers',
                    color=inlier_color, s=40)
        plt.scatter(X_outliers[:, 0], X_outliers[:, 1],
                    label='outliers', color=outlier_color, s=50, marker='^')
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc='best', prop={'size': 10})

# --- from yzhao062__pyod::pyod/utils/example.py::visualize._add_sub_plot ---
def _add_sub_plot(X_inliers, X_outliers, sub_plot_title,
                      inlier_color='blue', outlier_color='orange'):
        """Internal method to add subplot of inliers and outliers.

        Parameters
        ----------
        X_inliers : numpy array of shape (n_samples, n_features)
            Outliers.

        X_outliers : numpy array of shape (n_samples, n_features)
            Inliers.

        sub_plot_title : str
            Subplot title.

        inlier_color : str, optional (default='blue')
            The color of inliers.

        outlier_color : str, optional (default='orange')
            The color of outliers.

        """
        plt.axis("equal")
        plt.scatter(X_inliers[:, 0], X_inliers[:, 1], label='inliers',
                    color=inlier_color, s=40)
        plt.scatter(X_outliers[:, 0], X_outliers[:, 1],
                    label='outliers', color=outlier_color, s=50, marker='^')
        plt.title(sub_plot_title, fontsize=15)
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc=3, prop={'size': 10})
