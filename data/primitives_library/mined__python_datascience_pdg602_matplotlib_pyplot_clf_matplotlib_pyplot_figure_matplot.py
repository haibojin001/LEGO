# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg602::matplotlib.pyplot.clf+matplotlib.pyplot.figure+matplotlib.pyplot.imshow
# name: matplotlib_numpy_primitive
# summary: Uses matplotlib.pyplot.clf, matplotlib.pyplot.figure, matplotlib.pyplot.imshow, matplotlib.pyplot.plot across 2 repos
# anchor_symbols: ['matplotlib.pyplot.clf', 'matplotlib.pyplot.figure', 'matplotlib.pyplot.imshow', 'matplotlib.pyplot.plot', 'matplotlib.pyplot.scatter', 'matplotlib.pyplot.xlim', 'matplotlib.pyplot.xticks', 'matplotlib.pyplot.ylim', 'matplotlib.pyplot.yticks', 'numpy.arange', 'numpy.meshgrid', 'sklearn.decomposition.PCA']
# observed in 2 repos: ['hi-primus__optimus', 'practical-recommender-systems__moviegeek']...

# --- from practical-recommender-systems__moviegeek::builder/user_cluster_calculator.py::plot ---
def plot(user_ratings, kmeans, k):
        print("reduce dimensionality of data")
        h = 0.2
        reduced_data = PCA(n_components=2).fit_transform(user_ratings)
        print("cluster reduced data")

        if not kmeans:
            kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
            kmeans.fit(reduced_data)

        print("plot clustered reduced data")
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        centroids = kmeans.cluster_centers_
        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        plt.title('K-means clustering of the user')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())

        plt.savefig('cluster.png')

# --- from hi-primus__optimus::optimus/engines/pandas/ml/models.py::Model.plot_clusters ---
def plot_clusters(self):
        reduced_data = PCA(n_components=2).fit_transform(self.X_test)
        kmeans = self.model
        kmeans.fit(reduced_data)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation="nearest",
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired, aspect="auto", origin="lower")

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=5)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
                    color="w", zorder=10)
        # plt.title("K-means clustering on the digits dataset (PCA-reduced data)\n"
        #           "Centroids are marked with white cross")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()
