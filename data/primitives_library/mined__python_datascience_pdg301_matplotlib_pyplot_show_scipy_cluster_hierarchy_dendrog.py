# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg301::matplotlib.pyplot.show+scipy.cluster.hierarchy.dendrogram+scipy.cluster.hierarchy.linkage
# name: matplotlib_scipy_primitive
# summary: Uses matplotlib.pyplot.show, scipy.cluster.hierarchy.dendrogram, scipy.cluster.hierarchy.linkage across 2 repos
# anchor_symbols: ['matplotlib.pyplot.show', 'scipy.cluster.hierarchy.dendrogram', 'scipy.cluster.hierarchy.linkage']
# observed in 2 repos: ['apachecn__python_data_analysis_and_mining_action', 'lazyprogrammer__machine_learning_examples']...

# --- from apachecn__python_data_analysis_and_mining_action::chapter14/code.py::programmer_2 ---
def programmer_2():
    standardizedfile = "data/standardized.xls"
    data = pd.read_excel(standardizedfile, index_col=u"基站编号")

    Z = linkage(data, method="ward", metric="euclidean")
    P = dendrogram(Z, 0)
    plt.show()

    return P

# --- from lazyprogrammer__machine_learning_examples::unsupervised_class/hcluster.py::main ---
def main():
    D = 2 # so we can visualize it more easily
    s = 4 # separation so we can control how far apart the means are
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 900 # number of samples
    X = np.zeros((N, D))
    X[:300, :] = np.random.randn(300, D) + mu1
    X[300:600, :] = np.random.randn(300, D) + mu2
    X[600:, :] = np.random.randn(300, D) + mu3

    Z = linkage(X, 'ward')
    print("Z.shape:", Z.shape)
    # Z has the format [idx1, idx2, dist, sample_count]
    # therefore, its size will be (N-1, 4)

    # from documentation:
    # A (n-1) by 4 matrix Z is returned. At the i-th iteration,
    # clusters with indices Z[i, 0] and Z[i, 1] are combined to
    # form cluster n + i. A cluster with an index less than n
    # corresponds to one of the original observations.
    # The distance between clusters Z[i, 0] and Z[i, 1] is given
    # by Z[i, 2]. The fourth value Z[i, 3] represents the number
    # of original observations in the newly formed cluster.
    plt.title("Ward")
    dendrogram(Z)
    plt.show()

    Z = linkage(X, 'single')
    plt.title("Single")
    dendrogram(Z)
    plt.show()

    Z = linkage(X, 'complete')
    plt.title("Complete")
    dendrogram(Z)
    plt.show()
