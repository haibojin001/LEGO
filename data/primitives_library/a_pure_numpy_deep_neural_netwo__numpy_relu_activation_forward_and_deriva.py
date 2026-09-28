def relu(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X