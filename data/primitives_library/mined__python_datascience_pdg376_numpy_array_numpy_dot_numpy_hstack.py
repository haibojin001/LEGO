# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg376::numpy.array+numpy.dot+numpy.hstack
# name: numpy_primitive
# summary: Uses numpy.array, numpy.dot, numpy.hstack, numpy.ones across 2 repos
# anchor_symbols: ['numpy.array', 'numpy.dot', 'numpy.hstack', 'numpy.ones', 'numpy.zeros']
# observed in 2 repos: ['rasbt__mlxtend', 'rnorm__book_sample']...

# --- from rnorm__book_sample::code/chapter4/linear_regression.py::LR.predict ---
def predict(self, x):
        return np.dot(np.hstack((np.ones((x.shape[0], 1)), x)), self.coef)

# --- from rnorm__book_sample::code/chapter4/linear_regression_ridge.py::LR_Ridge.predict ---
def predict(self, x):
        new_x_transformed = self.scaler.transform(x)
        new_x_transformed = np.hstack(
            (np.ones((x.shape[0],1)), new_x_transformed)
        )
        return np.dot(new_x_transformed, self.coef)

# --- from rasbt__mlxtend::mlxtend/classifier/adaline.py::Adaline._normal_equation ---
def _normal_equation(self, X, y):
        """Solve linear regression analytically."""
        Xb = np.hstack((np.ones((X.shape[0], 1)), X))
        w = np.zeros(X.shape[1])
        z = np.linalg.inv(np.dot(Xb.T, Xb))
        params = np.dot(z, np.dot(Xb.T, y))
        b, w = np.array([params[0]]), params[1:].reshape(X.shape[1], 1)
        return b, w

# --- from rasbt__mlxtend::mlxtend/regressor/linear_regression.py::LinearRegression._normal_equation ---
def _normal_equation(self, X, y):
        """Solve linear regression analytically."""
        Xb = np.hstack((np.ones((X.shape[0], 1)), X))
        w = np.zeros(X.shape[1])
        z = np.linalg.inv(np.dot(Xb.T, Xb))
        params = np.dot(z, np.dot(Xb.T, y))
        b, w = np.array([params[0]]), params[1:].reshape(X.shape[1], 1)
        return b, w
