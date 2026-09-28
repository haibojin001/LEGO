# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg294::sklearn.datasets.load_breast_cancer+sklearn.model_selection.train_test_split
# name: sklearn_primitive
# summary: Uses sklearn.datasets.load_breast_cancer, sklearn.model_selection.train_test_split across 2 repos
# anchor_symbols: ['sklearn.datasets.load_breast_cancer', 'sklearn.model_selection.train_test_split']
# observed in 2 repos: ['lazyprogrammer__machine_learning_examples', 'yzhao062__combo']...

# --- from lazyprogrammer__machine_learning_examples::svm_class/linear_svm_gradient.py::medical ---
def medical():
  data = load_breast_cancer()
  X, Y = data.data, data.target
  Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
  return Xtrain, Xtest, Ytrain, Ytest, 1e-3, 200

# --- from lazyprogrammer__machine_learning_examples::svm_class/svm_gradient.py::medical ---
def medical():
  data = load_breast_cancer()
  X, Y = data.data, data.target
  Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
  return Xtrain, Xtest, Ytrain, Ytest, rbf, 1e-3, 200

# --- from yzhao062__combo::combo/test/test_base.py::TestBASE.setUp ---
def setUp(self):
        random_state = 42
        X, y = load_breast_cancer(return_X_y=True)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.4, random_state=random_state)
