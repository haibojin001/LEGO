import numpy as np

from sklearn.model_selection._search import BaseSearchCV, is_classifier
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_array, validate_data

from hyperopt.base import STATUS_OK, Trials
from hyperopt.fmin import fmin


class HyperoptSearchCV(BaseSearchCV):
    _required_parameters = ["estimator", "space", "max_evals"]

    def __init__(
        self,
        estimator,
        space,
        max_evals,
        trials=None,
        algo=None,
        warm_start=False,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        BaseSearchCV.__init__(
            self,
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.space = space
        self.max_evals = max_evals
        self.trials = trials
        self.algo = algo
        self.warm_start = warm_start
        self.random_state = random_state

    def _check_input_parameters(self, X, y=None, groups=None, split_params=None):
        if self.scoring is not None and not (
            isinstance(self.scoring, str) or callable(self.scoring)
        ):
            raise ValueError(
                "scoring parameter must be a string, "
                "a callable or None. Multimetric scoring is not supported."
            )

        check_array(X, accept_sparse=True)

        if y is not None:
            if is_classifier(self.estimator):
                y = validate_data(self, X="no_validation", y=y, multi_output=True)
                check_classification_targets(y)
                if len(unique_labels(y)) < 2:
                    raise ValueError(
                        "Classifier can't train when only one class is present."
                    )
            else:
                y = validate_data(self, X="no_validation", y=y)

        if not isinstance(self.refit, bool):
            raise ValueError(
                f"refit is expected to be a boolean. Got {type(self.refit)} instead."
            )

    def fit(self, X, y=None, groups=None, **fit_params):
        self._check_input_parameters(X=X, y=y, groups=groups)
        BaseSearchCV.fit(self, X, y=y, groups=groups, **fit_params)
        return self

    def _run_search(self, evaluate_candidates):
        def objective(parameters):
            candidate_results = evaluate_candidates([parameters])
            return {
                "loss": -candidate_results["mean_test_score"][-1],
                "params": parameters,
                "status": STATUS_OK,
            }

        if self.warm_start:
            self.trials_ = Trials() if self.trials is None else self.trials
        else:
            self.trials_ = Trials()

        fmin(
            fn=objective,
            space=self.space,
            algo=self.algo,
            max_evals=self.max_evals,
            trials=self.trials_,
            rstate=self.random_state,
        )