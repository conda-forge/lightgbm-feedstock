"""
A simple test for LightGBM based on scikit-learn.

Tests are not shipped with the source distribution so we include a simple
functional test here that is adapted from:

    https://github.com/Microsoft/LightGBM/blob/master/tests/python_package_test/test_sklearn.py

"""

import platform
import unittest

import lightgbm as lgb

from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split


def make_synthetic_regression(
    n_samples=100, n_features=4, n_informative=2, random_state=42
):
    return make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=random_state,
    )


class TestSklearn(unittest.TestCase):
    def test_binary(self):
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        gbm = lgb.LGBMClassifier(n_estimators=50, verbose=-1)
        gbm.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(5)],
        )
        ret = log_loss(y_test, gbm.predict_proba(X_test))
        self.assertLess(ret, 0.12)
        self.assertAlmostEqual(
            ret,
            gbm.evals_result_['valid_0']['binary_logloss'][gbm.best_iteration_ - 1],
            places=5,
        )

    def test_regression(self):
        X, y = make_synthetic_regression()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        gbm = lgb.LGBMRegressor(n_estimators=50, verbose=-1)
        gbm.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(5)],
        )
        ret = mean_squared_error(y_test, gbm.predict(X_test))
        self.assertLess(ret, 174)
        self.assertAlmostEqual(
            ret, gbm.evals_result_['valid_0']['l2'][gbm.best_iteration_ - 1], places=4
        )


if __name__ == '__main__':
    if platform.machine() != "ppc64le":
        unittest.main()
