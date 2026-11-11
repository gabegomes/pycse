"""Curve sources."""

import numpy as np
import pytest

from transferledger.curves import ModelCurves, SplineAdditive
from transferledger.forms import sigmoid


def data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(22, 50, n), rng.uniform(-30, 0, n), rng.integers(0, 2, n),
                         rng.choice([90.0, 105.0, 120.0], n), rng.normal(size=n)])  # fmt: skip
    eta = (
        -1.5 * sigmoid(X[:, 0] - 32.0)
        - 0.01 * (X[:, 1] + 15.0) ** 2
        + 0.8 * X[:, 2]
        + 0.02 * X[:, 3]
    )
    return X, eta, rng


def test_spline_source_recovers_additive_curves():
    X, eta, rng = data()
    m = SplineAdditive().fit(X, eta + 0.1 * rng.normal(size=len(X)))
    c = m.curves(X)
    assert np.corrcoef(c[0], -1.5 * sigmoid(X[:, 0] - 32.0))[0, 1] > 0.99
    assert np.corrcoef(c[1], -0.01 * (X[:, 1] + 15.0) ** 2)[0, 1] > 0.99
    assert c[4].std() < 0.05  # the unrelated variable
    assert np.allclose(m.intercept_ + c.sum(axis=0), m.predict(X))
    assert np.allclose(c.mean(axis=1), 0.0, atol=1e-8)


def test_the_fit_is_repeatable():
    X, eta, _ = data()
    a, b = SplineAdditive().fit(X, eta).predict(X), SplineAdditive().fit(X, eta).predict(X)
    assert np.array_equal(a, b)


def test_binomial_family_gives_probabilities():
    X, eta, rng = data(n=800)
    y = (rng.uniform(size=len(X)) < sigmoid(2.0 * (eta - eta.mean()))).astype(float)
    m = SplineAdditive(family="binomial").fit(X, y)
    p = m.predict(X)
    assert p.min() >= 0 and p.max() <= 1
    assert np.corrcoef(m.decision_function(X), eta)[0, 1] > 0.9


def test_weights_and_poisson_run():
    X, eta, rng = data()
    y = rng.poisson(np.exp(0.5 * (eta - eta.mean())))
    m = SplineAdditive(family="poisson").fit(X, y, sample_weight=rng.uniform(0.5, 2, len(X)))
    assert np.all(m.predict(X) > 0)


def test_strengths_tuned_on_held_out_groups_shrink_group_level_noise():
    """Variables constant within a group have only as many values as there are groups; tuning
    on rows cannot see that, tuning on held-out groups can."""
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(12), 40)
    Z = rng.normal(size=(12, 6))[g]  # six group-level variables, pure noise
    x = rng.normal(size=len(g))
    y = x + 2.0 * rng.normal(size=12)[g] + 0.1 * rng.normal(size=len(g))
    X = np.column_stack([x, Z])
    from sklearn.model_selection import GroupKFold

    cv = list(GroupKFold(4).split(X, y, g))
    rows, grouped = SplineAdditive().fit(X, y), SplineAdditive().fit(X, y, cv=cv)
    assert grouped.shrinkage_ > rows.shrinkage_
    assert grouped.curves(X)[1:].std() < 0.5 * rows.curves(X)[1:].std()


def test_curves_from_any_model():
    class Model:
        def predict(self, X):
            return 2.0 * X[:, 0] + np.sin(X[:, 1])

    X = np.random.default_rng(2).normal(size=(200, 2))
    c = ModelCurves(Model()).fit(X).curves(X)
    assert np.corrcoef(c[0], X[:, 0])[0, 1] == pytest.approx(1.0)
    assert np.corrcoef(c[1], np.sin(X[:, 1]))[0, 1] == pytest.approx(1.0)
