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


def test_separated_yes_no_fit_does_not_overshoot():
    """Sixteen units, an outcome that a window of one unit property decides almost exactly: full
    steps of the iterated fit overshoot and alternate between two bad fits (the fit was worse
    than the training mean by a factor of seven). Shortened steps must end in a fit that beats
    the mean on its own training rows."""
    from transferledger.curves import FAMILIES

    rng = np.random.default_rng(0)
    units, per = 16, 5
    x, z, q = rng.uniform(20, 40, units), rng.normal(size=units), rng.normal(size=units)
    u, r = np.repeat(np.arange(units), per), np.tile(np.arange(per), units)
    inside = (x > 25) & (x < 31.5)
    y = (inside[u] & (rng.random(len(u)) < 0.9)).astype(float)
    X = np.column_stack([x[u], z[u], q[u]] + [(r == k) for k in range(1, per)]).astype(float)
    m = SplineAdditive(family="binomial").fit(X, y)
    fam = FAMILIES["binomial"]
    mu = np.clip(m.predict(X), 1e-3, 1 - 1e-3)
    assert fam.loss(y, mu).sum() < fam.loss(y, np.full(len(y), y.mean())).sum()
    assert m.damped_


def test_weights_of_a_separating_piece_stay_finite():
    from transferledger.curves import FAMILIES
    from transferledger.model import fit_weights, linear_score

    rng = np.random.default_rng(1)
    P = np.column_stack([np.repeat([0.0, 1.0], 40), rng.normal(size=80)])
    y = P[:, 0].copy()  # the first piece separates the classes
    fam = FAMILIES["binomial"]
    coef = fit_weights(P, y, np.ones(80), fam)
    mu = fam.mean(linear_score(P, coef))
    assert np.all(np.isfinite(coef))
    assert ((mu > 0.5) == (y == 1)).all()


def test_a_stalled_separated_fit_ends():
    """Twenty reactions under a condition that never gives product, among 600 of which one in
    eleven does: the weight of that condition can only creep toward minus infinity. A shortened
    step that gains nothing must end the fit. It ran on to the limit of 200 steps before, for
    a change of under 1e-6 in any probability, and an outside table with such conditions took
    ten times as long."""
    from transferledger.curves import FAMILIES
    from transferledger.model import fit_weights, linear_score

    class Counting:
        def __init__(self, fam):
            self.fam, self.steps = fam, 0

        def __getattr__(self, name):
            return getattr(self.fam, name)

        def working(self, y, eta):
            self.steps += 1
            return self.fam.working(y, eta)

    rng = np.random.default_rng(0)
    dead = np.repeat([0.0, 1.0], [580, 20])
    y = ((dead == 0) & (rng.random(600) < 0.09)).astype(float)
    P = np.column_stack([dead, rng.normal(size=600)])
    fam = Counting(FAMILIES["binomial"])
    coef = fit_weights(P, y, np.ones(600), fam)
    mu = fam.mean(linear_score(P, coef))
    assert fam.steps < 40
    assert mu[dead == 1].max() < 1e-3
    assert abs(mu[dead == 0].mean() - y[dead == 0].mean()) < 1e-3
