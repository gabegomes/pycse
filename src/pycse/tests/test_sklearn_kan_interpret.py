"""Tests for pycse.sklearn.kan_interpret (fast: tiny data, few iterations)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from pycse.sklearn.kan import KAN
from pycse.sklearn.kan_interpret import (
    LINEAR_R2,
    classify_shape,
    cv_compare,
    nonlinearity_score,
    partial_effects,
    partial_effects_cv,
    plot_fold_scores,
    plot_partial_effect,
    sensitivity,
    shape_table,
    threshold_rules,
)
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X = rng.normal(size=(120, 3))
y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 2] ** 2 + 0.05 * rng.normal(size=120)
NAMES = ["a", "b", "c"]


def test_classify_shape_known_curves():
    x = np.linspace(-2, 2, 80)
    assert classify_shape(x, 3 * x + 1)[0] == "Linear"
    assert classify_shape(x, x**2)[0] == "U-shaped"
    assert classify_shape(x, -(x**2))[0] == "Inverted U"
    assert classify_shape(x, np.full_like(x, 2.0))[0] == "Constant"
    assert classify_shape(x, 0.01 * x**2, scale=10.0)[0] == "Constant"
    step = 1 / (1 + np.exp(-x / 0.1))
    assert classify_shape(x, step)[0] == "Step"
    assert classify_shape(x, step + 0.01 * np.sin(9 * x))[0] == "Step"  # wiggles on a plateau
    assert classify_shape(x, 1 - np.exp(-2 * (x + 2)))[0] == "Saturating"
    assert classify_shape(x, x**3)[0] == "Monotone"
    assert classify_shape(x, 1 / (1 + np.exp(-x / 0.8)))[0] == "Linear"  # a wide sigmoid is a line here
    assert classify_shape(x, np.sin(3 * x))[0] == "Complex"
    assert nonlinearity_score(x, 3 * x + 1) < 1e-6
    assert nonlinearity_score(x, x**2) > 0.3


def test_cv_compare_regression_pairwise():
    models = {"MLR": LinearRegression(), "KAN": KAN(layers=(3, 3, 1), grid_size=3)}
    res = cv_compare(
        models, X, y, n_splits=3, n_repeats=1, fit_kwargs={"KAN": {"maxiter": 5}}, verbose=False
    )
    assert list(res.summary.index) == ["MLR", "KAN"]
    assert set(res.folds.model) == {"MLR", "KAN"} and len(res.folds) == 6
    assert res.metric == "r2"
    assert res.pairwise["vs"] == "MLR" and res.pairwise["n"] == 3
    assert "Wilcoxon" in str(res)


def test_cv_compare_classification():
    yb = (y > np.median(y)).astype(int)
    models = {"LR": LogisticRegression(max_iter=500), "KAN": KAN(layers=(3, 3, 1), grid_size=3)}
    res = cv_compare(
        models,
        X,
        yb,
        task="classification",
        n_splits=3,
        n_repeats=1,
        fit_kwargs={"KAN": {"maxiter": 5}},
        verbose=False,
    )
    assert res.metric == "auc"
    assert res.summary.loc["LR", "auc_mean"] > 0.8  # a near-linear target
    assert 0 <= res.summary.loc["KAN", "accuracy_mean"] <= 1


def test_partial_effects_of_a_linear_model_are_linear():
    scaler = StandardScaler().fit(X)
    lr = LinearRegression().fit(scaler.transform(X), y)
    pe = partial_effects(lr, X, NAMES, scaler=scaler, extra_models={"copy": lr})
    assert set(pe) == set(NAMES)
    a = pe["a"]
    assert a.x_raw.shape == (80,) and np.isclose(a.x_raw.min(), X[:, 0].min())
    assert a.shape()[0] == "Linear"
    assert np.allclose(a.extra["copy"], a.y)
    # slope in raw units is the coefficient in raw units (2.0)
    slope = (a.y[-1] - a.y[0]) / (a.x_raw[-1] - a.x_raw[0])
    assert abs(slope - 2.0) < 0.1
    tbl = shape_table(pe)
    assert list(tbl.columns) == ["Feature", "Shape", "Interpretation", "Nonlinearity", "N_unique"]


def test_partial_effects_cv_envelope():
    pe = partial_effects_cv(
        lambda: LinearRegression(), X, y, NAMES, features=["a"], n_splits=3, verbose=False
    )
    a = pe["a"]
    assert a.curves.shape == (3, 80)
    assert np.all(a.lo <= a.median) and np.all(a.median <= a.hi)
    # folds of a linear model agree closely
    assert np.max(a.hi - a.lo) < 0.5


def test_sensitivity_matches_linear_coefficients():
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    lr = LinearRegression().fit(Xs, y)
    s = sensitivity(lr, Xs)
    assert np.allclose(s, np.abs(lr.coef_), atol=1e-6)


def test_threshold_rules_balanced():
    Xb = np.column_stack(
        [np.r_[rng.normal(-2, 0.3, 30), rng.normal(2, 0.3, 90)], rng.normal(size=120)]
    )
    yb = np.r_[np.zeros(30), np.ones(90)].astype(int)
    df = threshold_rules(Xb, yb, ["sep", "noise"])
    assert df.iloc[0]["Feature"] == "sep"
    assert df.iloc[0]["Balanced_Acc"] == pytest.approx(1.0)
    assert -1.5 < df.iloc[0]["Threshold"] < 1.5
    assert df.attrs["majority_accuracy"] == pytest.approx(0.75)


def test_plot_helpers_run():
    scaler = StandardScaler().fit(X)
    lr = LinearRegression().fit(scaler.transform(X), y)
    pe = partial_effects(lr, X, NAMES, features=["a"], scaler=scaler)
    pe = partial_effects_cv(
        lambda: LinearRegression(),
        X,
        y,
        NAMES,
        features=["a"],
        n_splits=3,
        effects=pe,
        verbose=False,
    )
    fig, ax = plt.subplots()
    plot_partial_effect(ax, pe["a"], ylabel="y")
    assert "Linear" in ax.get_title(loc="left")
    plt.close(fig)
    res = cv_compare(
        {"MLR": LinearRegression(), "KAN": LinearRegression()},
        X,
        y,
        n_splits=3,
        n_repeats=1,
        verbose=False,
    )
    fig, ax = plt.subplots()
    plot_fold_scores(ax, res)
    plt.close(fig)
    assert LINEAR_R2 == 0.9
