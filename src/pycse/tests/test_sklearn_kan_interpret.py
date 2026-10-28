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
    accumulated_local_effects,
    accumulated_local_effects_cv,
    classify_shape,
    cv_compare,
    nonlinearity_score,
    partial_effects,
    partial_effects_cv,
    plot_fold_scores,
    plot_partial_effect,
    prune_collinear,
    sensitivity,
    shape_table,
    threshold_rules,
    transitions,
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
    assert (
        classify_shape(x, 1 / (1 + np.exp(-x / 0.8)))[0] == "Linear"
    )  # a wide sigmoid is a line here
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


def _sig(z):
    return 1 / (1 + np.exp(-z))


def test_classify_shape_window_and_gap():
    x = np.linspace(-2, 2, 80)
    window = _sig((x + 0.8) / 0.08) - _sig((x - 0.9) / 0.08)
    assert classify_shape(x, window)[0] == "Window"
    assert classify_shape(x, 1 - window)[0] == "Gap"
    assert classify_shape(x, window + 0.02 * np.sin(11 * x))[0] == "Window"
    # a small rise before a large fall is still a window when both edges are real
    asym = 0.4 * _sig((x + 1.2) / 0.1) + 0.6 - _sig((x - 0.5) / 0.08)
    assert classify_shape(x, asym)[0] == "Window"
    # a ripple before a step is not
    assert (
        classify_shape(x, 0.1 * _sig((x + 1.2) / 0.1) + 0.9 - _sig((x - 0.5) / 0.08))[0] == "Step"
    )
    # the simpler templates keep their labels
    assert classify_shape(x, -(x**2))[0] == "Inverted U"
    assert classify_shape(x, _sig(x / 0.1))[0] == "Step"


def test_transitions_locate_edges():
    x = np.linspace(-2, 2, 401)
    tr = transitions(x, _sig((x + 0.8) / 0.05) - _sig((x - 0.9) / 0.05))
    assert abs(tr["rise"] + 0.8) < 0.05 and abs(tr["fall"] - 0.9) < 0.05
    tr = transitions(x, 1 - _sig((x - 0.3) / 0.05))
    assert tr["rise"] is None and abs(tr["fall"] - 0.3) < 0.05
    assert transitions(x, np.zeros_like(x))["fall"] is None
    # on a coarse grid the edge is interpolated, not snapped to a grid point
    xc = np.linspace(-2, 2, 9)
    assert abs(transitions(xc, 1 - _sig((xc - 0.3) / 0.05))["fall"] - 0.3) < 0.15


def test_prune_collinear_keeps_one_per_group():
    a = rng.normal(size=200)
    Xc = np.column_stack([a, 2 * a + 0.01 * rng.normal(size=200), rng.normal(size=200)])
    kept, dropped = prune_collinear(Xc, ["a", "a2", "c"])
    assert kept == ["a", "c"] and dropped["a2"][0] == "a" and dropped["a2"][1] > 0.99
    kept, dropped = prune_collinear(Xc, ["a", "a2", "c"], keep_first=["a2"])
    assert kept == ["a2", "c"] and dropped["a"][0] == "a2"


def test_ale_of_a_linear_model_has_the_raw_slope():
    scaler = StandardScaler().fit(X)
    lr = LinearRegression().fit(scaler.transform(X), y)
    ale = accumulated_local_effects(lr, X, NAMES, scaler=scaler, extra_models={"copy": lr})
    a = ale["a"]
    assert a.method == "accumulated local effect"
    slope = (a.y[-1] - a.y[0]) / (a.x_raw[-1] - a.x_raw[0])
    assert abs(slope - 2.0) < 0.1
    assert np.allclose(a.extra["copy"], a.y)
    # centred on the mean prediction
    assert (
        abs(np.mean(lr.predict(scaler.transform(X))) - np.interp(np.median(X[:, 0]), a.x_raw, a.y))
        < 0.5
    )


def test_ale_does_not_extrapolate_off_correlated_data():
    """Two nearly identical descriptors and a model that is only valid on the data."""
    n = 2000
    u = rng.uniform(-1, 1, n)
    Xc = np.column_stack([u, u + 0.02 * rng.normal(size=n)])

    class OnlyOnData:
        """Predicts 3*u on the diagonal, and a large penalty far from it."""

        def predict(self, Z):
            return 3 * Z[:, 0] - 50 * (Z[:, 0] - Z[:, 1]) ** 2

    m = OnlyOnData()
    ale = accumulated_local_effects(m, Xc, ["u", "v"], ["u"])["u"]
    slope = (ale.y[-1] - ale.y[0]) / (ale.x_raw[-1] - ale.x_raw[0])
    assert abs(slope - 3.0) < 0.5  # the true effect along the data
    pe = partial_effects(m, Xc, ["u", "v"], ["u"])["u"]
    assert pe.y.min() < -20  # the partial effect leaves the data and collapses


def test_ale_cv_envelope_shares_bins():
    scaler = StandardScaler().fit(X)
    lr = LinearRegression().fit(scaler.transform(X), y)
    ale = accumulated_local_effects(lr, X, NAMES, features=["a"], scaler=scaler, n_bins=10)
    ale = accumulated_local_effects_cv(
        lambda: LinearRegression(),
        X,
        y,
        NAMES,
        features=["a"],
        n_splits=3,
        n_bins=10,
        effects=ale,
        verbose=False,
    )["a"]
    assert ale.curves.shape == (3, len(ale.x_raw))
    assert np.max(ale.hi - ale.lo) < 0.5
    fig, ax = plt.subplots()
    plot_partial_effect(ax, ale, ylabel="y")
    assert "KAN accumulated local effect" in ax.get_legend_handles_labels()[1]
    plt.close(fig)
