"""The learning curve in units: what it measures, and what it refuses to say."""

import numpy as np
import pandas as pd
import pytest

from transferledger import NewGroups, backtest, learning_curve
from transferledger.learning import power_law


def population(n_groups, signal, seed=0, per=6):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    z = rng.normal(size=(n_groups, 2))
    offset = signal * z[:, 0] + 0.5 * rng.normal(size=n_groups)
    x = rng.normal(size=len(g))
    y = x + offset[g] + 0.3 * rng.normal(size=len(g))
    X = pd.DataFrame({"x": x, "z1": z[g, 0], "z2": z[g, 1]})
    return X, y, g


def test_power_law_recovers_a_known_curve():
    m = np.array([4, 6, 9, 14, 20, 30.0])
    a, b, c = power_law(m, 0.8 - 2.0 / m)
    assert abs(a - 0.8) < 1e-9 and abs(b - 2.0) < 1e-9 and c == 1.0
    # a curve that falls with more units is read as flat, and the ceiling stops at 1
    a, b, _ = power_law(m, 0.2 + 1.0 / m)
    assert b == 0.0 and abs(a - np.mean(0.2 + 1.0 / m)) < 1e-12
    assert power_law(m, 1.5 - 6.0 / m)[0] == 1.0


def test_curve_rises_when_a_group_property_carries():
    X, y, g = population(30, signal=1.5)
    L = learning_curve(X, y, NewGroups(g, "new site"), sizes=[4, 10, 24], n_repeats=1)
    assert L.unit == "site" and L.n_units == 30
    c = L.curve[L.curve.rung == L.best].set_index("units")
    assert c.loc[24, "skill"] > 0.5 and c.loc[24, "lo"] > 0
    assert c.loc[24, "skill"] > c.loc[4, "skill"]
    assert set(L.curve.columns) >= {"rung", "units", "skill", "lo", "hi", "spread", "draws"}
    assert "new site" in str(L) and L.state() == "established"


def test_curve_stays_at_zero_when_nothing_carries():
    X, y, g = population(30, signal=0.0, seed=1)
    # x is a row-level variable that does carry; leave it out so only group properties remain
    L = learning_curve(
        X.drop(columns="x"), y, NewGroups(g, "new site"), sizes=[4, 10, 24], n_repeats=1
    )
    top = L.curve[L.curve.units == 24]
    assert (top.skill < 0.1).all()
    # nothing is established, so there is no floor to hold: a curve at zero does not say where
    # it would leave zero
    assert L.state() == "nothing yet" and L.floor() is None
    assert "Nothing yet" in str(L) and "detectable" in L.advice()


def test_floor_is_the_lower_end_of_the_established_skill():
    X, y, g = population(30, signal=1.5)
    L = learning_curve(X, y, NewGroups(g, "new site"), sizes=[4, 10, 24], n_repeats=1)
    top = L.top()
    assert L.floor() == pytest.approx(top["lo"]) and 0 < L.floor() < top["skill"]
    assert L.advice().startswith("Established") and f"{L.floor():.2f}" in L.advice()
    d = L.to_dict()
    assert d["state"][L.best] == L.state() and d["floor"][L.best] == L.floor()
    # extending the fitted curve is available, and is not what the curve is said to support
    assert L.extend([30, 60]).shape == (2,) and "Not a forecast" in L.extend.__doc__


def test_rows_as_units_and_bad_input():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    y = 2.0 * X.a + 0.3 * rng.normal(size=120)
    L = learning_curve(X, y, sizes=[10, 30, 90], n_repeats=1)
    c = L.curve[L.curve.rung == "equation"].set_index("units")
    assert L.unit == "row" and c.loc[90, "skill"] > 0.9
    with pytest.raises(ValueError, match="too few"):
        learning_curve(X.iloc[:12], y[:12], NewGroups(np.repeat([0, 1, 2], 4)))


def test_yes_no_target_with_one_class_in_a_small_training_set():
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(20), 5)
    z = rng.normal(size=20)
    y = (rng.uniform(size=100) < 1 / (1 + np.exp(-3 * z[g]))).astype(float)
    X = pd.DataFrame({"z": z[g], "noise": rng.normal(size=100)})
    L = learning_curve(X, y, NewGroups(g), family="binomial", sizes=[2, 6, 14], n_repeats=1)
    assert np.isfinite(L.curve.skill).all()


def test_backtest_forecasts_first_and_then_reveals():
    X, y, g = population(40, signal=1.5, seed=3)
    b = backtest(X, y, NewGroups(g, "new site"), have=12, factors=(2.0, 5.0), n_trials=2,
                 n_acquisitions=2, sizes=[4, 9], n_repeats=1)  # fmt: skip
    # twelve units doubled leave sixteen to test on; five times twelve is more than there are
    assert set(b.units) == {24} and len(b) == 2 * 2
    assert set(b.columns) >= {"state", "held", "lo", "hi", "extended", "realized"}
    assert np.isfinite(b[["held", "lo", "hi", "extended", "realized"]].to_numpy()).all()
    assert set(b.state) <= {"nothing yet", "established"}
