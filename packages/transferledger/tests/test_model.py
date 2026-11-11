"""The piece model: recovery of a known equation, inputs, families, shift-aware selection."""

import numpy as np
import pandas as pd
import pytest

from transferledger import NewGroups, PieceModel
from transferledger.forms import sigmoid

R = 1.987e-3
SIGMA = np.array([0.78, 0.66, 0.54, 0.45, 0.23, 0.06, 0.0, -0.17, -0.20, -0.27, -0.83])


def calibration(n=600, seed=0, interaction=True, noise=0.15):
    """A rate law with a cliff, a volcano, an Eyring term and a slope that switches sign across
    the cliff; plus one unrelated column."""
    rng = np.random.default_rng(seed)
    sub = rng.integers(0, len(SIGMA), n)
    df = pd.DataFrame({
        "sigma": SIGMA[sub], "T": rng.uniform(288, 368, n), "V": rng.uniform(22, 50, n),
        "E": rng.uniform(-30, 0, n), "junk": rng.normal(size=n),
    })  # fmt: skip
    s = sigmoid(df.V - 32.0)
    rho = (2.0 - 2.5 * s) if interaction else 1.5
    y = (-3 + rho * df.sigma - 1.5 * s - 15 / (2.303 * R) * (1 / df["T"] - 1 / 298.15)
         - 0.01 * (df.E + 15) ** 2 + noise * rng.normal(size=n))  # fmt: skip
    return df, y.to_numpy(), sub


def test_recovers_the_generating_equation_with_its_interaction():
    X, y, _ = calibration()
    m = PieceModel(forms={"T": ["inverse"]}, products=True).fit(X, y)
    t = m.terms_.set_index("variable")
    assert set(m.variables_) == {"sigma", "T", "V", "E"}  # the unrelated column is left out
    assert t.loc["V", "form"] == "step" and t.loc["V", "location"] == pytest.approx(32.0, abs=0.4)
    assert t.loc["E", "form"] == "quadratic" and t.loc["E", "location"] == pytest.approx(
        -15.0, abs=0.5
    )
    barrier = -t.loc["T", "weight"] * X["T"].mean() * 2.303 * R
    assert barrier == pytest.approx(15.0, abs=0.5)
    # the slope of sigma on either side of the cliff: +2.0 below, -0.5 above
    below = t.loc["sigma", "weight"] / X.sigma.std(ddof=0)
    above = below + t.loc["V x sigma", "weight"] / X.sigma.std(ddof=0)
    assert below == pytest.approx(2.0, abs=0.2) and above == pytest.approx(-0.5, abs=0.2)
    assert m.skill_ > 0.97


def test_no_product_is_invented_for_an_additive_truth():
    X, y, _ = calibration(interaction=False)
    m = PieceModel(forms={"T": ["inverse"]}, products=True).fit(X, y)
    assert m.products_ == []


def test_table_input_text_columns_and_weights():
    X, y, _ = calibration(n=300)
    X["site"] = np.where(np.arange(len(X)) % 3 == 0, "north", "south")
    y = y + 2.0 * (X.site == "north")
    m = PieceModel().fit(X, y, sample_weight=np.ones(len(y)))
    assert "site" in m.variables_
    assert list(m.feature_names_in_) == list(X.columns) and m.n_features_in_ == 6
    assert np.allclose(m.decision_function(X), m.intercept_ + m.components(X).sum(axis=1))
    new = X.head(4).assign(site="west")
    assert m.support(new).unseen_level.all()


def test_missing_values_are_an_error_that_names_the_column():
    X, y, _ = calibration(n=200)
    X.loc[::9, "E"] = np.nan
    with pytest.raises(ValueError, match="missing values in: E"):
        PieceModel().fit(X, y)


def test_binary_target_gives_probabilities():
    X, y, _ = calibration(n=400, noise=0.0)
    yb = (np.random.default_rng(1).uniform(size=len(y)) < sigmoid(2.0 * (y - np.median(y)))).astype(
        float
    )
    m = PieceModel(family="binomial").fit(X, yb)
    p = m.predict(X)
    assert 0 <= p.min() and p.max() <= 1
    assert "log-odds" in m.equation()
    assert np.mean((p > 0.5) == (yb == 1)) > 0.75


def test_support_reports_rows_beyond_the_fitted_range():
    X, y, _ = calibration(n=300)
    m = PieceModel().fit(X, y)
    inside = m.support(X)
    assert inside.outside.sum() == 0 and np.allclose(inside.distance, 0)
    far = m.support(X.assign(V=80.0))
    assert (far.outside == 1).all() and (far.variables == "V").all() and (far.distance > 0).all()
    assert np.allclose(m.predict(X.assign(V=80.0)), m.predict(X.assign(V=X.V.max())))


def test_selection_under_a_shift_leaves_out_what_only_identifies_a_group():
    """Twelve groups, each with its own offset, and group-level columns that are pure noise.
    On random folds those columns earn a place by identifying the group; held to new groups
    they do not."""
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(12), 40)
    x = rng.normal(size=len(g))
    Z = rng.normal(size=(12, 3))[g]
    y = 1.5 * x + 2.0 * rng.normal(size=12)[g] + 0.2 * rng.normal(size=len(g))
    X = pd.DataFrame(np.column_stack([x, Z]), columns=["x", "z1", "z2", "z3"])
    on_rows = PieceModel(tune_source=False).fit(X, y)
    on_groups = PieceModel(select_on=NewGroups(g)).fit(X, y)
    assert len(on_rows.variables_) > 1
    assert on_groups.variables_ == ["x"]


def test_refits_are_identical_and_clone_works():
    from sklearn.base import clone

    X, y, _ = calibration(n=300)
    a = PieceModel(products=True).fit(X, y)
    b = clone(a).fit(X, y)
    assert a.equation() == b.equation()


def test_predictions_stay_near_the_range_of_the_training_target():
    X, y, _ = calibration(n=300)
    m = PieceModel().fit(X, y)
    m.intercept_ += 1e6  # an absurd equation
    lo, hi = y.min(), y.max()
    assert m.predict(X).max() <= hi + 0.5 * (hi - lo) + 1e-9
    assert PieceModel(limit=None).fit(X, y).limit is None
