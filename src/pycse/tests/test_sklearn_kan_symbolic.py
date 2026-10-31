"""Tests for pycse.sklearn.kan_symbolic: readable equations and the transfer ledger.

The fast tests use an exactly additive stand-in for a fitted KAN, so they test the matching,
the formula, the selection and the ledger bookkeeping without training anything.
"""

import os

import numpy as np
import pytest

from pycse.sklearn.kan_symbolic import (
    Term,
    _sig,
    extract_pieces,
    fit_form,
    group_columns,
    match_curve,
    piece_matrix,
    select_pieces,
    snap,
)


class Additive:
    """``predict(X) = sum_i f_i(X[:, i])``: what an additive KAN is."""

    def __init__(self, fns):
        self.fns = fns
        self.layers = (len(fns), 1)

    def predict(self, X):
        X = np.asarray(X, float)
        return sum(f(X[:, i]) for i, f in enumerate(self.fns))


SIGMA = np.array([0.78, 0.66, 0.54, 0.45, 0.23, 0.06, 0.0, -0.17, -0.20, -0.27, -0.83])


def truth_data(n=400, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [rng.choice(SIGMA, n), rng.uniform(22, 50, n), rng.uniform(-30, 0, n), rng.normal(size=n)]
    )
    fns = [
        lambda s: 1.5 * s,
        lambda v: -1.5 * _sig(v - 32.0),
        lambda e: -0.01 * (e + 15.0) ** 2,
        lambda z: 0.0 * z,
    ]
    y = -3.0 + Additive(fns).predict(X) + noise * rng.normal(size=n)
    return X, y, Additive(fns), ["sigma_p", "pct_Vbur_min", "dE_bind", "junk"]


def test_fit_form_recovers_a_step():
    z = np.linspace(-2, 2, 200)
    f = 2.0 * _sig((z - 0.3) / 0.1)
    p, r2 = fit_form("step", z, f)
    assert r2 > 0.999
    assert p[1] == pytest.approx(0.3, abs=0.02)
    assert np.exp(p[2]) == pytest.approx(0.1, rel=0.1)


def test_snap_recovers_the_generating_equation():
    X, y, model, names = truth_data(noise=0.05)
    F = snap(model, X, y, names)
    forms = {t.feature: t.form for t in F.terms}
    assert forms == {"sigma_p": "linear", "pct_Vbur_min": "step", "dE_bind": "quadratic"}
    t = {t.feature: t for t in F.terms}
    assert t["sigma_p"].params[0] / t["sigma_p"].sd == pytest.approx(1.5, abs=0.05)
    assert t["pct_Vbur_min"].x0 == pytest.approx(32.0, abs=0.3)
    assert t["pct_Vbur_min"].params[0] == pytest.approx(-1.5, abs=0.05)
    assert t["dE_bind"].x0 == pytest.approx(-15.0, abs=0.3)
    assert "maximum at -15" in t["dE_bind"].describe()
    assert np.corrcoef(F.predict(X), y)[0, 1] > 0.99


def test_inverse_form_returns_the_barrier():
    """An Eyring term, -dH / (2.303 R) / T, read as a/T gives dH back."""
    T = np.random.default_rng(1).uniform(288, 368, 300)
    a = -15.0 / (2.303 * 1.987e-3)
    term = match_curve("T_K", T, a / T, forms=["inverse"])
    assert term.form == "inverse"
    assert -term.params[0] * 2.303 * 1.987e-3 == pytest.approx(15.0, rel=1e-3)


def test_a_step_between_two_molecules_is_not_a_step():
    """An edge with fewer than min_side distinct values on one side is a lookup."""
    x = np.repeat(np.arange(12.0), 10)
    f = np.where(x >= 9.5, 1.0, 0.0)  # two molecules above the edge
    assert match_curve("x", x, f, forms=["step"], min_side=1).form == "step"
    lax = match_curve("x", x, f, min_side=1)  # a steep exponential is the same lookup
    assert lax.form in ("exponential", "step") and lax.r2_to_curve > 0.99
    strict = match_curve("x", x, f, min_side=3)
    assert strict.form not in ("exponential", "step") and strict.r2_to_curve < 0.95


def test_terms_do_not_extrapolate_past_their_data():
    t = Term("x", "linear", np.array([2.0]), 0.0, 1.0, -1.0, 1.0)
    assert t.value(5.0) == pytest.approx(t.value(1.0))
    assert t.unit(-9.0) == pytest.approx(-1.0)


def test_flags_and_v_shapes_become_pieces():
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.integers(0, 2, 300).astype(float), rng.uniform(-2, 2, 300)])
    model = Additive([lambda f: 3.0 * f, lambda x: np.abs(x - 0.4) + 0.5 * x])
    pieces = extract_pieces(model, X, ["flag", "x"])
    assert [p.form for p in pieces] == ["flag", "V-shape", "linear"]
    P = piece_matrix(pieces, X, ["flag", "x"])
    assert set(np.unique(P[:, 0])) == {0.0, 1.0}
    assert pieces[1].x0 == pytest.approx(0.4, abs=0.05)


def test_selection_on_new_molecules_starts_from_the_mean():
    """Held-out molecules with means of their own make even the mean score below zero; a piece
    that helps must still enter, and a piece of pure noise must not."""
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(10), 30)
    a = rng.normal(size=300)
    y = a + 2.0 * rng.normal(size=10)[groups] + 0.1 * rng.normal(size=300)
    P = np.column_stack([rng.normal(size=300), a])
    chosen, path = select_pieces(P, y, groups=groups)
    assert chosen == [1]
    assert path[0][1] < 0.5  # still poor on new molecules, but better than the mean


def test_group_columns_finds_the_molecule_descriptors():
    groups = np.repeat(np.arange(5), 4)
    X = np.column_stack([groups * 1.5, np.arange(20.0), np.ones(20)])
    assert group_columns(X, groups) == [0]


@pytest.mark.slow
def test_piece_model_and_ledger_end_to_end():
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    from pycse.sklearn.kan_symbolic import PieceModel, transfer_ledger

    X, y, _, names = truth_data(n=200, noise=0.1, seed=4)
    groups = np.searchsorted(np.sort(np.unique(X[:, 0])), X[:, 0])  # one group per substituent
    pm = PieceModel(feature_names=names, maxiter=150).fit(X, y)
    assert "pct_Vbur_min" in pm.chosen_features_
    assert np.corrcoef(pm.predict(X), y)[0, 1] > 0.95
    assert "y = " in pm.equation()
    L = transfer_ledger(
        X, y, groups, names, piece_options={"maxiter": 150}, group_name="substituent", verbose=False
    )
    assert list(L.ladder.index) == [
        "mean",
        "pieces",
        "pieces, chosen on new molecules",
        "KAN additive",
    ]
    v = dict(zip(L.descriptors.descriptor, L.descriptors.verdict))
    assert v["pct_Vbur_min"] == "transfers"
    assert L.ladder.loc["pieces", "new molecules"] > 0.8
    assert {"near", "middle", "far", "far / near"} <= set(L.tiers.columns)


@pytest.mark.slow
def test_pysr_products_recover_a_switching_slope():
    pytest.importorskip("pysr")
    from pycse.sklearn.kan_symbolic import pysr_regressor

    rng = np.random.default_rng(5)
    s, step = rng.normal(size=300), rng.integers(0, 2, 300).astype(float)
    r = 2.0 * s * step  # the slope of s switches on with the flag
    sr = pysr_regressor(niterations=20).fit(
        np.column_stack([s, step]), r, variable_names=["p0", "p1"]
    )
    pred = sr.predict(np.column_stack([s, step]))
    assert np.corrcoef(pred, r)[0, 1] > 0.99


def test_a_diverged_model_falls_back_to_the_mean():
    from pycse.sklearn.kan_symbolic import PieceModel

    class Diverged:
        layers = (2, 1)

        def predict(self, X):
            return np.full(len(X), np.nan)

    X, y = np.random.default_rng(6).normal(size=(50, 2)), np.arange(50.0)
    with pytest.warns(UserWarning, match="not finite"):
        pm = PieceModel().fit_pieces(Diverged(), X, y)
    assert pm.chosen_ == []
    assert np.allclose(pm.predict(X), y.mean())
