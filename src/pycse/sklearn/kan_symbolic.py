"""Readable equations from KANs, and which parts of a model transfer to new molecules.

An additive KAN (``layers=(d, 1)``) is ``y = b + sum_i phi_i(x_i)``: one curve per descriptor.
Each curve is a one-dimensional object that can be named. This module:

- matches each curve to the simplest form in a small library (:data:`FORMS`: line, 1/x, log,
  exponential, quadratic, V, step, window) that reproduces it, with a support rule: a step or a
  window edge must have at least ``min_side`` distinct descriptor values on each side, otherwise
  it is a lookup of a few molecules, not a shape;
- :func:`snap` turns the matched curves into a :class:`Formula` and refits every coefficient to
  the data, so the printed numbers (a slope, a step position, the optimum of a volcano, a
  barrier from a 1/T term) are the formula's own;
- :func:`extract_pieces` turns the matched curves into *pieces*: unit shapes of one descriptor
  (0 to 1 past a threshold, 1 inside a window, a standardized line, a squared distance from an
  optimum), clipped to the range they were fit on. :class:`PieceModel` sums the pieces that
  earn their place under cross-validation, optionally choosing them by how well they predict
  molecules the fit has not seen (``select_on="groups"``), and can hand the residual to PySR
  to propose products of pieces (``products=True``; needs the ``pysr`` extra). PySR never
  sees a raw descriptor, only named pieces, and only +, - and *;
- :func:`transfer_ledger` scores a ladder of models, from the mean to the black boxes, on random
  folds (test reactions reuse molecules seen in training: interpolation) and on folds that hold
  out whole molecules (extrapolation), with a per-descriptor ledger of what each descriptor
  adds under each scheme, and the error on each held-out molecule against its distance to the
  nearest training molecule.

A descriptor that adds on random folds and not on new molecules is acting as a molecule label;
a model rung whose gain is all on random folds is memorizing the molecules it was trained on.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

from pycse.sklearn.kan_interpret import _cv_splits, _fit, _predict


# -----------------------------------------------------------------------------
# the form library
# -----------------------------------------------------------------------------
def _sig(t):
    """Logistic function, written with tanh so it does not overflow."""
    return 0.5 * (1.0 + np.tanh(0.5 * t))


# Each form is a function of z = (x - mean) / sd and its parameters; the raw forms (1/x, log)
# also need the mean and sd to recover x.
def _lin(z, p):
    return p[0] * z


def _quad(z, p):
    return p[0] * z + p[1] * z**2


def _exp(z, p):
    return p[0] * np.exp(p[1] * z)


def _vee(z, p):
    return p[0] * np.abs(z - p[1]) + p[2] * z


def _step(z, p):
    return p[0] * _sig((z - p[1]) / np.exp(p[2]))


def _window(z, p):
    w = np.exp(p[3])
    return p[0] * (_sig((z - p[1]) / w) - _sig((z - p[2]) / w))


def _inv(z, p, m=0.0, s=1.0):
    return p[0] / (m + s * z)


def _log(z, p, m=0.0, s=1.0):
    return p[0] * np.log(m + s * z)


#: ``name: (function, number of parameters)``. The order of :data:`ORDER` is the order of
#: preference: the first form that reproduces a curve is taken.
FORMS = {
    "linear": (_lin, 1),
    "inverse": (_inv, 1),
    "log": (_log, 1),
    "exponential": (_exp, 2),
    "quadratic": (_quad, 2),
    "V-shape": (_vee, 3),
    "step": (_step, 3),
    "window": (_window, 4),
}
ORDER = ["linear", "exponential", "quadratic", "V-shape", "step", "window"]
_RAW = {"inverse", "log"}  # forms of the raw value; used only when x > 0
_WIDTH = {"step": 2, "window": 3}  # index of the log-width parameter


def _eval(form, z, p, m, s):
    fn = FORMS[form][0]
    return fn(z, p, m, s) if form in _RAW else fn(z, p)


def _starts(form, z, f):
    """A few starting points per form; the fit keeps the best."""
    rng = np.ptp(f) or 1.0
    q = np.quantile(z, [0.2, 0.35, 0.5, 0.65, 0.8])
    if form in ("linear", "inverse", "log"):
        return [[np.polyfit(z, f, 1)[0]]]
    if form == "quadratic":
        c = np.polyfit(z, f, 2)
        return [[c[1], c[0]]]
    if form == "exponential":
        return [[a * rng * 0.3, b] for a in (1, -1) for b in (1.0, -1.0)]
    if form == "V-shape":
        return [[a * rng / 2, c, 0.0] for a in (1, -1) for c in q]
    if form == "step":
        return [[a * rng, c, np.log(0.2)] for a in (1, -1) for c in q]
    if form == "window":
        pairs = [(q[0], q[2]), (q[1], q[3]), (q[2], q[4]), (q[0], q[4]), (q[1], q[4])]
        return [[a * rng, lo, hi, np.log(0.2)] for a in (1, -1) for lo, hi in pairs]
    raise ValueError(f"unknown form {form!r}")


def fit_form(form, z, f, *, mean=0.0, sd=1.0, weights=None):
    """Least-squares fit of one form plus a constant to the curve ``f`` sampled at ``z``.

    Args:
        form: a key of :data:`FORMS`.
        z: standardized descriptor values, ``(x - mean) / sd``.
        f: the curve at those values.
        mean, sd: the standardization, needed by the forms of the raw value (1/x, log).
        weights: optional weight per point (for example the number of reactions at each
            distinct value, when ``z`` holds the distinct values only).

    Returns:
        ``(params, r2)``: the form's parameters (in ``z`` units) and the fraction of the
        curve's variance the form reproduces.
    """
    z, f = np.asarray(z, float), np.asarray(f, float)
    w = np.ones_like(z) if weights is None else np.asarray(weights, float)
    sw = np.sqrt(w)
    k = FORMS[form][1]
    best = None
    for p0 in _starts(form, z, f):
        x0 = np.r_[p0, np.mean(f)]

        def res(p):
            return sw * (_eval(form, z, p[:k], mean, sd) + p[k] - f)

        try:
            if form in _WIDTH:
                # an edge narrower than 0.01 sd cannot be told apart from a sharp one
                lo = np.full(len(x0), -np.inf)
                lo[_WIDTH[form]] = np.log(0.01)
                x0[_WIDTH[form]] = max(x0[_WIDTH[form]], lo[_WIDTH[form]])
                r = least_squares(res, x0, bounds=(lo, np.inf), method="trf", max_nfev=300)
            else:
                r = least_squares(
                    res, x0, method="lm" if len(z) > len(x0) else "trf", max_nfev=2000
                )
        except Exception:  # a start outside the form's domain (log of a negative, say)
            continue
        if np.isfinite(r.cost) and (best is None or r.cost < best.cost):
            best = r
    if best is None:
        return np.full(k, np.nan), -np.inf
    fm = np.average(f, weights=w)
    ss = np.sum(w * (f - fm) ** 2) or 1e-12
    return best.x[:k], float(1.0 - 2.0 * best.cost / ss)


def _supported(form, p, z, min_side):
    """Is a threshold-type form carried by enough distinct values on each side of its edges?"""
    u = np.unique(z)
    if form in ("step", "V-shape"):
        return (u < p[1]).sum() >= min_side and (u > p[1]).sum() >= min_side
    if form == "window":
        lo, hi = sorted(p[1:3])
        inside = ((u > lo) & (u < hi)).sum()
        return inside >= min_side and (u <= lo).sum() + (u >= hi).sum() >= min_side
    if form == "exponential":
        # a steep exponential is a step at the end of the range: where does it change by half?
        v = _exp(u, p)
        mid = 0.5 * (v.min() + v.max())
        return (v < mid).sum() >= min_side and (v > mid).sum() >= min_side
    if form == "quadratic" and p[1] != 0:
        z0 = -p[0] / (2 * p[1])
        if u.min() < z0 < u.max():
            return (u < z0).sum() >= min_side and (u > z0).sum() >= min_side
    return True


# -----------------------------------------------------------------------------
# terms: one matched curve of one descriptor
# -----------------------------------------------------------------------------
def _g(v, digits=3):
    return f"{v:.{digits}g}"


def _minus(name, v):
    """``name - v`` written without a double sign."""
    return f"{name} - {_g(v)}" if v >= 0 else f"{name} + {_g(-v)}"


@dataclass
class Term:
    """One descriptor's curve, matched to a form.

    ``value(x)`` is the term with its amplitude (a term of a :class:`Formula`); ``unit(x)`` is
    the unit shape (a piece): 0 to 1 for a step, 1 inside a window, ``z`` for a line, the
    squared standardized distance from the turning point for a quadratic. Both clip ``x`` to
    the range the term was fit on, so a formula never extrapolates a shape past its data.
    """

    feature: str
    form: str
    params: np.ndarray
    mean: float
    sd: float
    lo: float
    hi: float
    r2_to_curve: float = float("nan")

    def _z(self, x):
        return (np.clip(np.asarray(x, float), self.lo, self.hi) - self.mean) / self.sd

    def value(self, x):
        if self.form == "flag":
            return self.params[0] * self.unit(x)
        return _eval(self.form, self._z(x), self.params, self.mean, self.sd)

    def unit(self, x):
        z, p, m, s = self._z(x), self.params, self.mean, self.sd
        if self.form == "flag":
            return (np.asarray(x, float) > 0.5 * (self.lo + self.hi)).astype(float)
        if self.form == "linear":
            return z
        if self.form == "inverse":
            return m / (m + s * z)
        if self.form == "log":
            return np.log((m + s * z) / m)
        if self.form == "quadratic":
            return (z + p[0] / (2 * p[1])) ** 2
        if self.form == "exponential":
            return np.exp(p[1] * z)
        if self.form == "V-shape":
            return np.abs(z - p[1])
        if self.form == "step":
            return _sig((z - p[1]) / np.exp(p[2]))
        if self.form == "window":
            lo, hi = sorted(p[1:3])
            w = np.exp(p[3])
            return _sig((z - lo) / w) - _sig((z - hi) / w)
        raise ValueError(self.form)

    @property
    def x0(self):
        """The location the form names, in the descriptor's units (None for a line)."""
        m, s, p = self.mean, self.sd, self.params
        if self.form == "quadratic":
            return m + s * (-p[0] / (2 * p[1]))
        if self.form in ("step", "V-shape"):
            return m + s * p[1]
        if self.form == "window":
            return tuple(m + s * v for v in sorted(p[1:3]))
        return None

    def describe(self):
        """The term with its coefficients, in the descriptor's own units."""
        m, s, p, n = self.mean, self.sd, self.params, self.feature
        if self.form == "flag":
            return f"{_g(p[0])} * [{n} = {_g(self.hi)}]"
        if self.form == "linear":
            return f"{_g(p[0] / s)} * {n}"
        if self.form == "inverse":
            return f"{_g(p[0])} / {n}"
        if self.form == "log":
            return f"{_g(p[0])} * ln({n})"
        if self.form == "quadratic":
            x0, k = self.x0, p[1] / s**2
            kind = "minimum" if k > 0 else "maximum"
            return f"{_g(k)} * ({_minus(n, x0)})^2   [{kind} at {_g(x0)}]"
        if self.form == "exponential":
            return f"{_g(p[0] * np.exp(-p[1] * m / s))} * exp({_g(p[1] / s)} * {n})"
        if self.form == "V-shape":
            return f"{_g(p[0] / s)} * |{_minus(n, self.x0)}| + {_g(p[2] / s)} * {n}"
        if self.form == "step":
            w = s * np.exp(p[2])
            d = "rises" if p[0] > 0 else "falls"
            return f"{_g(p[0])} * S(({_minus(n, self.x0)}) / {_g(w)})   [{d} by {_g(abs(p[0]))} at {_g(self.x0)}]"
        if self.form == "window":
            (a, b), w = self.x0, s * np.exp(p[3])
            return f"{_g(p[0])} * W({n}; {_g(a)}, {_g(b)}, {_g(w)})   [window {_g(a)} to {_g(b)}]"
        raise ValueError(self.form)

    def describe_unit(self):
        """The unit shape (the piece), in the descriptor's own units."""
        m, s, p, n = self.mean, self.sd, self.params, self.feature
        if self.form == "flag":
            return f"[{n} = {_g(self.hi)}]  (0 or 1)"
        if self.form == "linear":
            return f"({_minus(n, m)}) / {_g(s)}"
        if self.form == "inverse":
            return f"{_g(m)} / {n}"
        if self.form == "log":
            return f"ln({n} / {_g(m)})"
        if self.form == "quadratic":
            return f"(({_minus(n, self.x0)}) / {_g(s)})^2   [turns at {_g(self.x0)}]"
        if self.form == "exponential":
            return f"exp({_g(p[1] / s)} * ({_minus(n, m)}))"
        if self.form == "V-shape":
            return f"|{_minus(n, self.x0)}| / {_g(s)}"
        if self.form == "step":
            return f"S(({_minus(n, self.x0)}) / {_g(s * np.exp(p[2]))})   [0 below {_g(self.x0)}, 1 above]"
        if self.form == "window":
            (a, b), w = self.x0, s * np.exp(p[3])
            return f"W({n}; {_g(a)}, {_g(b)}, {_g(w)})   [1 inside {_g(a)} to {_g(b)}]"
        raise ValueError(self.form)


_FOOTNOTES = {
    "step": "S(t) = 1 / (1 + exp(-t)), a step from 0 to 1 of width about 4 in t",
    "window": "W(x; a, b, w) = S((x - a) / w) - S((x - b) / w), 1 between a and b",
}


def match_curve(name, x, f, *, forms=None, min_r2=0.97, min_side=3):
    """Match one curve ``f(x)`` to the first form in ``forms`` that reproduces it.

    A descriptor with two values is a flag. A form needs more distinct values than it has
    parameters plus one, a threshold-type form needs the support of :func:`_supported`, and
    when no form reaches ``min_r2`` the best supported one is returned.
    """
    x, f = np.asarray(x, float), np.asarray(f, float)
    m, s = x.mean(), x.std() or 1.0
    lo, hi = x.min(), x.max()
    f = f - f.mean()
    if len(np.unique(x)) == 2:
        top = x > 0.5 * (lo + hi)
        amp = f[top].mean() - f[~top].mean() if top.any() and (~top).any() else 0.0
        return Term(name, "flag", np.array([amp]), m, s, lo, hi, 1.0)
    # fit at the distinct values, weighted by how many reactions sit at each: the same least
    # squares as fitting every reaction, at a fraction of the cost on few-level descriptors
    xu, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    fu = np.bincount(inv, weights=f) / cnt
    z = (xu - m) / s
    best = None
    for form in forms or ORDER:
        if form in _RAW and lo <= 0:
            continue
        if form != "linear" and len(np.unique(x)) <= FORMS[form][1] + 1:
            continue
        p, r2 = fit_form(form, z, fu, mean=m, sd=s, weights=cnt)
        if not np.isfinite(r2) or not _supported(form, p, z, min_side):
            continue
        if best is None or r2 > best[2] + 1e-9:
            best = (form, p, r2)
        if r2 >= min_r2:
            best = (form, p, r2)
            break
    if best is None:  # nothing supported: a line always is
        p, r2 = fit_form("linear", z, fu, weights=cnt)
        best = ("linear", p, r2)
    return Term(name, best[0], np.asarray(best[1], float), m, s, lo, hi, best[2])


def additive_curves(model, X):
    """Each descriptor's curve in an additive model, at the observed values.

    The model's prediction with one column moved (the others at their mean) differs from that
    descriptor's term by a constant, so this is exact for an additive KAN (``layers=(d, 1)``).
    For a model with interactions it is the curve at the mean of the other descriptors.

    Returns:
        ``(d, n)`` array, row ``i`` the curve of descriptor ``i`` at ``X[:, i]``.
    """
    X = np.asarray(X, float)
    ref = X.mean(axis=0)
    out = np.empty((X.shape[1], len(X)))
    for i in range(X.shape[1]):
        Z = np.tile(ref, (len(X), 1))
        Z[:, i] = X[:, i]
        out[i] = np.asarray(model.predict(Z), float).ravel()
    return out


def _check_additive(model):
    layers = getattr(model, "layers", None)
    if layers is not None and len(layers) != 2:
        raise ValueError(
            f"an additive KAN has layers=(d, 1); got {tuple(layers)}. With a hidden layer the "
            "one-descriptor curves do not add up to the model."
        )


def _match_all(model, X, feature_names, *, forms, per_feature, min_r2, min_side, min_share):
    X = np.asarray(X, float)
    _check_additive(model)
    curves = additive_curves(model, X)
    spread = curves.std(axis=1)
    share = spread / (spread.sum() or 1.0)
    per_feature = per_feature or {}
    terms = []
    for i, name in enumerate(feature_names):
        if share[i] < min_share or spread[i] == 0:
            continue
        terms.append(
            match_curve(
                name, X[:, i], curves[i], forms=per_feature.get(name, forms), min_r2=min_r2,
                min_side=min_side,
            )
        )  # fmt: skip
    return terms


# -----------------------------------------------------------------------------
# the formula: every coefficient refit to the data
# -----------------------------------------------------------------------------
@dataclass
class Formula:
    """``y = intercept + sum of terms``, each term a named form of one descriptor."""

    intercept: float
    terms: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)

    def predict(self, X):
        X = np.asarray(X, float)
        out = np.full(len(X), self.intercept)
        for t in self.terms:
            out = out + t.value(X[:, self.feature_names.index(t.feature)])
        return out

    def refit(self, X, y):
        """Refit every parameter of every term, and the intercept, to ``(X, y)``."""
        X, y = np.asarray(X, float), np.asarray(y, float)
        sizes = [len(t.params) for t in self.terms]
        cols = [X[:, self.feature_names.index(t.feature)] for t in self.terms]

        def unpack(p):
            out, i = [], 1
            for k in sizes:
                out.append(p[i : i + k])
                i += k
            return p[0], out

        def res(p):
            b, ps = unpack(p)
            pred = np.full(len(X), b)
            for t, c, pp in zip(self.terms, cols, ps):
                if t.form == "flag":
                    pred = pred + pp[0] * t.unit(c)
                else:
                    pred = pred + _eval(t.form, t._z(c), pp, t.mean, t.sd)
            return pred - y

        p0 = np.r_[self.intercept, *[t.params for t in self.terms]]
        r = least_squares(res, p0, method="trf", max_nfev=20000)
        b, ps = unpack(r.x)
        self.intercept = float(b)
        for t, pp in zip(self.terms, ps):
            t.params = np.asarray(pp, float)
        return self

    def __str__(self):
        lines = [f"y = {self.intercept:.4g}"] + [f"    + {t.describe()}" for t in self.terms]
        notes = [_FOOTNOTES[f] for f in ("step", "window") if any(t.form == f for t in self.terms)]
        if notes:
            lines.append("  where " + "; ".join(notes))
        return "\n".join(lines)


def snap(model, X, y, feature_names, *, forms=None, per_feature=None, min_r2=0.97, min_side=3,
         min_share=0.02):  # fmt: skip
    """A :class:`Formula` from a fitted additive KAN.

    Each descriptor's curve is matched to the first form in ``forms`` (default :data:`ORDER`)
    that reproduces it to ``min_r2``; a descriptor whose curve carries less than ``min_share``
    of the summed spread of all curves is dropped. Then every coefficient is refit to the data.

    Args:
        model: fitted model with ``predict`` on raw X and ``layers=(d, 1)`` if it is a KAN.
        X, y: the data the formula is refit to (raw units).
        feature_names: one name per column.
        per_feature: ``{name: [forms]}`` for a relationship the chemistry already names, for
            example ``{"T_K": ["inverse"]}`` for an Arrhenius or Eyring term.

    Returns:
        :class:`Formula`.
    """
    terms = _match_all(
        model, X, feature_names, forms=forms, per_feature=per_feature, min_r2=min_r2,
        min_side=min_side, min_share=min_share,
    )  # fmt: skip
    F = Formula(0.0, terms, list(feature_names))
    F.intercept = float(np.mean(np.asarray(y, float) - F.predict(X)))
    return F.refit(X, y)


# -----------------------------------------------------------------------------
# pieces, their selection, and the piece model
# -----------------------------------------------------------------------------
def extract_pieces(model, X, feature_names, *, forms=None, per_feature=None, min_r2=0.97,
                   min_side=3):  # fmt: skip
    """The unit shapes (see :meth:`Term.unit`) of every descriptor of an additive KAN.

    A V-shape contributes two pieces, ``|x - c|`` and the line, so a sum of pieces can weigh
    the two halves independently.
    """
    out = []
    for t in _match_all(
        model, X, feature_names, forms=forms, per_feature=per_feature, min_r2=min_r2,
        min_side=min_side, min_share=0.0,
    ):  # fmt: skip
        out.append(t)
        if t.form == "V-shape":
            out.append(
                Term(t.feature, "linear", np.ones(1), t.mean, t.sd, t.lo, t.hi, t.r2_to_curve)
            )
    return out


def piece_matrix(pieces, X, feature_names):
    """Columns of unit shapes, one per piece."""
    X = np.asarray(X, float)
    return np.column_stack([p.unit(X[:, list(feature_names).index(p.feature)]) for p in pieces])


def select_pieces(P, y, *, groups=None, max_pieces=12, min_gain=0.005, random_state=0):
    """Forward selection of piece columns for a linear model, by 5-fold cross-validated R2.

    With ``groups``, each fold holds out whole molecules, so a piece enters only if it helps
    predict molecules the fit has not seen. Selection starts from what the training mean
    scores on the same folds, which is below zero when held-out molecules have means of
    their own, and stops when the best next piece adds less than ``min_gain``.

    Returns:
        ``(chosen, path)``: column indices in the order they entered, and the cross-validated
        R2 after each.
    """
    P, y = np.asarray(P, float), np.asarray(y, float)
    if groups is None:
        cv = KFold(5, shuffle=True, random_state=random_state)
    else:
        cv = list(GroupKFold(5, shuffle=True, random_state=random_state).split(P, y, groups))
    score = cross_val_score(DummyRegressor(), P[:, :1], y, cv=cv, scoring="r2").mean()
    chosen, path = [], []
    while len(chosen) < min(max_pieces, P.shape[1]):
        best = None
        for j in range(P.shape[1]):
            if j in chosen:
                continue
            s = cross_val_score(
                LinearRegression(), P[:, chosen + [j]], y, cv=cv, scoring="r2"
            ).mean()
            if best is None or s > best[1]:
                best = (j, s)
        if best[1] - score < min_gain:
            break
        chosen.append(best[0])
        score = best[1]
        path.append((best[0], float(score)))
    return chosen, path


def pysr_regressor(**options):
    """A PySRRegressor set up to assemble pieces: +, - and * only, no unary operators.

    Needs the ``pysr`` extra (``pip install "pycse[pysr]"``), which installs Julia on first use.
    """
    try:
        from pysr import PySRRegressor
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError('products=True needs PySR: pip install "pycse[pysr]"') from e
    opts = dict(
        niterations=80, maxsize=30, binary_operators=["+", "-", "*"], unary_operators=[],
        model_selection="best", populations=12, population_size=40, random_state=0,
        progress=False, verbosity=0, temp_equation_file=True, delete_tempfiles=True,
    )  # fmt: skip
    opts.update(options)
    return PySRRegressor(**opts)


def knee(regressor, y_var, tol=0.01):
    """Index of the smallest equation on a PySR front within ``tol`` of the best explained
    fraction ``1 - loss / y_var``."""
    eq = regressor.equations_
    r2 = 1.0 - eq["loss"].to_numpy() / y_var
    return int(eq.index[r2 >= np.nanmax(r2) - tol][0])


class PieceModel(BaseEstimator, RegressorMixin):
    """An additive KAN, read as a sum of named pieces, with optional products from PySR.

    ``fit`` trains an additive KAN, matches each descriptor's curve to a form, turns the forms
    into unit pieces (:func:`extract_pieces`), keeps the pieces that earn their place under
    cross-validation (:func:`select_pieces`), and fits their sum by least squares. With
    ``products=True`` the residual of that sum goes to PySR, restricted to +, - and * of the
    chosen pieces, and the smallest equation within ``products_tol`` of the best is added.

    Args:
        feature_names: names for the equation (default ``x0, x1, ...``).
        grid_size, reg, maxiter, seed: the additive KAN (``reg`` is both the L1 and the entropy
            penalty).
        forms, per_feature, min_r2, min_side: the matching (see :func:`match_curve`).
        select_on: ``"random"`` (pieces that help on random folds) or ``"groups"`` (pieces that
            help on held-out molecules; ``fit`` then needs ``groups``).
        max_pieces, min_gain: the selection.
        products: hand the residual to PySR.
        pysr_options: overrides for :func:`pysr_regressor`.
        products_tol: the knee tolerance (see :func:`knee`).

    Attributes:
        kan_, pieces_ (all pieces), chosen_ (indices into ``pieces_``), linear_ (the sum),
        products_ (the PySR regressor or None), products_index_ (the equation used).
    """

    def __init__(self, feature_names=None, grid_size=5, reg=1e-3, maxiter=300, seed=19,
                 forms=None, per_feature=None, min_r2=0.97, min_side=3, select_on="random",
                 max_pieces=12, min_gain=0.005, products=False, pysr_options=None,
                 products_tol=0.01):  # fmt: skip
        self.feature_names = feature_names
        self.grid_size = grid_size
        self.reg = reg
        self.maxiter = maxiter
        self.seed = seed
        self.forms = forms
        self.per_feature = per_feature
        self.min_r2 = min_r2
        self.min_side = min_side
        self.select_on = select_on
        self.max_pieces = max_pieces
        self.min_gain = min_gain
        self.products = products
        self.pysr_options = pysr_options
        self.products_tol = products_tol

    def _names(self, d):
        return (
            list(self.feature_names)
            if self.feature_names is not None
            else [f"x{i}" for i in range(d)]
        )

    def fit(self, X, y, groups=None):
        from pycse.sklearn.kan import KAN

        X, y = np.asarray(X, float), np.asarray(y, float)
        self.kan_ = KAN(
            layers=(X.shape[1], 1), grid_size=self.grid_size, l1_spline=self.reg,
            entropy_reg=self.reg, seed=self.seed,
        ).fit(X, y, maxiter=self.maxiter)  # fmt: skip
        return self.fit_pieces(self.kan_, X, y, groups=groups)

    def fit_pieces(self, model, X, y, groups=None):
        """Everything after the KAN: pieces from an already fitted additive ``model``."""
        X, y = np.asarray(X, float), np.asarray(y, float)
        names = self._names(X.shape[1])
        self.pieces_ = extract_pieces(
            model, X, names, forms=self.forms, per_feature=self.per_feature, min_r2=self.min_r2,
            min_side=self.min_side,
        )  # fmt: skip
        P = piece_matrix(self.pieces_, X, names)
        if self.select_on == "groups" and groups is None:
            raise ValueError('select_on="groups" needs fit(X, y, groups=...)')
        g = groups if self.select_on == "groups" else None
        self.chosen_, self.path_ = select_pieces(
            P, y, groups=g, max_pieces=self.max_pieces, min_gain=self.min_gain
        )
        Pc = P[:, self.chosen_]
        self.linear_ = (
            LinearRegression().fit(Pc, y) if self.chosen_ else DummyRegressor().fit(P[:, :1], y)
        )
        self.products_, self.products_index_ = None, None
        if self.products and self.chosen_:
            r = y - self.linear_.predict(Pc)
            self.products_ = pysr_regressor(**(self.pysr_options or {})).fit(
                Pc, r, variable_names=[f"p{k}" for k in range(len(self.chosen_))]
            )
            self.products_index_ = knee(self.products_, np.var(y), self.products_tol)
        return self

    def _P(self, X):
        names = self._names(np.asarray(X).shape[1])
        return (
            piece_matrix([self.pieces_[j] for j in self.chosen_], X, names)
            if self.chosen_
            else np.zeros((len(X), 1))
        )

    def predict(self, X):
        P = self._P(X)
        out = self.linear_.predict(P)
        if self.products_ is not None:
            extra = np.asarray(self.products_.predict(P, index=self.products_index_), float)
            out = out + np.where(np.isfinite(extra), extra, 0.0)
        return out

    def equation(self):
        """The fitted model, written out: the pieces, their sum, and the product terms."""
        lines = ["pieces:"]
        for k, j in enumerate(self.chosen_):
            lines.append(f"  p{k} = {self.pieces_[j].describe_unit()}")
        if hasattr(self.linear_, "coef_"):
            s = " ".join(f"{c:+.4g}*p{k}" for k, c in enumerate(self.linear_.coef_))
            lines.append(f"y = {self.linear_.intercept_:.4g} {s}")
        else:
            lines.append(f"y = {float(self.linear_.constant_.ravel()[0]):.4g}")
        if self.products_ is not None:
            lines.append(f"    + {self.products_.sympy(self.products_index_)}")
        notes = [
            _FOOTNOTES[f]
            for f in ("step", "window")
            if any(self.pieces_[j].form == f for j in self.chosen_)
        ]
        if notes:
            lines.append("  where " + "; ".join(notes))
        return "\n".join(lines)

    @property
    def chosen_features_(self):
        return list(dict.fromkeys(self.pieces_[j].feature for j in self.chosen_))


# -----------------------------------------------------------------------------
# the interpolation / extrapolation ledger
# -----------------------------------------------------------------------------
@dataclass
class TransferLedger:
    """Output of :func:`transfer_ledger`.

    Attributes:
        ladder: per rung, the score on random folds and on new molecules, the gain over the rung
            above in each, and the fraction of the random-fold score kept on new molecules.
        descriptors: per descriptor, the score lost when its pieces leave the sum, on each
            scheme, and a verdict (transfers, memorizes, helps only new molecules, minor).
        novelty: per held-out molecule, its distance to the nearest training molecule and each
            rung's mean absolute error on its reactions.
        tiers: mean error per rung for the nearest, middle and farthest third of held-out
            molecules, with the far/near ratio and the Spearman correlation of error with
            distance.
        memorized: the random-fold score lost when pieces are chosen on new molecules instead
            of random folds.
        folds: per fold and rung, the score (both schemes).
        metric, group_name.
    """

    ladder: pd.DataFrame
    descriptors: pd.DataFrame
    novelty: pd.DataFrame
    tiers: pd.DataFrame
    memorized: float
    folds: pd.DataFrame
    metric: str
    group_name: str
    equations: dict = field(default_factory=dict)

    def __str__(self):
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            return "\n".join([
                f"Ladder ({self.metric}; random folds reuse {self.group_name}s seen in training, "
                f"new = whole {self.group_name}s held out):",
                self.ladder.round(3).to_string(),
                f"\nRandom-fold {self.metric} that is memorized (lost when pieces must help new "
                f"{self.group_name}s): {self.memorized:.3f}",
                "\nDescriptor ledger (score lost when the descriptor leaves the sum of pieces):",
                self.descriptors.round(3).to_string(index=False),
                f"\nMean |error| on held-out {self.group_name}s by distance to the nearest training {self.group_name}:",
                self.tiers.round(3).to_string(),
            ])  # fmt: skip


def _score(task, y, p):
    if task == "classification":
        return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    return r2_score(y, p)


def group_columns(X, groups):
    """Indices of the columns that are constant within every group: the molecule's own
    descriptors, the space in which novelty is measured."""
    X, groups = np.asarray(X, float), np.asarray(groups)
    cols = []
    for j in range(X.shape[1]):
        if np.ptp(X[:, j]) > 0 and all(
            np.ptp(X[groups == g, j]) < 1e-12 for g in np.unique(groups)
        ):
            cols.append(j)
    return cols


def transfer_ledger(X, y, groups, feature_names, *, task="regression", models=None,
                    fit_kwargs=None, piece_options=None, products=False, n_splits=5,
                    random_state=42, tol=0.01, group_name="molecule", verbose=True):  # fmt: skip
    """Interpolation and extrapolation, rung by rung and descriptor by descriptor.

    Every rung is scored on the same random folds and on the same folds of whole held-out
    molecules (``groups``). The rungs, from simplest:

    - ``mean``: the training mean (or class prior);
    - ``pieces``: the sum of pieces chosen on random folds (:class:`PieceModel`);
    - ``pieces, chosen on new molecules``: the same pieces, chosen by how well they predict
      held-out molecules within the training set;
    - ``pieces + products`` (``products=True``): PySR products of the chosen pieces on the residual;
    - ``KAN additive``: the additive KAN the pieces came from, unsnapped;
    - then each of ``models`` (for example the usual KAN and gradient boosting), in order.

    Args:
        X, y, groups: raw descriptors, target, and the molecule each reaction is grouped by.
        task: ``"regression"`` (R2) or ``"classification"`` (AUC; the piece models are fit on
            the 0/1 target as a probability).
        models: ``{name: unfitted estimator}`` for the upper rungs; cloned per fold and fit on
            raw X (a KAN standardizes internally).
        fit_kwargs: ``{name: kwargs}`` for those models' ``fit``.
        piece_options: keyword arguments for :class:`PieceModel`.
        tol: the score change the descriptor verdicts treat as real.

    Returns:
        :class:`TransferLedger`.
    """
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    names = list(feature_names)
    models, fit_kwargs, piece_options = models or {}, fit_kwargs or {}, dict(piece_options or {})
    piece_options.setdefault("feature_names", names)
    rungs = ["mean", "pieces", "pieces, chosen on new molecules"]
    rungs += ["pieces + products"] if products else []
    rungs += ["KAN additive", *models]
    cols = group_columns(X, groups)
    Z = (X[:, cols] - X[:, cols].mean(0)) / X[:, cols].std(0) if cols else np.zeros((len(X), 1))
    gz = {g: Z[groups == g][0] for g in np.unique(groups)}

    fold_rows, drops, nov_rows = [], {"random": {}, "grouped": {}}, []
    equations = {}
    for scheme in ["random", "grouped"]:
        splits = _cv_splits(
            X, y, task, n_splits, 1, random_state, groups if scheme == "grouped" else None
        )
        for k, (tr, te) in enumerate(splits):
            preds = {"mean": np.full(len(te), y[tr].mean())}
            pm = PieceModel(**{**piece_options, "products": products}).fit(X[tr], y[tr])
            preds["pieces"] = (
                pm.predict(X[te]) if not products else pm.linear_.predict(pm._P(X[te]))
            )
            if products:
                preds["pieces + products"] = pm.predict(X[te])
            pg = (
                clone(pm)
                .set_params(select_on="groups", products=False)
                .fit_pieces(pm.kan_, X[tr], y[tr], groups=groups[tr])
            )
            preds["pieces, chosen on new molecules"] = pg.predict(X[te])
            preds["KAN additive"] = np.asarray(pm.kan_.predict(X[te]), float).ravel()
            for name, proto in models.items():
                m = _fit(clone(proto), X[tr], y[tr], task, fit_kwargs.get(name))
                preds[name] = _predict(m, X[te], task)
            for r in rungs:
                fold_rows.append(
                    {"scheme": scheme, "fold": k, "rung": r, "score": _score(task, y[te], preds[r])}
                )
            # descriptor ledger on the sum of pieces chosen on random folds
            Ptr, Pte = pm._P(X[tr]), pm._P(X[te])
            base = _score(task, y[te], pm.linear_.predict(Pte))
            feats = [pm.pieces_[j].feature for j in pm.chosen_]
            for f in dict.fromkeys(feats):
                keep = [i for i, g in enumerate(feats) if g != f]
                pd_ = (
                    LinearRegression().fit(Ptr[:, keep], y[tr]).predict(Pte[:, keep])
                    if keep
                    else np.full(len(te), y[tr].mean())
                )
                drops[scheme].setdefault(f, []).append(base - _score(task, y[te], pd_))
            if scheme == "grouped":
                train_g = np.unique(groups[tr])
                Ztr = np.array([gz[g] for g in train_g])
                for g in np.unique(groups[te]):
                    sel = groups[te] == g
                    row = {group_name: g, "novelty": float(np.min(np.linalg.norm(Ztr - gz[g], axis=1)) / np.sqrt(max(len(cols), 1))),
                           "reactions": int(sel.sum())}  # fmt: skip
                    for r in rungs:
                        p = (
                            np.clip(preds[r][sel], 0, 1)
                            if task == "classification"
                            else preds[r][sel]
                        )
                        row[r] = float(np.mean(np.abs(y[te][sel] - p)))
                    nov_rows.append(row)
            if verbose:
                print(f"  {scheme} fold {k + 1}/{n_splits}: " + ", ".join(
                    f"{r} {fold_rows[-len(rungs) + i]['score']:.3f}" for i, r in enumerate(rungs)), flush=True)  # fmt: skip

    folds = pd.DataFrame(fold_rows)
    t = folds.groupby(["rung", "scheme"])["score"].mean().unstack().loc[rungs]
    ladder = pd.DataFrame({"random folds": t["random"], "new molecules": t["grouped"]})
    ladder["gain, random"] = ladder["random folds"].diff()
    ladder["gain, new"] = ladder["new molecules"].diff()
    ladder["kept on new"] = ladder["new molecules"] / ladder["random folds"]
    memorized = float(
        ladder.loc["pieces", "random folds"]
        - ladder.loc["pieces, chosen on new molecules", "random folds"]
    )

    rows = []
    for f in dict.fromkeys(list(drops["random"]) + list(drops["grouped"])):
        # a fold whose selection left the descriptor out contributes 0
        r = float(np.sum(drops["random"].get(f, [])) / n_splits)
        g = float(np.sum(drops["grouped"].get(f, [])) / n_splits)
        verdict = ("transfers" if r > tol and g > tol else "memorizes" if r > tol and g <= 0
                   else "helps only new molecules" if g > tol and r <= tol else "minor")  # fmt: skip
        rows.append({"descriptor": f, "lost, random": r, "lost, new": g,
                     "folds (random/new)": f"{len(drops['random'].get(f, []))}/{len(drops['grouped'].get(f, []))}",
                     "verdict": verdict})  # fmt: skip
    descriptors = (
        pd.DataFrame(rows).sort_values("lost, random", ascending=False).reset_index(drop=True)
    )

    novelty = pd.DataFrame(nov_rows)
    q = novelty["novelty"].quantile([1 / 3, 2 / 3]).to_numpy()
    novelty["tier"] = np.where(
        novelty.novelty <= q[0], "near", np.where(novelty.novelty <= q[1], "middle", "far")
    )
    tiers = novelty.groupby("tier")[rungs].mean().reindex(["near", "middle", "far"]).T
    tiers["far / near"] = tiers["far"] / tiers["near"]
    tiers["Spearman(distance, error)"] = pd.Series(
        {r: spearmanr(novelty.novelty, novelty[r]).statistic for r in rungs}
    )
    metric = "AUC" if task == "classification" else "R2"
    return TransferLedger(
        ladder, descriptors, novelty, tiers, memorized, folds, metric, group_name, equations
    )
