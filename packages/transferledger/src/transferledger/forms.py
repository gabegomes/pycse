"""Named one-variable forms, and matching a curve to the simplest one that reproduces it.

A *form* is a unit shape of one variable with at most a few shape parameters: a line, a step
at ``c`` of width ``w``, a window between two edges, a squared distance from a turning point.
Matching is separable least squares: the shape parameters are searched on a grid and then
polished, and for each candidate the amplitude and offset are solved exactly. The search has
no starting-point dependence, and a candidate is only admissible when the data support it: an
edge needs ``min_side`` distinct values on each side, otherwise the form would be a lookup of
one or two cases rather than a shape.

New forms are added with :func:`register_form`; nothing else in the package needs to know them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize

__all__ = ["Form", "Scale", "Piece", "FORMS", "DEFAULT_ORDER", "register_form", "fit_form",
           "match_curve", "sigmoid"]  # fmt: skip


def sigmoid(t):
    """Logistic function, written with tanh so it cannot overflow."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(t, float)))


@dataclass(frozen=True)
class Scale:
    """How one variable was standardized, and the range it was seen on."""

    mean: float
    sd: float
    lo: float
    hi: float

    def z(self, x, clip=True):
        x = np.asarray(x, float)
        if clip:
            x = np.clip(x, self.lo, self.hi)
        return (x - self.mean) / self.sd

    def x(self, z):
        return self.mean + self.sd * np.asarray(z, float)

    @property
    def zlo(self):
        return (self.lo - self.mean) / self.sd

    @property
    def zhi(self):
        return (self.hi - self.mean) / self.sd


@dataclass(frozen=True)
class Form:
    """A unit shape of one standardized variable ``z``.

    Attributes:
        name: the key in :data:`FORMS`.
        shape: ``shape(z, theta, scale) -> array``, the unit shape.
        grid: ``grid(u, scale) -> (k, n_theta) array`` of candidate shape parameters, given the
            sorted distinct values ``u`` of ``z``. Forms without shape parameters return one
            empty row.
        valid: ``valid(theta, u, scale, min_side) -> bool``; the support rule.
        text: ``text(name, theta, scale) -> str``, the unit shape in the variable's own units.
        location: ``location(theta, scale)``, the value(s) the form names, in the variable's
            units (an edge, a turning point), or None.
        n_theta: number of shape parameters.
        companion_line: the form is fitted together with a line in the same variable (a V is
            ``|z - c|`` plus a line), and the line becomes a piece of its own.
        positive_only: the form is of the raw value and needs it positive.
        note: a footnote explaining notation the text uses.
    """

    name: str
    shape: Callable
    grid: Callable
    valid: Callable
    text: Callable
    location: Callable | None = None
    n_theta: int = 0
    companion_line: bool = False
    positive_only: bool = False
    note: str = ""


def _g(v, digits=3):
    return f"{v:.{digits}g}"


def _minus(name, v, sd=1.0):
    """``name - v`` without a double sign, and without subtracting a zero."""
    if abs(v) < 1e-9 * sd:
        return f"{name}"
    return f"{name} - {_g(v)}" if v >= 0 else f"{name} + {_g(-v)}"


def _sides(u, c):
    return int((u < c).sum()), int((u > c).sum())


_NONE = np.zeros((1, 0))


def _always(theta, u, scale, min_side):
    return True


# ---- line, and forms of the raw value -------------------------------------------------------
LINEAR = Form(
    "linear",
    shape=lambda z, th, sc: np.asarray(z, float),
    grid=lambda u, sc: _NONE,
    valid=_always,
    text=lambda n, th, sc: f"({_minus(n, sc.mean, sc.sd)}) / {_g(sc.sd)}",
)
INVERSE = Form(
    "inverse",
    shape=lambda z, th, sc: sc.mean / sc.x(z),
    grid=lambda u, sc: _NONE,
    valid=_always,
    text=lambda n, th, sc: f"{_g(sc.mean)} / {n}",
    positive_only=True,
)
LOG = Form(
    "log",
    shape=lambda z, th, sc: np.log(sc.x(z) / sc.mean),
    grid=lambda u, sc: _NONE,
    valid=_always,
    text=lambda n, th, sc: f"ln({n} / {_g(sc.mean)})",
    positive_only=True,
)


# ---- quadratic: squared distance from a turning point ----------------------------------------
def _quad_grid(u, sc):
    r = u[-1] - u[0]
    return np.linspace(u[0] - 0.5 * r, u[-1] + 0.5 * r, 81)[:, None]


def _quad_valid(th, u, sc, min_side):
    r = u[-1] - u[0]
    if not (u[0] - 0.5 * r <= th[0] <= u[-1] + 0.5 * r):
        return False
    if u[0] < th[0] < u[-1]:  # a turning point inside the data needs data on both sides
        return min(_sides(u, th[0])) >= min_side
    return True


QUADRATIC = Form(
    "quadratic",
    shape=lambda z, th, sc: (np.asarray(z, float) - th[0]) ** 2,
    grid=_quad_grid,
    valid=_quad_valid,
    text=lambda n, th, sc: (
        f"(({_minus(n, sc.x(th[0]), sc.sd)}) / {_g(sc.sd)})^2   [turns at {_g(sc.x(th[0]))}]"
    ),
    location=lambda th, sc: float(sc.x(th[0])),
    n_theta=1,
)


# ---- exponential ------------------------------------------------------------------------------
def _exp_ref(th, sc):
    return sc.zhi if th[0] > 0 else sc.zlo  # keeps the shape in (0, 1]


def _exp_shape(z, th, sc):
    return np.exp(th[0] * (np.asarray(z, float) - _exp_ref(th, sc)))


def _exp_grid(u, sc):
    r = u[-1] - u[0]
    b = np.geomspace(1.0 / r, 8.0 / r, 14)
    return np.r_[b, -b][:, None]


def _exp_valid(th, u, sc, min_side):
    r = u[-1] - u[0]
    if not (1.0 <= abs(th[0]) * r <= 8.0):  # slower is a line, faster is a jump at the end
        return False
    v = _exp_shape(u, th, sc)
    mid = 0.5 * (v.min() + v.max())
    return int((v < mid).sum()) >= min_side and int((v > mid).sum()) >= min_side


EXPONENTIAL = Form(
    "exponential",
    shape=_exp_shape,
    grid=_exp_grid,
    valid=_exp_valid,
    text=lambda n, th, sc: (
        f"exp({_g(th[0] / sc.sd)} * ({_minus(n, sc.x(_exp_ref(th, sc)), sc.sd)}))"
    ),
    n_theta=1,
)


# ---- saturation: x / (K + x), for a non-negative variable ------------------------------------
def _sat_shape(z, th, sc):
    x = sc.x(z)
    return x / (th[0] + x)


def _sat_grid(u, sc):
    hi = max(sc.hi, 1e-12)
    return np.geomspace(0.02 * hi, 5.0 * hi, 24)[:, None]


def _sat_valid(th, u, sc, min_side):
    if th[0] <= 0:
        return False
    v = _sat_shape(u, th, sc)
    mid = 0.5 * (v.min() + v.max())
    return int((v < mid).sum()) >= min_side and int((v > mid).sum()) >= min_side


SATURATION = Form(
    "saturation",
    shape=_sat_shape,
    grid=_sat_grid,
    valid=_sat_valid,
    text=lambda n, th, sc: f"{n} / ({_g(th[0])} + {n})   [half its limit at {_g(th[0])}]",
    location=lambda th, sc: float(th[0]),
    n_theta=1,
    positive_only=True,
)


# ---- V: |z - c| plus a line -------------------------------------------------------------------
def _edge_candidates(u, n_max):
    """Candidate edge positions: between neighbouring distinct values, thinned to ``n_max``."""
    mids = 0.5 * (u[1:] + u[:-1])
    if len(mids) > n_max:
        mids = np.quantile(mids, np.linspace(0, 1, n_max))
    return mids


VEE = Form(
    "vee",
    shape=lambda z, th, sc: np.abs(np.asarray(z, float) - th[0]),
    grid=lambda u, sc: _edge_candidates(u, 60)[:, None],
    valid=lambda th, u, sc, min_side: min(_sides(u, th[0])) >= min_side,
    text=lambda n, th, sc: f"|{_minus(n, sc.x(th[0]), sc.sd)}| / {_g(sc.sd)}",
    location=lambda th, sc: float(sc.x(th[0])),
    n_theta=1,
    companion_line=True,
)

# ---- step and window ---------------------------------------------------------------------------
_WIDTHS = np.array([0.004, 0.012, 0.03, 0.07, 0.15, 0.3])  # as fractions of the range


def _step_grid(u, sc):
    r = u[-1] - u[0]
    c = _edge_candidates(u, 60)
    return np.array([(ci, w * r) for ci in c for w in _WIDTHS])


def _width_ok(w, u):
    r = u[-1] - u[0]
    return 0.002 * r <= w <= 0.35 * r


def _step_valid(th, u, sc, min_side):
    return _width_ok(th[1], u) and min(_sides(u, th[0])) >= min_side


STEP = Form(
    "step",
    shape=lambda z, th, sc: sigmoid((np.asarray(z, float) - th[0]) / th[1]),
    grid=_step_grid,
    valid=_step_valid,
    text=lambda n, th, sc: (
        f"S(({_minus(n, sc.x(th[0]), sc.sd)}) / {_g(sc.sd * th[1])})   [0 below {_g(sc.x(th[0]))}, 1 above]"
    ),
    location=lambda th, sc: float(sc.x(th[0])),
    n_theta=2,
    note="S(t) = 1 / (1 + exp(-t)), a step from 0 to 1 of width about 4 in t",
)


def _win_grid(u, sc):
    r = u[-1] - u[0]
    c = _edge_candidates(u, 24)
    out = [(a, b, w * r) for i, a in enumerate(c) for b in c[i + 1 :] for w in _WIDTHS[:5:2]]
    return np.array(out) if out else np.zeros((0, 3))


def _win_valid(th, u, sc, min_side):
    a, b, w = th
    if not (a < b and _width_ok(w, u)):
        return False
    inside = int(((u > a) & (u < b)).sum())
    return inside >= min_side and int((u < a).sum()) + int((u > b).sum()) >= min_side


WINDOW = Form(
    "window",
    shape=lambda z, th, sc: (
        sigmoid((np.asarray(z, float) - th[0]) / th[2])
        - sigmoid((np.asarray(z, float) - th[1]) / th[2])
    ),
    grid=_win_grid,
    valid=_win_valid,
    text=lambda n, th, sc: (
        f"W({n}; {_g(sc.x(th[0]))}, {_g(sc.x(th[1]))}, {_g(sc.sd * th[2])})"
        f"   [1 inside {_g(sc.x(th[0]))} to {_g(sc.x(th[1]))}]"
    ),
    location=lambda th, sc: (float(sc.x(th[0])), float(sc.x(th[1]))),
    n_theta=3,
    note="W(x; a, b, w) = S((x - a) / w) - S((x - b) / w), 1 between a and b",
)

#: Every known form, by name.
FORMS: dict[str, Form] = {
    f.name: f for f in [LINEAR, INVERSE, LOG, EXPONENTIAL, SATURATION, QUADRATIC, VEE, STEP, WINDOW]
}
#: The forms tried by default, simplest first. Forms of the raw value (``inverse``, ``log``,
#: ``saturation``) are opt-in per variable, for a relationship the field already names.
DEFAULT_ORDER = ["linear", "exponential", "quadratic", "vee", "step", "window"]


def register_form(form: Form, *, default: bool = False, before: str | None = None):
    """Add a form to :data:`FORMS`; with ``default`` also to the default search order."""
    FORMS[form.name] = form
    if default and form.name not in DEFAULT_ORDER:
        i = DEFAULT_ORDER.index(before) if before else len(DEFAULT_ORDER)
        DEFAULT_ORDER.insert(i, form.name)
    return form


# -----------------------------------------------------------------------------
# matching
# -----------------------------------------------------------------------------
def _r2_columns(C, f, w):
    """Weighted R2 of regressing f on each row of C (plus a constant), vectorized."""
    sw = w / w.sum()
    fc = f - sw @ f
    Cc = C - (C @ sw)[:, None]
    var_f = sw @ fc**2
    var_c = (Cc**2) @ sw
    cov = Cc @ (sw * fc)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(var_c > 1e-14, cov**2 / (var_c * var_f), 0.0)
    return np.nan_to_num(r2)


def _r2_with_line(c, z, f, w):
    """Weighted R2 of f on [c, z, 1]."""
    A = np.column_stack([c, z, np.ones_like(z)]) * np.sqrt(w)[:, None]
    b = f * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = b - A @ coef
    fm = np.average(f, weights=w)
    return 1.0 - float(res @ res) / float(np.sum(w * (f - fm) ** 2))


def fit_form(form: Form, u, f, w, scale: Scale, *, min_side=3, polish=True):
    """Best admissible shape parameters of ``form`` for the curve ``f`` at distinct values ``u``.

    Args:
        form: the form.
        u: sorted distinct standardized values.
        f: the curve at those values.
        w: weight of each value (how many observations sit there).
        scale: the variable's standardization.
        min_side: the support rule's count.
        polish: refine the best grid candidate locally.

    Returns:
        ``(theta, r2)``; ``r2`` is ``-inf`` when no candidate is admissible.
    """
    u, f, w = np.asarray(u, float), np.asarray(f, float), np.asarray(w, float)
    if form.n_theta == 0:
        grid = np.zeros((1, 0))
    else:
        cand = np.asarray(form.grid(u, scale), float).reshape(-1, form.n_theta)
        ok = [th for th in cand if form.valid(th, u, scale, min_side)]
        if not ok:
            return np.zeros(form.n_theta), -np.inf
        grid = np.array(ok)

    def r2_of(th):
        c = form.shape(u, th, scale)
        if form.companion_line:
            return _r2_with_line(c, u, f, w)
        return float(_r2_columns(c[None, :], f, w)[0])

    if form.companion_line:
        scores = np.array([r2_of(th) for th in grid])
    else:
        scores = _r2_columns(np.array([form.shape(u, th, scale) for th in grid]), f, w)
    best = int(np.argmax(scores))
    theta, r2 = grid[best].copy(), float(scores[best])
    if polish and form.n_theta:

        def cost(th):
            if not form.valid(th, u, scale, min_side):
                return 1.0  # worse than any admissible fit
            return -r2_of(th)

        step = np.maximum(np.abs(theta) * 0.05, 0.02 * (u[-1] - u[0]))
        simplex = np.vstack(
            [theta] + [theta + step * np.eye(len(theta))[i] for i in range(len(theta))]
        )
        res = minimize(cost, theta, method="Nelder-Mead",
                       options={"initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-7, "maxiter": 200})  # fmt: skip
        if -res.fun > r2:
            theta, r2 = np.asarray(res.x, float), float(-res.fun)
    return theta, r2


@dataclass
class Piece:
    """A unit shape of one variable: what the equation is a sum (and product) of.

    ``unit(x)`` evaluates it. Values outside the range the piece was fitted on are clipped to
    that range, so a shape is never extended past its data; :meth:`outside` says where that
    happened.
    """

    feature: str
    form: str
    theta: np.ndarray
    scale: Scale
    r2: float = float("nan")
    companion_of: str | None = None
    column: int | None = None  # index of the variable in the design matrix
    group: str | None = None  # the input column the variable came from
    extrapolate: bool = field(default=False)

    def unit(self, x):
        x = np.asarray(x, float)
        if self.form == "flag":
            return (x > 0.5 * (self.scale.lo + self.scale.hi)).astype(float)
        z = self.scale.z(x, clip=not self.extrapolate)
        return FORMS[self.form].shape(z, self.theta, self.scale)

    def outside(self, x):
        x = np.asarray(x, float)
        tol = 1e-9 * max(abs(self.scale.lo), abs(self.scale.hi), 1.0)
        return (x < self.scale.lo - tol) | (x > self.scale.hi + tol)

    @property
    def location(self):
        if self.form == "flag":
            return None
        loc = FORMS[self.form].location
        return None if loc is None else loc(self.theta, self.scale)

    @property
    def note(self):
        return "" if self.form == "flag" else FORMS[self.form].note

    def text(self):
        if self.form == "flag":
            if "=" in self.feature:  # an indicator of one level of a categorical column
                return f"[{self.feature}]  (0 or 1)"
            return f"[{self.feature} = {_g(self.scale.hi)}]  (0 or 1)"
        return FORMS[self.form].text(self.feature, self.theta, self.scale)


def _collapse(x, f, weights, max_points):
    """The curve at distinct x (weighted), binned by quantile when there are very many."""
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    xu, inv = np.unique(x, return_inverse=True)
    cnt = np.bincount(inv, weights=w)
    fu = np.bincount(inv, weights=w * f) / cnt
    if len(xu) > max_points:
        edges = np.quantile(xu, np.linspace(0, 1, max_points + 1))
        b = np.clip(np.searchsorted(edges, xu, side="right") - 1, 0, max_points - 1)
        keep = np.bincount(b, minlength=max_points) > 0
        wb = np.bincount(b, weights=cnt, minlength=max_points)[keep]
        fu = np.bincount(b, weights=cnt * fu, minlength=max_points)[keep] / wb
        xu = np.bincount(b, weights=cnt * xu, minlength=max_points)[keep] / wb
        cnt = wb
    return xu, fu, cnt


def match_curve(name, x, f, *, weights=None, forms=None, min_r2=0.97, min_side=3,
                max_points=256, column=None):  # fmt: skip
    """Match the curve ``f(x)`` of one variable to the first form that reproduces it.

    Forms are tried in the order given (default :data:`DEFAULT_ORDER`, simplest first) and the
    first whose admissible fit explains ``min_r2`` of the curve's variance is taken; when none
    does, the best admissible fit is. A variable with two values is a flag. A form with shape
    parameters needs more distinct values than it has parameters, plus two.

    Args:
        name: the variable's name.
        x, f: its observed values and the curve there.
        weights: optional observation weights.
        forms: names of the forms to try, in order.
        min_r2: fraction of the curve's variance a form must reproduce to be accepted.
        min_side: distinct values required on each side of an edge.
        max_points: the curve is binned to at most this many points before matching.
        column: the variable's column index, stored on the pieces.

    Returns:
        A list of :class:`Piece`: one, or two for a form with a companion line.
    """
    x, f = np.asarray(x, float), np.asarray(f, float)
    scale = Scale(float(x.mean()), float(x.std() or 1.0), float(x.min()), float(x.max()))
    xu, fu, cnt = _collapse(x, f, weights, max_points)
    if len(xu) < 2:
        return []
    if len(xu) == 2:
        return [Piece(name, "flag", np.zeros(0), scale, 1.0, column=column)]
    u = (xu - scale.mean) / scale.sd
    best = None
    for fname in forms or DEFAULT_ORDER:
        form = FORMS[fname]
        if form.positive_only and scale.lo <= 0:
            continue
        if form.n_theta and len(xu) < form.n_theta + int(form.companion_line) + 3:
            continue
        theta, r2 = fit_form(form, u, fu, cnt, scale, min_side=min_side)
        if not np.isfinite(r2):
            continue
        if best is None or r2 > best[2] + 1e-9:
            best = (form, theta, r2)
        if r2 >= min_r2:
            best = (form, theta, r2)
            break
    if best is None:  # nothing admissible among the requested forms: a line always is
        best = (LINEAR, np.zeros(0), fit_form(LINEAR, u, fu, cnt, scale)[1])
    form, theta, r2 = best
    out = [Piece(name, form.name, np.asarray(theta, float), scale, float(r2), column=column)]
    if form.companion_line:
        out.append(Piece(name, "linear", np.zeros(0), scale, float(r2), companion_of=form.name,
                         column=column))  # fmt: skip
    return out
