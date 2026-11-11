"""Curve sources: additive models that give one curve per variable.

The equation's pieces are read off these curves, so a curve source only has to do two things:
``fit(X, y, sample_weight)`` and ``curves(X)``, the contribution of each variable at the rows of
``X``. :class:`SplineAdditive` is the default: penalized splines fitted by linear algebra, so
the fit is exact, repeatable and fast enough to refit hundreds of times for intervals.
:class:`KANAdditive` wraps a Kolmogorov-Arnold network with no hidden layer (the same family of
curves, fitted by iterative optimization), and :class:`ModelCurves` reads curves off any fitted
model that predicts.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

__all__ = ["SplineAdditive", "KANAdditive", "ModelCurves", "FAMILIES"]


# -----------------------------------------------------------------------------
# families: the link between the additive score and the target
# -----------------------------------------------------------------------------
class _Gaussian:
    name = "gaussian"

    @staticmethod
    def mean(eta):
        return eta

    @staticmethod
    def start(y, w):
        return np.full(len(y), np.average(y, weights=w))

    @staticmethod
    def working(y, eta):
        return eta * 0 + y, np.ones_like(y)  # response, weight

    @staticmethod
    def loss(y, mu):
        return (y - mu) ** 2


class _Binomial:
    """A 0/1 target: the additive score is the log-odds."""

    name = "binomial"

    @staticmethod
    def mean(eta):
        return 0.5 * (1.0 + np.tanh(0.5 * eta))

    @staticmethod
    def start(y, w):
        p = np.clip(np.average(y, weights=w), 1e-3, 1 - 1e-3)
        return np.full(len(y), np.log(p / (1 - p)))

    @classmethod
    def working(cls, y, eta):
        mu = np.clip(cls.mean(eta), 1e-6, 1 - 1e-6)
        v = mu * (1 - mu)
        return eta + (y - mu) / v, v

    @staticmethod
    def loss(y, mu):
        mu = np.clip(mu, 1e-9, 1 - 1e-9)
        return -2.0 * (y * np.log(mu) + (1 - y) * np.log(1 - mu))


class _Poisson:
    """A count target: the additive score is the log of the mean."""

    name = "poisson"

    @staticmethod
    def mean(eta):
        return np.exp(np.clip(eta, -30, 30))

    @staticmethod
    def start(y, w):
        return np.full(len(y), np.log(max(np.average(y, weights=w), 1e-6)))

    @classmethod
    def working(cls, y, eta):
        mu = np.maximum(cls.mean(eta), 1e-9)
        return eta + (y - mu) / mu, mu

    @staticmethod
    def loss(y, mu):
        mu = np.maximum(mu, 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(y > 0, y * np.log(y / mu), 0.0)
        return 2.0 * (t - (y - mu))


#: ``name: family``. The loss of a family is its deviance per observation.
FAMILIES = {"gaussian": _Gaussian, "binomial": _Binomial, "poisson": _Poisson}


def get_family(family):
    if isinstance(family, str):
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}; choose from {sorted(FAMILIES)}")
        return FAMILIES[family]
    return family


# -----------------------------------------------------------------------------
# penalized additive splines
# -----------------------------------------------------------------------------
class _Block:
    """One variable's columns in the design: an indicator, level indicators, or a spline."""

    def __init__(self, x, n_knots, degree, max_levels):
        u = np.unique(x)
        self.lo, self.hi = float(u[0]), float(u[-1])
        if len(u) <= 1:
            self.kind, self.width = "none", 0
        elif len(u) == 2:
            self.kind, self.width = "flag", 1
            self.cut = 0.5 * (u[0] + u[1])
        elif len(u) <= max_levels:
            # few distinct values: a free mean per value, which is all the data can give
            self.kind, self.levels, self.width = "levels", u, len(u)
        else:
            self.kind = "spline"
            k = int(min(n_knots, max(4, len(u) // 3)))
            inner = np.unique(np.quantile(u, np.linspace(0, 1, k)))
            self.degree = degree
            self.t = np.r_[[inner[0]] * degree, inner, [inner[-1]] * degree]
            self.width = len(self.t) - degree - 1

    def design(self, x):
        x = np.asarray(x, float)
        if self.kind == "none":
            return np.zeros((len(x), 0))
        if self.kind == "flag":
            return (x > self.cut).astype(float)[:, None]
        if self.kind == "levels":
            # nearest level, so an unseen value takes the curve of its neighbour
            idx = np.abs(x[:, None] - self.levels[None, :]).argmin(axis=1)
            return np.eye(len(self.levels))[idx]
        xc = np.clip(x, self.lo, self.hi)  # flat outside the fitted range
        return BSpline.design_matrix(xc, self.t, self.degree, extrapolate=False).toarray()

    def penalty(self):
        if self.kind == "spline":
            D = np.diff(np.eye(self.width), n=2, axis=0)
            return D.T @ D
        if self.kind == "levels":
            return 1e-4 * np.eye(self.width)  # only enough to make the level means identifiable
        return np.zeros((self.width, self.width))


class SplineAdditive:
    """Additive model of penalized splines, one smooth curve per variable.

    ``y`` (or its link) is modelled as ``b + sum_j f_j(x_j)``. Each ``f_j`` is a cubic B-spline
    with knots at quantiles and a second-difference penalty (a P-spline); a variable with two
    values gets an indicator and one with at most ``max_levels`` values a free mean per value.
    Two strengths are chosen together by generalized cross-validation: one for wiggliness and
    one that shrinks whole curves toward zero, so that near-duplicate variables do not trade
    large opposite effects. The fit is a linear solve (iterated for non-Gaussian families), so
    the same data always give the same curves.

    Args:
        n_knots: knots per smooth variable.
        degree: spline degree.
        max_levels: variables with at most this many distinct values get a mean per value.
        family: ``"gaussian"``, ``"binomial"`` (0/1 target, log-odds scale) or ``"poisson"``.
        lambdas: smoothing strengths to choose from.
        shrinkage: shrinkage strengths to choose from, relative to the scale of the design.

    Attributes:
        intercept_, lambda_, edf_ (effective degrees of freedom of the chosen fit).
    """

    def __init__(
        self, n_knots=12, degree=3, max_levels=6, family="gaussian", lambdas=None, shrinkage=None
    ):
        self.n_knots = n_knots
        self.degree = degree
        self.max_levels = max_levels
        self.family = family
        self.lambdas = lambdas
        self.shrinkage = shrinkage

    def _design(self, X):
        cols = [b.design(X[:, j]) for j, b in enumerate(self.blocks_)]
        return np.hstack(cols) if cols else np.zeros((len(X), 0))

    def fit(self, X, y, sample_weight=None, cv=None):
        """Fit the curves.

        Args:
            X, y: numeric design and target.
            sample_weight: optional weight per row.
            cv: optional list of ``(train, test)`` row-index pairs. The two strengths are then
                chosen by the held-out loss on those splits instead of by generalized
                cross-validation, which matters when the splits hold out whole groups: row-wise
                criteria cannot see that a variable constant within groups has only as many
                independent values as there are groups.
        """
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, d = X.shape
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        fam = get_family(self.family)
        self.blocks_ = [
            _Block(X[:, j], self.n_knots, self.degree, self.max_levels) for j in range(d)
        ]
        B = self._design(X)
        self.slices_, i = [], 0
        for b in self.blocks_:
            self.slices_.append(slice(i, i + b.width))
            i += b.width
        P = np.zeros((B.shape[1], B.shape[1]))
        for b, s in zip(self.blocks_, self.slices_):
            P[s, s] = b.penalty()
        lambdas = np.logspace(-2, 4, 7) if self.lambdas is None else np.asarray(self.lambdas)
        shrink = [1e-6, 1e-3, 1e-2, 1e-1, 1.0, 10.0] if self.shrinkage is None else self.shrinkage
        eye = np.eye(B.shape[1])
        scale = float(np.trace(B.T @ (B * w[:, None]))) / max(B.shape[1], 1) / n

        def fit_rows(rows, lam, rho):
            """Penalized fit on ``rows``; returns the coefficients, intercept and edf."""
            Br, yr, wr = B[rows], y[rows], w[rows]
            eta = fam.start(yr, wr)
            for _ in range(1 if fam.name == "gaussian" else 25):
                z, v = fam.working(yr, eta)
                ww = wr * v
                sw = ww / ww.sum()
                mu_B = sw @ Br
                Bc = Br - mu_B
                A = Bc.T @ (Bc * ww[:, None])
                M = A + lam * P + rho * scale * ww.sum() * eye
                beta = np.linalg.solve(M, Bc.T @ (ww * (z - sw @ z)))
                b0 = float(sw @ z - mu_B @ beta)
                new = b0 + Br @ beta
                done = np.max(np.abs(new - eta)) < 1e-6 * (1 + np.max(np.abs(new)))
                eta = new
                if done:
                    break
            edf = float(np.trace(np.linalg.solve(M, A))) + 1.0
            return beta, b0, edf, eta

        every = np.arange(n)
        best = None
        for lam, rho in [(a, b) for a in lambdas for b in shrink]:
            if cv is None:
                beta, b0, edf, eta = fit_rows(every, lam, rho)
                dev = float(np.sum(w * fam.loss(y, fam.mean(eta))))
                n_eff = float(w.sum() ** 2 / np.sum(w**2))
                crit = n_eff * dev / max(n_eff - edf, 1e-9) ** 2
            else:
                crit = 0.0
                for tr, te in cv:
                    beta, b0, _, _ = fit_rows(np.asarray(tr), lam, rho)
                    mu = fam.mean(b0 + B[te] @ beta)
                    crit += float(np.sum(w[te] * fam.loss(y[te], mu)))
            if best is None or crit < best[0] * (1 - 1e-9):
                best = (crit, lam, rho)
        _, self.lambda_, self.shrinkage_ = best
        self.coef_, self.intercept_, self.edf_, _ = fit_rows(every, self.lambda_, self.shrinkage_)
        # centre each variable's curve on the training rows; the intercept absorbs the shift
        self.centers_ = np.array([
            np.average(B[:, s] @ self.coef_[s], weights=w) if s.stop > s.start else 0.0
            for s in self.slices_
        ])  # fmt: skip
        self.intercept_ = float(self.intercept_ + self.centers_.sum())
        self.n_features_in_ = d
        return self

    def curves(self, X):
        """``(d, n)``: each variable's contribution to the additive score at the rows of X."""
        X = np.asarray(X, float)
        out = np.zeros((X.shape[1], len(X)))
        for j, (b, s) in enumerate(zip(self.blocks_, self.slices_)):
            if b.width:
                out[j] = b.design(X[:, j]) @ self.coef_[s] - self.centers_[j]
        return out

    def decision_function(self, X):
        return self.intercept_ + self.curves(X).sum(axis=0)

    def predict(self, X):
        return get_family(self.family).mean(self.decision_function(X))


# -----------------------------------------------------------------------------
# other sources
# -----------------------------------------------------------------------------
class ModelCurves:
    """Curves read off any fitted model: its prediction as one variable moves and the others sit
    at their training mean. Exact for an additive model; for a model with interactions it is the
    curve at the mean of the other variables."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None, sample_weight=None):
        self.ref_ = np.asarray(X, float).mean(axis=0)
        return self

    def curves(self, X):
        X = np.asarray(X, float)
        out = np.empty((X.shape[1], len(X)))
        base = float(np.asarray(self.model.predict(self.ref_[None, :]), float).ravel()[0])
        for j in range(X.shape[1]):
            Z = np.tile(self.ref_, (len(X), 1))
            Z[:, j] = X[:, j]
            out[j] = np.asarray(self.model.predict(Z), float).ravel() - base
        return out

    def predict(self, X):
        return np.asarray(self.model.predict(np.asarray(X, float)), float).ravel()


class KANAdditive:
    """A Kolmogorov-Arnold network with no hidden layer (``layers=(d, 1)``) as a curve source.

    Needs the ``kan`` extra (``pycse``, which brings JAX). The model is the same family of curves
    as :class:`SplineAdditive`, fitted by L-BFGS with L1 and entropy penalties ``reg``.
    """

    def __init__(self, grid_size=5, reg=1e-3, maxiter=300, seed=19):
        self.grid_size = grid_size
        self.reg = reg
        self.maxiter = maxiter
        self.seed = seed

    def fit(self, X, y, sample_weight=None):
        try:
            from pycse.sklearn.kan import KAN
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError('KANAdditive needs pycse: pip install "transferledger[kan]"') from e
        if sample_weight is not None and np.ptp(sample_weight) > 0:
            raise ValueError("KANAdditive does not take sample weights; use SplineAdditive")
        X = np.asarray(X, float)
        self.kan_ = KAN(layers=(X.shape[1], 1), grid_size=self.grid_size, l1_spline=self.reg,
                        entropy_reg=self.reg, seed=self.seed).fit(X, np.asarray(y, float), maxiter=self.maxiter)  # fmt: skip
        self._curves = ModelCurves(self.kan_).fit(X)
        return self

    def curves(self, X):
        return self._curves.curves(X)

    def predict(self, X):
        return np.asarray(self.kan_.predict(np.asarray(X, float)), float).ravel()
