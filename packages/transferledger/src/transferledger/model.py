"""The piece model: an equation that is a sum (and a few products) of named one-variable shapes.

Fitting has four stages, each replaceable:

1. a *curve source* (default: penalized additive splines) gives one curve per variable;
2. each curve is matched to the simplest named form that reproduces it, which makes a *piece*:
   a unit shape of that variable in its own units;
3. variables enter the equation one at a time while their pieces improve the held-out loss
   under a chosen *shift* (default: random folds), so a variable can be required to help on
   cases unlike the training cases. Inside each selection fold the shapes are fitted on that
   fold's training rows only: a shape fitted on all rows has already seen the held-out cases;
4. optionally, products of the chosen pieces are admitted by the same rule. A product is one
   variable changing the effect of another.

The curve source only proposes shapes; the equation's weights are fitted to the data.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t
from sklearn.base import BaseEstimator, RegressorMixin

from ._data import Design
from .curves import KANAdditive, SplineAdditive, get_family
from .forms import DEFAULT_ORDER, match_curve
from .shifts import Reference, Shift

__all__ = ["PieceModel", "fit_weights", "linear_score", "greedy", "make_pieces", "piece_matrix"]


# -----------------------------------------------------------------------------
# the linear stage, shared with the ledger
# -----------------------------------------------------------------------------
def fit_weights(P, y, w, fam, ridge=1e-8):
    """Intercept and weights of ``y ~ 1 + P`` for a family; returns ``coef`` with the intercept
    first. Gaussian is weighted least squares; other families iterate it."""
    A = np.column_stack([np.ones(len(y)), P])
    if fam.name == "gaussian":
        sw = np.sqrt(w)
        return np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    pen = np.eye(A.shape[1])
    pen[0, 0] = 0.0
    eta = fam.start(y, w)
    coef = np.zeros(A.shape[1])
    for _ in range(50):
        z, v = fam.working(y, eta)
        ww = w * v
        M = A.T @ (A * ww[:, None])
        # a small ridge keeps the weights finite when a piece separates the classes
        coef = np.linalg.solve(M + max(ridge, 1e-6) * np.trace(M) / len(M) * pen, A.T @ (ww * z))
        new = np.clip(A @ coef, -30, 30)
        if np.max(np.abs(new - eta)) < 1e-7 * (1 + np.max(np.abs(new))):
            eta = new
            break
        eta = new
    return coef


def linear_score(P, coef):
    return coef[0] + P @ coef[1:]


def greedy(
    candidates, loss_rows, units, *, start=(), max_terms=12, min_gain=0.005, min_evidence=0.0
):
    """Forward selection by held-out skill, with an evidence rule.

    ``loss_rows(selection)`` is the held-out loss of every test row with the items of
    ``selection`` in the model. Skill is ``1 - loss / loss of the empty model``. A candidate is
    admissible when it raises the skill by at least ``min_gain`` *and* that gain is at least
    ``min_evidence`` standard errors above zero, the standard error taken over ``units`` (the
    groups, when whole groups are held out: twelve groups cannot vouch for much). Among the
    admissible candidates the one with the largest gain is added, until none is left.

    Returns:
        ``(added, path)``: the added candidates in order, and the skill after each.
    """
    chosen, pool = list(start), [c for c in candidates if c not in start]
    _, unit_idx = np.unique(np.asarray(units), return_inverse=True)
    n_units = int(unit_idx.max()) + 1
    # the same confidence as ``min_evidence`` normal standard errors, with as many degrees of
    # freedom as there are units to spare: four groups have to agree much more than forty
    need = 0.0
    if min_evidence > 0:
        need = float(student_t.ppf(norm.cdf(min_evidence), max(n_units - 1, 1)))
    base = float(loss_rows([]).sum())
    current = loss_rows(chosen)
    added, path = [], []
    while pool and len(added) < max_terms:
        best = None
        for c in pool:
            d = current - loss_rows(chosen + [c])
            gain = float(d.sum()) / base if base > 0 else 0.0
            if gain < min_gain or (best is not None and gain <= best[1]):
                continue
            if min_evidence > 0 and n_units > 1:
                du = np.bincount(unit_idx, weights=d, minlength=n_units)
                se = np.sqrt(n_units) * du.std(ddof=1)
                if se > 0 and du.sum() / se < need:
                    continue
            best = (c, gain)
        if best is None:
            break
        chosen.append(best[0])
        added.append(best[0])
        pool.remove(best[0])
        current = loss_rows(chosen)
        path.append(1.0 - float(current.sum()) / base if base > 0 else 0.0)
    return added, path


def _forms_for(forms, name, source_name):
    if forms is None:
        return DEFAULT_ORDER
    if isinstance(forms, dict):
        return forms.get(name, forms.get(source_name, forms.get("*", DEFAULT_ORDER)))
    return list(forms)


def make_pieces(curves, X, names, sources, *, weights=None, forms=None, min_r2=0.97, min_side=3):
    """Match every variable's curve to a form; returns the list of pieces.

    Args:
        curves: ``(d, n)`` curves from a curve source.
        X: the ``(n, d)`` numeric design.
        names, sources: design column names, and the input column each came from.
        forms: None (default order), a list for every variable, or ``{name: list}``.
    """
    if not np.all(np.isfinite(curves)):
        bad = [names[j] for j in range(len(names)) if not np.all(np.isfinite(curves[j]))]
        raise ValueError(
            f"the curve source returned non-finite curves for {bad[:5]}; its fit diverged"
        )
    pieces = []
    for j, name in enumerate(names):
        if np.ptp(curves[j]) < 1e-12 * (1.0 + np.abs(curves).max()):
            continue
        for p in match_curve(name, X[:, j], curves[j], weights=weights, column=j, min_r2=min_r2,
                             min_side=min_side, forms=_forms_for(forms, name, sources[j])):  # fmt: skip
            p.group = sources[j]
            pieces.append(p)
    return pieces


def piece_matrix(pieces, X):
    if not pieces:
        return np.zeros((len(X), 0))
    return np.column_stack([p.unit(X[:, p.column]) for p in pieces])


def _make_source(source, family):
    if isinstance(source, str):
        if source == "spline":
            return SplineAdditive(family=family)
        if source == "kan":
            if family != "gaussian":
                raise ValueError("the KAN source fits squared error only; use source='spline'")
            return KANAdditive()
        raise ValueError(
            f"unknown source {source!r}; use 'spline', 'kan' or an object with fit and curves"
        )
    return copy.deepcopy(source)


_LINK_TEXT = {"gaussian": "y", "binomial": "log-odds(y)", "poisson": "log(mean y)"}


class PieceModel(RegressorMixin, BaseEstimator):
    """An equation of named one-variable pieces, fitted to tabular data.

    Args:
        source: where the curves come from: ``"spline"`` (penalized additive splines; exact,
            repeatable, fast), ``"kan"`` (an additive Kolmogorov-Arnold network; needs the
            ``kan`` extra), or any object with ``fit(X, y, sample_weight)`` and ``curves(X)``.
        family: ``"gaussian"`` (a number), ``"binomial"`` (a 0/1 outcome; the equation is for the
            log-odds and ``predict`` returns a probability) or ``"poisson"`` (a count).
        forms: the forms to try: None for the default order, a list for every variable, or a
            dict ``{variable: [forms]}`` for a relationship the field already names, such as
            ``{"T_K": ["inverse"]}`` for an Arrhenius term.
        min_r2: share of a curve's variance a form must reproduce to be accepted.
        min_side: distinct values required on each side of an edge.
        select_on: a :class:`~transferledger.shifts.Shift` over the training rows. Variables are
            admitted while they improve the held-out loss under it. None means random folds.
        n_splits: folds used for that.
        max_variables, min_gain: stop after this many variables, or when the next adds less
            skill.
        min_evidence: a variable is admitted only if its gain is at least this many standard
            errors above zero, the standard error taken over the shift's units (groups, when
            groups are held out). With few groups a chance association is real within the
            sample, and only its uncertainty gives it away. 0 turns the rule off.
        products: also admit products of two chosen variables' leading pieces, by the same rule
            (a product is one variable changing the effect of another).
        max_products: at most this many.
        tune_source: let the curve source tune its strengths on the same splits (a spline source
            does; it matters when the splits hold out whole groups).
        refit_shapes: fit the shapes inside each selection fold on that fold's training rows
            (the honest choice, and the default). False reuses the all-row shapes, which is
            faster and lets a shape that has seen the held-out cases vouch for itself.
        limit: predictions are kept within the range of the training target, widened on each
            side by this fraction of the range. An equation makes no claim far outside anything
            it was fitted to, and near-duplicate variables can otherwise send a new case to an
            absurd value. None removes the limit; it does not apply to a 0/1 target.
        feature_names: column names when ``X`` is an array.
        random_state: seed of the selection folds.

    Attributes:
        pieces_: every candidate piece, fitted on all rows. ``chosen_``: indices of those in the
        equation, grouped by variable in the order the variables entered. ``products_``: pairs of
        indices into ``pieces_``. ``intercept_``, ``coef_`` (one per chosen piece, then one per
        product). ``skill_``: held-out skill under ``select_on`` when selection stopped.
        ``path_``: that skill after each addition. ``source_``, ``design_``.
    """

    def __init__(self, source="spline", family="gaussian", forms=None, min_r2=0.97, min_side=3,
                 select_on=None, n_splits=5, max_variables=12, min_gain=0.005, min_evidence=1.645,
                 products=True,
                 max_products=3, tune_source=True, refit_shapes=True, limit=0.5,
                 feature_names=None, random_state=0):  # fmt: skip
        self.source = source
        self.family = family
        self.forms = forms
        self.min_r2 = min_r2
        self.min_side = min_side
        self.select_on = select_on
        self.n_splits = n_splits
        self.max_variables = max_variables
        self.min_gain = min_gain
        self.min_evidence = min_evidence
        self.products = products
        self.max_products = max_products
        self.tune_source = tune_source
        self.refit_shapes = refit_shapes
        self.limit = limit
        self.feature_names = feature_names
        self.random_state = random_state

    # ---- fitting ------------------------------------------------------------------------------
    def _shift(self, n, y):
        shift = self.select_on
        if shift is None:
            shift = Reference(n, stratify=y if self.family == "binomial" else None)
        if not isinstance(shift, Shift):
            raise TypeError("select_on must be a Shift (for example NewGroups(groups)) or None")
        if len(shift) != n:
            raise ValueError(f"select_on covers {len(shift)} rows but X has {n}")
        return shift

    def _pieces_on(self, source, Xn, y, w, rows):
        """Fit ``source`` on ``rows`` and match its curves there: the pieces those rows support."""
        weighted = np.ptp(w) > 0
        if isinstance(source, SplineAdditive):
            source.fit(Xn[rows], y[rows], sample_weight=w[rows])
        else:
            source.fit(Xn[rows], y[rows], sample_weight=w[rows] if weighted else None)
        return make_pieces(
            source.curves(Xn[rows]), Xn[rows], self.design_.names_, self.design_.source_,
            weights=w[rows], forms=self.forms, min_r2=self.min_r2, min_side=self.min_side,
        )  # fmt: skip

    def fit(self, X, y, sample_weight=None):
        self.design_ = Design(self.feature_names).fit(X)
        Xn = self.design_.transform(X)
        y = np.asarray(y, float).ravel()
        if len(y) != len(Xn):
            raise ValueError(f"X has {len(Xn)} rows and y has {len(y)}")
        if not np.all(np.isfinite(y)):
            raise ValueError("y has missing or infinite values")
        fam = get_family(self.family)
        if fam.name == "binomial" and not np.all(np.isin(y, (0.0, 1.0))):
            raise ValueError("family='binomial' needs a 0/1 target")
        if fam.name == "poisson" and np.any(y < 0):
            raise ValueError("family='poisson' needs a non-negative target")
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        shift = self._shift(len(y), y)
        splits = shift.splits(self.n_splits, self.random_state)
        every = np.arange(len(y))

        # the curve source on all rows; a spline source tunes its two strengths on the splits
        self.source_ = _make_source(self.source, self.family)
        if isinstance(self.source_, SplineAdditive) and self.tune_source:
            self.source_.fit(Xn, y, sample_weight=w, cv=splits)
            self.source_.set_params(
                lambdas=[self.source_.lambda_], shrinkage=[self.source_.shrinkage_]
            )
        self.pieces_ = self._pieces_on(self.source_, Xn, y, w, every)
        P = piece_matrix(self.pieces_, Xn)

        # per selection fold: shapes fitted on that fold's training rows only
        folds = []
        for tr, te in splits:
            pcs = self.pieces_ if not self.refit_shapes else self._pieces_on(
                copy.deepcopy(self.source_), Xn, y, w, tr
            )  # fmt: skip
            by_var = {}
            for c, p in enumerate(pcs):
                by_var.setdefault(p.group, []).append(c)
            folds.append({
                "tr": tr, "te": te, "by_var": by_var, "Ptr": piece_matrix(pcs, Xn[tr]),
                "Pte": piece_matrix(pcs, Xn[te]),
            })  # fmt: skip

        def columns(fold, which, selection):
            Pm = fold[which]
            cols = []
            for item in selection:
                if isinstance(item, tuple):  # a product of two variables' leading pieces
                    a, b = (fold["by_var"].get(v) for v in item)
                    if a and b:
                        cols.append(Pm[:, a[0]] * Pm[:, b[0]])
                else:
                    cols.extend(Pm[:, c] for c in fold["by_var"].get(item, []))
            return np.column_stack(cols) if cols else np.zeros((len(Pm), 0))

        def loss_rows(selection):
            out = []
            for f in folds:
                coef = fit_weights(columns(f, "Ptr", selection), y[f["tr"]], w[f["tr"]], fam)
                mu = fam.mean(linear_score(columns(f, "Pte", selection), coef))
                out.append(w[f["te"]] * fam.loss(y[f["te"]], mu))
            return np.concatenate(out)

        units = np.concatenate([shift.units()[te] for _, te in splits])
        variables = list(dict.fromkeys(p.group for p in self.pieces_))
        rule = dict(min_gain=self.min_gain, min_evidence=self.min_evidence)
        chosen, self.path_ = greedy(
            variables, loss_rows, units, max_terms=self.max_variables, **rule
        )
        pairs = []
        if self.products and len(chosen) > 1:
            cand = [(a, b) for i, a in enumerate(chosen) for b in chosen[i + 1 :]]
            pairs, path = greedy(
                cand, loss_rows, units, start=chosen, max_terms=self.max_products, **rule
            )
            self.path_ = self.path_ + path
        self.skill_ = self.path_[-1] if self.path_ else 0.0

        # the equation: all-row shapes of the chosen variables, weights fitted to all rows
        lead = {}
        for c, p in enumerate(self.pieces_):
            lead.setdefault(p.group, c)
        self.chosen_ = [c for v in chosen for c, p in enumerate(self.pieces_) if p.group == v]
        self.products_ = [(lead[a], lead[b]) for a, b in pairs]
        cols = [P[:, c] for c in self.chosen_] + [P[:, a] * P[:, b] for a, b in self.products_]
        coef = fit_weights(np.column_stack(cols) if cols else np.zeros((len(y), 0)), y, w, fam)
        self.intercept_, self.coef_ = float(coef[0]), np.asarray(coef[1:], float)
        # for support(): the chosen variables' training values, standardized
        self._used = sorted({self.pieces_[j].column for j in self.chosen_})
        if self._used:
            self._mu, self._sd = Xn[:, self._used].mean(axis=0), Xn[:, self._used].std(axis=0)
            self._sd = np.where(self._sd > 0, self._sd, 1.0)
            self._train = (Xn[:, self._used] - self._mu) / self._sd
        self.target_range_ = (float(y.min()), float(y.max()))
        self.n_features_in_ = self.design_.n_features_in_
        self.feature_names_in_ = np.asarray(self.design_.feature_names_in_, dtype=object)
        return self

    # ---- prediction ---------------------------------------------------------------------------
    def _columns(self, Xn):
        cols = [self.pieces_[j].unit(Xn[:, self.pieces_[j].column]) for j in self.chosen_]
        for a, b in self.products_:
            cols.append(
                self.pieces_[a].unit(Xn[:, self.pieces_[a].column])
                * self.pieces_[b].unit(Xn[:, self.pieces_[b].column])
            )
        return np.column_stack(cols) if cols else np.zeros((len(Xn), 0))

    def decision_function(self, X):
        """The equation's value: the prediction itself for a Gaussian family, the log-odds or
        log-mean otherwise."""
        Xn = self.design_.transform(X)
        return self.intercept_ + self._columns(Xn) @ self.coef_

    def predict(self, X):
        mu = get_family(self.family).mean(self.decision_function(X))
        if self.limit is not None and self.family != "binomial":
            lo, hi = self.target_range_
            pad = self.limit * (hi - lo)
            mu = np.clip(mu, max(lo - pad, 0.0) if self.family == "poisson" else lo - pad, hi + pad)
        return mu

    def components(self, X):
        """Each term's contribution to the equation's value at the rows of ``X``."""
        Xn = self.design_.transform(X)
        return pd.DataFrame(self._columns(Xn) * self.coef_, columns=self._term_names())

    def support(self, X):
        """Whether each row is inside what the equation was fitted on.

        Returns a table with, per row: ``outside``, how many of the equation's variables lie
        beyond the range seen in training (their pieces are held at the edge value);
        ``variables``, which ones; ``unseen_level``, whether a categorical value was never seen;
        and ``distance``, the distance to the nearest training row in the standardized space of
        the equation's variables (zero for a training row).
        """
        Xn = self.design_.transform(X)
        unseen = self.design_.unseen_level_.copy()
        out_names = [[] for _ in range(len(Xn))]
        for c in self._used:
            p = next(self.pieces_[j] for j in self.chosen_ if self.pieces_[j].column == c)
            for i in np.flatnonzero(p.outside(Xn[:, c])):
                out_names[i].append(self.design_.names_[c])
        if self._used:
            from sklearn.neighbors import NearestNeighbors

            Z = (Xn[:, self._used] - self._mu) / self._sd
            dist = NearestNeighbors(n_neighbors=1).fit(self._train).kneighbors(Z)[0][:, 0]
            dist = dist / np.sqrt(len(self._used))
        else:
            dist = np.zeros(len(Xn))
        return pd.DataFrame({
            "outside": [len(v) for v in out_names], "variables": [", ".join(v) for v in out_names],
            "unseen_level": unseen, "distance": dist,
        })  # fmt: skip

    # ---- reading ------------------------------------------------------------------------------
    def _term_names(self):
        names = [f"p{k}" for k in range(len(self.chosen_))]
        index = {j: k for k, j in enumerate(self.chosen_)}
        return names + [f"p{index[a]}*p{index[b]}" for a, b in self.products_]

    def _term_variables(self):
        """For each column of the equation, the input columns it involves."""
        owners = [{self.pieces_[j].group} for j in self.chosen_]
        return owners + [{self.pieces_[a].group, self.pieces_[b].group} for a, b in self.products_]

    @property
    def variables_(self):
        """The input columns the equation uses, in the order they entered."""
        return list(dict.fromkeys(self.pieces_[j].group for j in self.chosen_))

    @property
    def terms_(self):
        """One row per term: its name, variable, form, the location the form names, its weight."""
        rows = []
        for k, j in enumerate(self.chosen_):
            p = self.pieces_[j]
            rows.append({"term": f"p{k}", "variable": p.group, "form": p.form, "location": p.location,
                         "weight": self.coef_[k], "shape": p.text()})  # fmt: skip
        for (a, b), name, c in zip(self.products_, self._term_names()[len(self.chosen_):], self.coef_[len(self.chosen_):]):  # fmt: skip
            rows.append({"term": name, "variable": f"{self.pieces_[a].group} x {self.pieces_[b].group}",
                         "form": "product", "location": None, "weight": c, "shape": name})  # fmt: skip
        return pd.DataFrame(
            rows, columns=["term", "variable", "form", "location", "weight", "shape"]
        )

    def equation(self):
        """The fitted equation, written out with every piece in its variable's own units."""
        lines = ["pieces:"] + [
            f"  p{k} = {self.pieces_[j].text()}" for k, j in enumerate(self.chosen_)
        ]
        terms = " ".join(f"{c:+.4g}*{n}" for n, c in zip(self._term_names(), self.coef_))
        lines.append(
            f"{_LINK_TEXT[get_family(self.family).name]} = {self.intercept_:.4g} {terms}".rstrip()
        )
        notes = list(
            dict.fromkeys(self.pieces_[j].note for j in self.chosen_ if self.pieces_[j].note)
        )
        if notes:
            lines.append("  where " + "; ".join(notes))
        return "\n".join(lines)
