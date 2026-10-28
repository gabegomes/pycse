"""Interpretation toolkit for KAN (and other) models on chemistry data.

The pieces the KAN chemistry notebooks share, so every example runs the same
honest pipeline:

- :func:`cv_compare`: repeated cross-validation of several models on identical
  folds, with a paired comparison of one model against the best of the others.
- :func:`partial_effects` and :func:`partial_effects_cv`: one-descriptor response
  curves in the descriptor's own units, and the envelope of those curves over
  per-fold refits (a curve that changes shape between folds is noise).
- :func:`accumulated_local_effects` and :func:`accumulated_local_effects_cv`: the
  same curves read the way correlated descriptors require, by moving a descriptor
  only within reactions whose value is already near the point being evaluated.
- :func:`prune_collinear`: keep one descriptor from each group of near-duplicates,
  so an effect is not split arbitrarily between interchangeable columns.
- :func:`transitions`: where a curve rises to and falls from its maximum, in the
  descriptor's units (the location of a step or of the edges of a window).
- :func:`sensitivity`: mean absolute finite-difference gradient per descriptor.
- :func:`classify_shape`: shape labels with an explicit linearity criterion.
- :func:`threshold_rules`: single-descriptor cut rules scored by balanced
  accuracy, against the majority-class baseline.
- :func:`plot_partial_effect`: the curve, its fold envelope and the data support,
  in the shared chart style.

Everything takes raw (unscaled) ``X``; scaling happens inside, per fold where it
matters, so that every number reported is in the units a chemist uses.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pycse.sklearn._plotstyle import INK_2, MUTED, PX, SERIES, SURFACE, finish_axes, style_axes

#: A spline is never exactly a line. A curve is "Linear" when a straight line
#: explains at least this fraction of its variance over the data range.
LINEAR_R2 = 0.90


# -----------------------------------------------------------------------------
# prediction helpers
# -----------------------------------------------------------------------------
def _predict(model, X, task):
    """Predicted value (regression) or P(positive class) (classification)."""
    if task == "classification":
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(X))[:, 1]
        return np.clip(np.asarray(model.predict(X), dtype=float).ravel(), 0.0, 1.0)
    return np.asarray(model.predict(X), dtype=float).ravel()


def _fit(model, X, y, task, kwargs):
    if task == "classification" and not hasattr(model, "predict_proba"):
        y = np.asarray(y, dtype=float)  # a regressor used as a probability model
    model.fit(X, y, **(kwargs or {}))
    return model


# -----------------------------------------------------------------------------
# repeated cross-validation on identical folds
# -----------------------------------------------------------------------------
@dataclass
class CVResult:
    """Output of :func:`cv_compare`."""

    folds: pd.DataFrame  #: one row per (fold, model) with the fold's scores
    summary: pd.DataFrame  #: mean and sd per model, in the order given
    pairwise: dict  #: paired comparison of ``focus`` against the best other model
    metric: str  #: the metric the ranking uses ("r2" or "auc")

    def __str__(self):
        lines = [f"{'Model':<22} {self.metric:<16} {'other'}"]
        other = "mae" if self.metric == "r2" else "accuracy"
        for name, row in self.summary.iterrows():
            lines.append(
                f"{name:<22} {row[self.metric + '_mean']:.3f} ± {row[self.metric + '_std']:.3f}   "
                f"{row[other + '_mean']:.3f} ± {row[other + '_std']:.3f} ({other})"
            )
        p = self.pairwise
        if p:
            lines.append(
                f"{p['focus']} vs {p['vs']}, paired over {p['n']} folds: Δ{self.metric} = "
                f"{p['mean']:+.4f} ± {p['sd']:.4f}, {p['focus']} better in {p['wins']}/{p['n']}, "
                f"Wilcoxon p = {p['p']:.3g}"
            )
        return "\n".join(lines)


def cv_compare(
    models,
    X,
    y,
    *,
    task="regression",
    n_splits=5,
    n_repeats=3,
    random_state=42,
    fit_kwargs=None,
    focus="KAN",
    verbose=True,
):
    """Score several models on identical repeated folds.

    Args:
        models: ``{name: unfitted estimator}``; each is cloned per fold. Any object
            with ``get_params`` works (sklearn models, ``KAN``, ...).
        X, y: raw features and target. Features are standardized inside each fold.
        task: ``"regression"`` (R², MAE) or ``"classification"`` (AUC, accuracy; a
            model without ``predict_proba`` is fit on ``y`` as float and its
            clipped prediction is used as the probability).
        n_splits, n_repeats, random_state: the fold design (stratified for
            classification).
        fit_kwargs: ``{name: kwargs}`` passed to that model's ``fit`` (for
            example ``{"KAN": {"maxiter": 300}}``).
        focus: the model to compare pairwise against the best of the others.

    Returns:
        :class:`CVResult`.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    fit_kwargs = fit_kwargs or {}
    if task == "classification":
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        splits = cv.split(X, y)
    else:
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        splits = cv.split(X)

    rows = []
    for fold, (tr, te) in enumerate(splits):
        scaler = StandardScaler().fit(X[tr])
        X_tr, X_te = scaler.transform(X[tr]), scaler.transform(X[te])
        for name, proto in models.items():
            model = _fit(clone(proto), X_tr, y[tr], task, fit_kwargs.get(name))
            pred = _predict(model, X_te, task)
            if task == "classification":
                rows.append(
                    {
                        "fold": fold,
                        "model": name,
                        "auc": roc_auc_score(y[te], pred),
                        "accuracy": accuracy_score(y[te], (pred > 0.5).astype(int)),
                    }
                )
            else:
                rows.append(
                    {
                        "fold": fold,
                        "model": name,
                        "r2": r2_score(y[te], pred),
                        "mae": mean_absolute_error(y[te], pred),
                    }
                )
        if verbose:
            print(f"  fold {fold + 1}/{n_splits * n_repeats} done")

    folds = pd.DataFrame(rows)
    metric = "auc" if task == "classification" else "r2"
    cols = ["auc", "accuracy"] if task == "classification" else ["r2", "mae"]
    summary = folds.groupby("model")[cols].agg(["mean", "std"]).loc[list(models)]
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]

    pairwise = {}
    others = [m for m in models if m != focus]
    if focus in models and others:
        vs = summary.loc[others, f"{metric}_mean"].idxmax()
        a = folds[folds.model == focus].sort_values("fold")[metric].to_numpy()
        b = folds[folds.model == vs].sort_values("fold")[metric].to_numpy()
        d = a - b
        pairwise = {
            "focus": focus,
            "vs": vs,
            "n": int(len(d)),
            "mean": float(d.mean()),
            "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
            "wins": int((d > 0).sum()),
            "p": float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0,
        }
    return CVResult(folds=folds, summary=summary, pairwise=pairwise, metric=metric)


# -----------------------------------------------------------------------------
# partial effects
# -----------------------------------------------------------------------------
@dataclass
class PartialEffect:
    """One descriptor's response curve with the other descriptors at their mean."""

    feature: str
    x_raw: np.ndarray  #: grid in the descriptor's own units
    x_std: np.ndarray  #: the same grid, standardized
    y: np.ndarray  #: prediction along the grid
    support: np.ndarray  #: the raw values observed in the data
    curves: np.ndarray | None = None  #: per-fold refit curves, shape (n_folds, n_grid)
    extra: dict = field(default_factory=dict)  #: other models' curves, keyed by name
    method: str = "partial effect"  #: "partial effect" or "accumulated local effect"

    @property
    def lo(self):
        return None if self.curves is None else self.curves.min(axis=0)

    @property
    def hi(self):
        return None if self.curves is None else self.curves.max(axis=0)

    @property
    def median(self):
        return None if self.curves is None else np.median(self.curves, axis=0)

    @property
    def n_unique(self):
        return int(len(np.unique(np.round(self.support, 6))))

    def shape(self, linear_r2=LINEAR_R2, scale=None):
        return classify_shape(self.x_std, self.y, linear_r2=linear_r2, scale=scale)

    def nonlinearity(self):
        return nonlinearity_score(self.x_std, self.y)


def _grid_curve(model, X_std, j, grid_std, task):
    X_tmp = np.tile(X_std.mean(axis=0), (len(grid_std), 1))
    X_tmp[:, j] = grid_std
    return _predict(model, X_tmp, task)


def partial_effects(
    model,
    X,
    feature_names,
    features=None,
    *,
    scaler=None,
    task="regression",
    n_grid=80,
    extra_models=None,
):
    """Response curves of a fitted model, one per requested feature.

    Args:
        model: fitted on ``scaler.transform(X)`` (or on ``X`` if ``scaler`` is None).
        X: raw features the model was trained on (used for the grid range, the mean
            of the other descriptors and the support).
        feature_names: names for the columns of ``X``.
        features: subset of names to compute (default: all).
        scaler: the fitted ``StandardScaler``; None means the model took raw ``X``.
        task: ``"regression"`` or ``"classification"``.
        extra_models: ``{name: fitted model}`` evaluated on the same grid (for
            example the MLR baseline), stored in ``PartialEffect.extra``.

    Returns:
        ``{feature: PartialEffect}``.
    """
    X = np.asarray(X, dtype=float)
    names = list(feature_names)
    feats = names if features is None else list(features)
    X_std = scaler.transform(X) if scaler is not None else X
    mean = scaler.mean_ if scaler is not None else np.zeros(X.shape[1])
    scale = scaler.scale_ if scaler is not None else np.ones(X.shape[1])
    out = {}
    for f in feats:
        j = names.index(f)
        grid = np.linspace(X_std[:, j].min(), X_std[:, j].max(), n_grid)
        pe = PartialEffect(
            feature=f,
            x_raw=grid * scale[j] + mean[j],
            x_std=grid,
            y=_grid_curve(model, X_std, j, grid, task),
            support=X[:, j].copy(),
        )
        for name, m in (extra_models or {}).items():
            pe.extra[name] = _grid_curve(m, X_std, j, grid, task)
        out[f] = pe
    return out


def partial_effects_cv(
    make_model,
    X,
    y,
    feature_names,
    features=None,
    *,
    task="regression",
    n_splits=5,
    random_state=42,
    fit_kwargs=None,
    n_grid=80,
    effects=None,
    verbose=True,
):
    """Refit a model on each fold and collect every fold's response curves.

    The curves are evaluated on the full-data grid in raw units, through each
    fold's own scaler, so they are comparable to the full-data curve.

    Args:
        make_model: zero-argument callable returning an unfitted model.
        X, y: raw features and target.
        feature_names, features, task, n_grid: as in :func:`partial_effects`.
        n_splits, random_state: the folds (stratified for classification).
        fit_kwargs: kwargs for ``fit``.
        effects: an existing ``{feature: PartialEffect}`` (from
            :func:`partial_effects`) to attach the fold curves to; otherwise a new
            dict with the full-data model's curve left empty.

    Returns:
        ``{feature: PartialEffect}`` with ``curves`` filled in.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    names = list(feature_names)
    feats = names if features is None else list(features)
    full_scaler = StandardScaler().fit(X)
    X_std = full_scaler.transform(X)
    grids = {
        f: np.linspace(X_std[:, names.index(f)].min(), X_std[:, names.index(f)].max(), n_grid)
        for f in feats
    }
    if effects is None:
        effects = {
            f: PartialEffect(
                feature=f,
                x_raw=grids[f] * full_scaler.scale_[names.index(f)]
                + full_scaler.mean_[names.index(f)],
                x_std=grids[f],
                y=np.full(n_grid, np.nan),
                support=X[:, names.index(f)].copy(),
            )
            for f in feats
        }
    if task == "classification":
        splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
            X, y
        )
    else:
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X)
    curves = {f: [] for f in feats}
    for k, (tr, _te) in enumerate(splits):
        scaler = StandardScaler().fit(X[tr])
        model = _fit(make_model(), scaler.transform(X[tr]), y[tr], task, fit_kwargs)
        base = np.tile(X[tr].mean(axis=0), (n_grid, 1))  # raw-space mean of this fold
        for f in feats:
            j = names.index(f)
            X_tmp = base.copy()
            X_tmp[:, j] = effects[f].x_raw
            curves[f].append(_predict(model, scaler.transform(X_tmp), task))
        if verbose:
            print(f"  refit {k + 1}/{n_splits} done")
    for f in feats:
        effects[f].curves = np.array(curves[f])
    return effects


# -----------------------------------------------------------------------------
# accumulated local effects
# -----------------------------------------------------------------------------
def _ale_edges(x, n_bins):
    """Quantile bin edges of one descriptor; fewer when it takes few values."""
    return np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))


def _ale_curve(f, X, j, edges):
    """First-order ALE of column ``j`` at ``edges``, centred on the mean prediction.

    In each bin only the reactions whose value lies in that bin are moved, from
    the bin's lower edge to its upper edge, so the model is never asked about a
    combination of descriptors that does not occur in the data.
    """
    x = X[:, j]
    nb = len(edges) - 1
    bins = np.clip(np.searchsorted(edges, x, side="left") - 1, 0, nb - 1)
    local = np.zeros(nb)
    counts = np.zeros(nb)
    for k in range(nb):
        m = bins == k
        counts[k] = m.sum()
        if not counts[k]:
            continue
        lo, hi = X[m].copy(), X[m].copy()
        lo[:, j], hi[:, j] = edges[k], edges[k + 1]
        local[k] = np.mean(f(hi) - f(lo))
    ale = np.concatenate([[0.0], np.cumsum(local)])
    centre = np.sum(counts * 0.5 * (ale[:-1] + ale[1:])) / max(counts.sum(), 1)
    return ale - centre + float(np.mean(f(X)))


def _as_function(model, scaler, task):
    if scaler is None:
        return lambda Z: _predict(model, Z, task)
    return lambda Z: _predict(model, scaler.transform(Z), task)


def accumulated_local_effects(
    model,
    X,
    feature_names,
    features=None,
    *,
    scaler=None,
    task="regression",
    n_bins=20,
    extra_models=None,
):
    """Accumulated local effect curves of a fitted model, one per requested feature.

    A partial effect fixes every other descriptor at its mean, which describes a
    compound that does not exist, and when descriptors are correlated it asks the
    model about combinations it never saw. The accumulated local effect (Apley and
    Zhu, JRSS B 2020) only moves a descriptor within the reactions that already
    sit near each value, and adds the local changes up. It is the reading to use
    when descriptors are correlated, which in reaction data they almost always are.

    The curve is centred on the model's mean prediction, so it is in the units of
    the target and comparable with group means.

    Args:
        model, X, feature_names, features, scaler, task: as in :func:`partial_effects`.
        n_bins: number of quantile bins (fewer when the descriptor takes fewer values).
        extra_models: ``{name: fitted model}`` evaluated on the same bins.

    Returns:
        ``{feature: PartialEffect}`` with ``method == "accumulated local effect"``.
    """
    X = np.asarray(X, dtype=float)
    names = list(feature_names)
    feats = names if features is None else list(features)
    mean = scaler.mean_ if scaler is not None else np.zeros(X.shape[1])
    scale = scaler.scale_ if scaler is not None else np.ones(X.shape[1])
    f = _as_function(model, scaler, task)
    out = {}
    for name in feats:
        j = names.index(name)
        edges = _ale_edges(X[:, j], n_bins)
        pe = PartialEffect(
            feature=name,
            x_raw=edges,
            x_std=(edges - mean[j]) / scale[j],
            y=_ale_curve(f, X, j, edges),
            support=X[:, j].copy(),
            method="accumulated local effect",
        )
        for label, m in (extra_models or {}).items():
            pe.extra[label] = _ale_curve(_as_function(m, scaler, task), X, j, edges)
        out[name] = pe
    return out


def accumulated_local_effects_cv(
    make_model,
    X,
    y,
    feature_names,
    features=None,
    *,
    task="regression",
    n_splits=5,
    random_state=42,
    fit_kwargs=None,
    n_bins=20,
    effects=None,
    verbose=True,
):
    """Refit on each fold and collect every fold's accumulated local effect curves.

    All folds share the bins of the full data and are evaluated on all reactions,
    so the curves differ only because the model differs. Arguments as in
    :func:`partial_effects_cv`; ``effects`` may come from
    :func:`accumulated_local_effects`.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    names = list(feature_names)
    feats = names if features is None else list(features)
    full = StandardScaler().fit(X)
    if effects is None:
        effects = {}
        for name in feats:
            j = names.index(name)
            edges = _ale_edges(X[:, j], n_bins)
            effects[name] = PartialEffect(
                feature=name,
                x_raw=edges,
                x_std=(edges - full.mean_[j]) / full.scale_[j],
                y=np.full(len(edges), np.nan),
                support=X[:, j].copy(),
                method="accumulated local effect",
            )
    if task == "classification":
        splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
            X, y
        )
    else:
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X)
    curves = {name: [] for name in feats}
    for k, (tr, _te) in enumerate(splits):
        scaler = StandardScaler().fit(X[tr])
        model = _fit(make_model(), scaler.transform(X[tr]), y[tr], task, fit_kwargs)
        f = _as_function(model, scaler, task)
        for name in feats:
            curves[name].append(_ale_curve(f, X, names.index(name), effects[name].x_raw))
        if verbose:
            print(f"  refit {k + 1}/{n_splits} done")
    for name in feats:
        effects[name].curves = np.array(curves[name])
    return effects


# -----------------------------------------------------------------------------
# collinearity and transitions
# -----------------------------------------------------------------------------
def prune_collinear(X, feature_names, threshold=0.85, keep_first=None):
    """Keep one descriptor from each group whose absolute correlation exceeds ``threshold``.

    Descriptors are visited in the order ``keep_first`` (for example the one a
    paper names) followed by the rest in their given order; a descriptor is kept
    when it is not correlated above the threshold with any descriptor already kept.

    Returns:
        ``(kept_names, dropped)`` where ``dropped`` maps each removed descriptor
        to ``(the kept descriptor it duplicates, |r|)``.
    """
    df = pd.DataFrame(np.asarray(X, dtype=float), columns=list(feature_names))
    corr = df.corr().abs().fillna(0.0)
    order = list(dict.fromkeys(list(keep_first or []) + list(feature_names)))
    kept, dropped = [], {}
    for name in order:
        partner = max(kept, key=lambda k: corr.loc[name, k], default=None)
        if partner is not None and corr.loc[name, partner] > threshold:
            dropped[name] = (partner, float(corr.loc[name, partner]))
        else:
            kept.append(name)
    kept = [n for n in feature_names if n in kept]  # original column order
    return kept, dropped


def transitions(x, y, min_change=0.25):
    """Where a curve rises to its maximum and where it falls from it.

    The rise is where the curve, before its maximum, last crosses halfway between
    its earlier minimum and the maximum; the fall is where, after the maximum, it
    first drops through halfway to its later minimum. Both are interpolated
    linearly between grid points, so a coarse grid (such as accumulated local
    effect bins) still locates an edge between its points.
    A side whose change is smaller than ``min_change`` of the curve's range is
    ``None``. A rising step has only a rise, a falling step only a fall, and a
    window both.

    Returns:
        ``{"peak": x at the maximum, "rise": x or None, "fall": x or None}``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    span = np.ptp(y)
    k = int(np.argmax(y))
    out = {"peak": float(x[k]), "rise": None, "fall": None}
    if span <= 0:
        return out

    def cross(i0, i1, level):
        """x where the straight line between grid points i0 and i1 reaches ``level``."""
        y0, y1 = y[i0], y[i1]
        if y1 == y0:
            return float(x[i1])
        return float(x[i0] + (level - y0) * (x[i1] - x[i0]) / (y1 - y0))

    if k > 0 and (y[k] - y[:k].min()) >= min_change * span:
        half = 0.5 * (y[k] + y[:k].min())
        below = np.nonzero(y[:k] <= half)[0]
        if len(below):
            i = int(below[-1])
            out["rise"] = cross(i, i + 1, half)
    if k < len(y) - 1 and (y[k] - y[k:].min()) >= min_change * span:
        half = 0.5 * (y[k] + y[k:].min())
        below = np.nonzero(y[k:] <= half)[0]
        if len(below):
            i = k + int(below[0])
            out["fall"] = cross(i - 1, i, half)
    return out


# -----------------------------------------------------------------------------
# sensitivity and shapes
# -----------------------------------------------------------------------------
def sensitivity(model, X_std, *, task="regression", eps=0.01):
    """Mean absolute finite-difference gradient per feature, over the rows of ``X_std``."""
    X_std = np.asarray(X_std, dtype=float)
    base = _predict(model, X_std, task)
    out = np.zeros(X_std.shape[1])
    for j in range(X_std.shape[1]):
        X_eps = X_std.copy()
        X_eps[:, j] += eps
        out[j] = np.mean(np.abs(_predict(model, X_eps, task) - base)) / eps
    return out


def nonlinearity_score(x, y):
    """RMSE of the best straight line through ``y(x)``, divided by the sd of ``y``."""
    y = np.asarray(y, dtype=float)
    if y.std() < 1e-8 or not np.all(np.isfinite(y)):
        return 0.0
    line = np.polyval(np.polyfit(x, y, 1), x)
    return float(np.sqrt(np.mean((y - line) ** 2)) / y.std())


def _r2(y, fit):
    return 1 - np.sum((y - fit) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)


def _logistic(x, a, b, c, w):
    return a + b / (1.0 + np.exp(-(x - c) / w))


def _window(x, a, b1, c1, w1, b2, c2, w2):
    """Two logistic edges with a plateau between them; each edge has its own height."""
    return a + b1 / (1.0 + np.exp(-(x - c1) / w1)) - b2 / (1.0 + np.exp(-(x - c2) / w2))


def _fit_window(x, y_norm):
    """Best two-edge fit as ``(r2, (a, b1, c1, w1, b2, c2, w2))``, or None.

    Started from the curve's own transitions, for a high plateau and a low one.
    """
    from scipy.optimize import curve_fit

    lo, hi = x.min(), x.max()
    span = hi - lo
    best = None
    for sign, yy in ((1.0, y_norm), (-1.0, -y_norm)):
        tr = transitions(x, yy)
        if tr["rise"] is None or tr["fall"] is None:
            continue
        k = int(np.argmax(yy))
        b1 = (yy[k] - yy[:k].min()) * sign
        b2 = (yy[k] - yy[k:].min()) * sign
        p0 = [yy[:k].min() * sign, b1, tr["rise"], 0.05 * span, b2, tr["fall"], 0.05 * span]
        bounds = (
            [-np.inf, -np.inf, lo - 0.1 * span, 1e-3 * span, -np.inf, lo - 0.1 * span, 1e-3 * span],
            [np.inf, np.inf, hi + 0.1 * span, span, np.inf, hi + 0.1 * span, span],
        )
        try:
            popt, _ = curve_fit(_window, x, y_norm, p0=p0, bounds=bounds, maxfev=8000)
        except (RuntimeError, ValueError):
            continue
        r2 = _r2(y_norm, _window(x, *popt))
        if best is None or r2 > best[0]:
            best = (r2, popt)
    return best


def classify_shape(x, y, linear_r2=LINEAR_R2, scale=None):
    """Label a response curve by the simplest template that explains it.

    Labels: ``Constant``, ``Linear``, ``Step``, ``Saturating``, ``Monotone``,
    ``U-shaped``, ``Inverted U``, ``Window``, ``Gap`` or ``Complex``.

    Three templates are fit by least squares: a line, a parabola and a logistic
    step. "Linear" means the line explains at least ``linear_r2`` of the variance
    of ``y`` over the grid. Otherwise the better of the parabola and the step is
    taken if it reaches the same criterion: a parabola with its vertex inside the
    grid is "U-shaped" or "Inverted U", one that flattens at an end is
    "Saturating", one that does not is "Monotone"; a step whose width is under
    15 % of the grid is "Step", a wider one that levels off inside the grid is
    "Saturating", else "Monotone". A curve that no template explains is
    "Monotone" when it never reverses direction and "Complex" otherwise. Spline
    wiggles on a plateau therefore do not change the label as long as the
    template still explains 90 % of the curve.

    A fourth template has two sharp logistic edges with a plateau between them:
    "Window" when the curve is high between the edges (active only inside a
    range) and "Gap" when it is low. It replaces the simpler labels only when it
    explains at least 5 % more of the variance than both of them, or when they
    fail the criterion and it passes.

    Args:
        x, y: the grid and the curve.
        linear_r2: the criterion for every template.
        scale: when given, a curve whose range is below 5 % of ``scale`` (for
            example the sd of the target) is "Constant".

    Returns:
        ``(label, one-line interpretation)``.
    """
    from scipy.optimize import curve_fit

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.std() < 1e-8 or (scale is not None and np.ptp(y) < 0.05 * scale):
        return "Constant", "No effect"
    y_norm = (y - y.mean()) / y.std()
    line = np.polyval(np.polyfit(x, y_norm, 1), x)
    if _r2(y_norm, line) >= linear_r2:
        return "Linear", "Constant marginal effect"
    lo, hi = x.min(), x.max()
    span = hi - lo
    # parabola
    qa, qb, qc = np.polyfit(x, y_norm, 2)
    r2_quad = _r2(y_norm, np.polyval((qa, qb, qc), x))
    vertex = -qb / (2 * qa) if abs(qa) > 1e-12 else np.inf
    # logistic step
    r2_step, width, center = -np.inf, np.nan, np.nan
    try:
        rising = y_norm[-1] >= y_norm[0]
        p0 = [y_norm.min() if rising else y_norm.max(), np.ptp(y_norm) * (1 if rising else -1),
              x[np.argmin(np.abs(y_norm - y_norm.mean()))], 0.1 * span]  # fmt: skip
        popt, _ = curve_fit(
            _logistic, x, y_norm, p0=p0, maxfev=4000,
            bounds=([-np.inf, -np.inf, lo - span, 1e-3 * span], [np.inf, np.inf, hi + span, 2 * span]),
        )  # fmt: skip
        r2_step, center, width = _r2(y_norm, _logistic(x, *popt)), popt[2], popt[3]
    except (RuntimeError, ValueError):
        pass
    yn = (y_norm - y_norm.min()) / np.ptp(y_norm)
    q = max(2, len(yn) // 4)
    flat_end = abs(yn[-1] - yn[-q]) < 0.1  # levels off over the last quarter of the grid
    net_over_tv = abs(yn[-1] - yn[0]) / (np.sum(np.abs(np.diff(yn))) + 1e-12)
    win = _fit_window(x, y_norm)
    if win is not None:
        r2_win, (_a, b1, c1, w1, b2, c2, w2) = win
        rng = np.ptp(y_norm)
        same_sign = np.sign(b1) == np.sign(b2)
        both_edges = (
            min(abs(b1), abs(b2)) >= 0.25 * rng
        )  # each edge moves the curve by a quarter or more
        sharp = max(w1, w2) <= 0.15 * span and (c2 - c1) >= 0.1 * span
        inside = lo + 0.05 * span < c1 and c2 < hi - 0.05 * span
        better = r2_win >= linear_r2 and (
            max(r2_quad, r2_step) < linear_r2 or r2_win >= max(r2_quad, r2_step) + 0.05
        )
        if same_sign and both_edges and sharp and inside and better:
            if b1 > 0:
                return "Window", "Active only between two thresholds"
            return "Gap", "Inactive only between two thresholds"
    if max(r2_quad, r2_step) < linear_r2:
        if net_over_tv >= 0.95:
            return "Monotone", "Curved but one-directional"
        return "Complex", "Multi-regime behavior"
    if r2_step >= r2_quad:
        if width <= 0.15 * span and lo + 0.05 * span < center < hi - 0.05 * span:
            return "Step", "Threshold transition"
        if flat_end:
            return "Saturating", "Diminishing returns"
        return "Monotone", "Curved but one-directional"
    if lo + 0.1 * span < vertex < hi - 0.1 * span:
        if qa > 0:
            return "U-shaped", "Minimum at an intermediate value"
        return "Inverted U", "Optimum at an intermediate value"
    if flat_end:
        return "Saturating", "Diminishing returns"
    return "Monotone", "Curved but one-directional"


def shape_table(effects, linear_r2=LINEAR_R2, scale=None):
    """One row per :class:`PartialEffect`: shape, interpretation, nonlinearity, support size."""
    rows = []
    for f, pe in effects.items():
        label, note = pe.shape(linear_r2, scale=scale)
        rows.append(
            {
                "Feature": f,
                "Shape": label,
                "Interpretation": note,
                "Nonlinearity": pe.nonlinearity(),
                "N_unique": pe.n_unique,
            }
        )
    return pd.DataFrame(rows).sort_values("Nonlinearity", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# single-descriptor threshold rules
# -----------------------------------------------------------------------------
def threshold_rules(X_train, y_train, feature_names, X_test=None, y_test=None, *, random_state=42):
    """Best single cut per descriptor, scored by balanced accuracy.

    A depth-1 tree with ``class_weight="balanced"`` finds the cut that best
    separates the classes (an unweighted stump on imbalanced data puts the
    majority class in both leaves and scores the majority rate). Thresholds are
    in the units of ``X_train``. Scored on ``X_test, y_test`` when given, else on
    the training data.

    Returns:
        DataFrame sorted by balanced accuracy, plus the majority baseline in
        ``df.attrs["majority_accuracy"]``.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train).astype(int)
    if X_test is None:
        X_test, y_test = X_train, y_train
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test).astype(int)
    rows = []
    for j, name in enumerate(feature_names):
        tree = DecisionTreeClassifier(
            max_depth=1, class_weight="balanced", random_state=random_state
        )
        tree.fit(X_train[:, j : j + 1], y_train)
        if tree.tree_.threshold[0] == -2.0:  # no split possible (constant feature)
            continue
        pred = tree.predict(X_test[:, j : j + 1])
        rows.append(
            {
                "Feature": name,
                "Threshold": float(tree.tree_.threshold[0]),
                "Balanced_Acc": balanced_accuracy_score(y_test, pred),
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
            }
        )
    df = pd.DataFrame(rows).sort_values("Balanced_Acc", ascending=False).reset_index(drop=True)
    df.attrs["majority_accuracy"] = float(max(y_test.mean(), 1 - y_test.mean()))
    return df


# -----------------------------------------------------------------------------
# plotting
# -----------------------------------------------------------------------------
def plot_partial_effect(
    ax, pe, *, ylabel="prediction", xlabel=None, extra_label=None, show_support=True, color=None
):
    """Draw one :class:`PartialEffect`: fold envelope, fold median, the curve, extra models, support.

    The support is a rug (or a thin bar strip when the descriptor takes few
    distinct values) along the bottom of the axes, so the reader can see where
    the curve is interpolating.
    """
    color = color or SERIES[0]
    style_axes(ax)
    x = pe.x_raw
    if pe.curves is not None:
        ax.fill_between(
            x,
            pe.lo,
            pe.hi,
            color=color,
            alpha=0.12,
            linewidth=0,
            label="range of fold refits",
            zorder=1,
        )
        ax.plot(
            x,
            pe.median,
            color=color,
            linewidth=1.5 * PX,
            linestyle=":",
            label="median of refits",
            zorder=2,
        )
    for k, (name, yv) in enumerate(pe.extra.items()):
        ax.plot(
            x, yv, color=SERIES[1 + k % 2], linewidth=2 * PX, label=extra_label or name, zorder=3
        )
    if np.all(np.isfinite(pe.y)):
        ax.plot(x, pe.y, color=color, linewidth=2.5 * PX, label=f"KAN {pe.method}", zorder=4)
    if show_support:
        vals, counts = np.unique(np.round(pe.support, 6), return_counts=True)
        y0, y1 = ax.get_ylim()
        if len(vals) <= 30:
            h = 0.05 * (y1 - y0) * counts / counts.max()
            ax.bar(
                vals,
                h,
                bottom=y0,
                width=(x.max() - x.min()) / 120,
                color=MUTED,
                alpha=0.5,
                linewidth=0,
                zorder=2,
            )
        else:
            ax.plot(
                pe.support,
                np.full(len(pe.support), y0),
                "|",
                color=MUTED,
                alpha=0.4,
                markersize=6,
                zorder=2,
            )
        ax.set_ylim(y0, y1)
    label, _ = pe.shape()
    finish_axes(
        ax,
        f"{pe.feature}  ({label}, {pe.n_unique} distinct values)",
        xlabel=xlabel or pe.feature,
        ylabel=ylabel,
    )
    ax.title.set_fontsize(10)
    return ax


def plot_fold_scores(ax, cv_result, models=None):
    """Per-fold scores for two or more models on the same folds (paired view)."""
    style_axes(ax)
    folds = cv_result.folds
    names = models or list(cv_result.summary.index)
    for k, name in enumerate(names):
        sub = folds[folds.model == name].sort_values("fold")
        ax.plot(
            sub["fold"] + 1,
            sub[cv_result.metric],
            marker="o",
            markersize=5,
            color=SERIES[k % len(SERIES)] if k < len(SERIES) else INK_2,
            markeredgecolor=SURFACE,
            linewidth=1.5 * PX,
            label=name,
        )
    finish_axes(
        ax,
        f"Per-fold {cv_result.metric.upper()} on identical folds",
        xlabel="fold",
        ylabel=cv_result.metric.upper(),
    )
    return ax
