"""Is there anything about a unit to explain?

A verdict of "does not transfer" for the description of a unit (a ligand, a site, a patient) is
only informative when units differ. In a table where the outcome is settled by the other
variables, no description of the unit can add anything, and a test of one is empty. This
happened on an outside table: every active Ni reaction also ran without the metal, so a property
of the ligand had nothing to mark.

:func:`unit_ceiling` measures that before any description is tried. On random folds, where
every unit is seen in training, it compares three models: the variables that change within a
unit, the same plus the unit's description, and the same plus the unit's identity (one shrunken
level per unit). What identity adds is the ceiling for any description of the unit; what the
description adds is the part of it the descriptors reach, as labels of units already seen.
Whether that part carries to a new unit is the ledger's question.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from ._data import Design
from .curves import SplineAdditive, get_family
from .shifts import NewGroups

__all__ = ["unit_ceiling", "Ceiling"]

# scikit-learn 1.9 moved the encoder's shuffling into the ``cv`` argument
_CV_OBJECT = "shuffle" in inspect.signature(TargetEncoder.__init__).parameters and (
    inspect.signature(TargetEncoder.__init__).parameters["shuffle"].default == "deprecated"
)


def _level_encoder(binary, seed):
    """The cross-fitted mean outcome per unit; the same folds on every scikit-learn version."""
    kind = "binary" if binary else "continuous"
    if not _CV_OBJECT:
        return TargetEncoder(target_type=kind, random_state=seed)
    folds = (StratifiedKFold if binary else KFold)(5, shuffle=True, random_state=seed)
    return TargetEncoder(target_type=kind, cv=folds)


WITHOUT, DESCRIPTION, IDENTITY = "without the unit", "with its description", "with its identity"


@dataclass
class Ceiling:
    """Result of :func:`unit_ceiling`.

    Attributes:
        skill: one row per model and kind ("additive", "any"): skill on held-out rows of units
            seen in training, with interval.
        adds: what identity and description add over the model without the unit, with interval.
        unit: what a unit is called.
        described_by: the columns taken as the unit's description.
        tol: the smallest addition that counts.
    """

    skill: pd.DataFrame
    adds: pd.DataFrame
    unit: str
    described_by: list
    tol: float

    def _row(self, what, kind):
        t = self.adds
        return t[(t.adds == what) & (t.kind == kind)].iloc[0]

    def ceiling(self, kind=None):
        """What knowing the unit adds: the larger of the additive and the any-model figure, or
        the one named by ``kind``."""
        rows = [self._row("identity", k) for k in ([kind] if kind else self.adds.kind.unique())]
        return max(rows, key=lambda r: r.contribution)

    def state(self):
        """``"units differ"`` when identity adds at least ``tol`` with an interval above zero in
        either kind of model, else ``"nothing to explain"``."""
        for k in self.adds.kind.unique():
            r = self._row("identity", k)
            if r.lo > 0 and r.contribution >= self.tol:
                return "units differ"
        return "nothing to explain"

    def captured(self):
        """Share of the ceiling that the description reaches on units already seen (None when
        there is nothing to explain or no description). Above 1 is possible: a description is
        shared between units, an identity has to be learned unit by unit. With few units and
        several descriptors it is close to 1 almost by construction, because the descriptors
        then pin down each unit and act as its label; a low share is the informative case."""
        if self.state() != "units differ" or not self.described_by:
            return None
        top = self.ceiling()
        return float(self._row("description", top.kind).contribution / top.contribution)

    def to_dict(self):
        return {
            "unit": self.unit,
            "state": self.state(),
            "captured": self.captured(),
            "described_by": list(self.described_by),
            "skill": self.skill.to_dict("records"),
            "adds": self.adds.to_dict("records"),
        }

    def __str__(self):
        with pd.option_context("display.width", 200, "display.float_format", "{:+.3f}".format):
            t = self.skill.assign(
                skill=[f"{s:.3f} [{a:.3f}, {b:.3f}]" for s, a, b in
                       zip(self.skill.skill, self.skill.lo, self.skill.hi)]
            ).pivot(index="model", columns="kind", values="skill")  # fmt: skip
            t = t.reindex([m for m in (WITHOUT, DESCRIPTION, IDENTITY) if m in t.index])
            lines = [
                f"Skill on held-out rows of {self.unit}s seen in training (95 % intervals over "
                f"{self.unit}s)",
                t.to_string(),
                "",
            ]
        lines.append(self.verdict())
        return "\n".join(lines)

    def verdict(self):
        """The reading in words: what identity adds, and either that there is nothing to explain
        or how much of it the description reaches."""
        top = self.ceiling()
        lines = [
            f"Knowing which {self.unit} it is adds {top.contribution:+.3f} "
            f"[{top.lo:+.3f}, {top.hi:+.3f}] ({top.kind} model)."
        ]
        if self.state() == "nothing to explain":
            lines.append(
                f"Nothing to explain: once the other variables are known, {self.unit}s do not "
                f"differ by an amount that can be told from zero. No description of a {self.unit} "
                "can be tested on this table."
            )
        elif self.described_by:
            d = self._row("description", top.kind)
            lines.append(
                f"Its description adds {d.contribution:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}], "
                f"{self.captured():.0%} of that, on {self.unit}s already seen. Whether it carries "
                f"to a new {self.unit} is the ledger's verdict."
            )
        return "\n".join(lines)


def _boosting(fam):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    if fam.name == "binomial":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
    loss = "poisson" if fam.name == "poisson" else "squared_error"
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, loss=loss, random_state=0)


def _fold(XA, XD, code, y, w, tr, te, fam, seed):
    from .ledger import _loss, _predict

    ytr, wtr = y[tr], w[tr]
    out = {"rows": te, "loss": {}}
    out["base"] = _loss(fam, y[te], np.full(len(te), np.average(ytr, weights=wtr)), w[te])
    U = np.zeros((len(y), int(code.max()) + 1))
    U[np.arange(len(y)), code] = 1.0
    U = U[:, U[tr].sum(axis=0) > 0]  # a unit absent from training has no level to learn
    designs = {WITHOUT: XA, IDENTITY: np.hstack([XA, U])}
    if XD.shape[1]:
        designs[DESCRIPTION] = np.hstack([XA, XD])
    for name, Z in designs.items():
        if Z.shape[1] == 0:
            mu = np.full(len(te), np.average(ytr, weights=wtr))
        elif name == IDENTITY:
            # the unit levels get shrinkage of their own, chosen on held-out rows
            perm = np.random.default_rng(seed).permutation(len(tr))
            inner = [(np.setdiff1d(np.arange(len(tr)), p), p) for p in np.array_split(perm, 5)]
            mask = np.r_[np.zeros(XA.shape[1], bool), np.ones(U.shape[1], bool)]
            m = SplineAdditive(family=fam.name).fit(Z[tr], ytr, wtr, cv=inner, cv_features=mask)
            mu = m.predict(Z[te])
        else:
            mu = SplineAdditive(family=fam.name).fit(Z[tr], ytr, wtr).predict(Z[te])
        out["loss"][(name, "additive")] = _loss(fam, y[te], mu, w[te])
    # gradient boosting: the unit enters as its level (the mean outcome of its training rows,
    # cross-fitted so that a row does not see itself) and as a category. The level is what lets
    # a unit with a handful of rows count; a category that thin is ignored by the trees
    yb = ytr.astype(int) if fam.name == "binomial" else ytr
    enc = _level_encoder(fam.name == "binomial", seed)
    constant = fam.name == "binomial" and len(np.unique(yb)) < 2
    level = np.zeros((len(y), 1))
    if not constant:
        level[tr] = enc.fit_transform(code[tr].reshape(-1, 1), yb)
        level[te] = enc.transform(code[te].reshape(-1, 1))
    boosted = {WITHOUT: XA, IDENTITY: np.column_stack([XA, level, code])}
    if XD.shape[1]:
        boosted[DESCRIPTION] = np.hstack([XA, XD])
    for name, Z in boosted.items():
        if Z.shape[1] == 0 or constant:
            mu = np.full(len(te), np.average(ytr, weights=wtr))
        else:
            est = _boosting(fam)
            if name == IDENTITY and code.max() < 255:
                est.set_params(categorical_features=[Z.shape[1] - 1])
            mu = _predict(est.fit(Z[tr], yb, sample_weight=wtr), Z[te], fam)
        out["loss"][(name, "any")] = _loss(fam, y[te], mu, w[te])
    return out


def unit_ceiling(X, y, shift, *, family="gaussian", describe=None, sample_weight=None,
                 n_splits=5, n_repeats=1, level=0.95, tol=0.01, random_state=0, n_jobs=None):  # fmt: skip
    """Measure what knowing the unit adds, and how much of it the unit's description reaches.

    Run it before reading a ledger verdict for a block of unit descriptors. ``state()`` of
    ``"nothing to explain"`` means the table cannot test any description of the unit, however
    many units it has; ``"units differ"`` means it can, and ``captured()`` says how much of the
    difference the description reaches on units already seen.

    Two kinds of model are scored: "additive" (the package's curves, where a unit shifts the
    score by one amount) and "any" (gradient boosting, where a unit may also change how the
    other variables act). The boosted model needs rows to learn a unit: with a handful of rows
    per unit it shows little, and the additive figure is the one to read. Rows are dealt into folds within each unit, so every unit with at
    least two rows is seen in training.

    Args:
        X: table or array of predictors.
        y: the target.
        shift: the :class:`~transferledger.shifts.NewGroups` whose groups are the units.
        family: ``"gaussian"``, ``"binomial"`` or ``"poisson"``.
        describe: columns that describe the unit. Default: every column that is constant
            within each unit.
        sample_weight: optional weight per row.
        n_splits: folds.
        n_repeats: repeats of the deal.
        level: coverage of the intervals (delete-one-unit jackknife, as in the ledger).
        tol: the smallest addition that counts.
        random_state: seed for the folds.
        n_jobs: parallel workers over folds (joblib).

    Returns:
        :class:`Ceiling`.
    """
    from .ledger import _intervals

    if not isinstance(shift, NewGroups):
        raise TypeError("unit_ceiling needs a NewGroups shift: its groups are the units")
    fam = get_family(family)
    frame = Design()._frame(X)
    n = len(frame)
    if len(shift) != n:
        raise ValueError(f"the shift covers {len(shift)} rows but X has {n}")
    y = np.asarray(y, float).ravel()
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
    labels, code = np.unique(shift.groups, return_inverse=True)
    if describe is None:
        describe = [c for c in frame.columns if (frame[c].groupby(code).nunique() <= 1).all()]
    describe = [str(c) for c in describe]
    missing = [c for c in describe if c not in {str(k) for k in frame.columns}]
    if missing:
        raise ValueError(f"describe names columns that X does not have: {missing}")
    inside = [c for c in frame.columns if str(c) not in describe]
    XA = Design().fit_transform(frame[inside]) if inside else np.zeros((n, 0))
    XD = Design().fit_transform(frame[describe]) if describe else np.zeros((n, 0))
    XD = XD[:, XD.std(axis=0) > 0] if XD.shape[1] else XD

    tasks = []
    for r in range(n_repeats):
        rng = np.random.default_rng(random_state + r)
        fold = np.empty(n, int)
        for u in range(len(labels)):  # deal each unit's rows over the folds
            rows = rng.permutation(np.flatnonzero(code == u))
            fold[rows] = (np.arange(len(rows)) + rng.integers(n_splits)) % n_splits
        for k in range(n_splits):
            te = np.flatnonzero(fold == k)
            if len(te) and len(te) < n:
                tasks.append((np.flatnonzero(fold != k), te, random_state + r))
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fold)(XA, XD, code, y, w, tr, te, fam, seed) for tr, te, seed in tasks
    )
    keys = list(results[0]["loss"])
    L = {k: np.zeros(n) for k in keys}
    base, count = np.zeros(n), np.zeros(n)
    for r in results:
        count[r["rows"]] += 1
        base[r["rows"]] += r["base"]
        for k in keys:
            L[k][r["rows"]] += r["loss"][k]
    seen = count > 0
    b = base[seen] / count[seen]
    _, unit_idx = np.unique(code[seen], return_inverse=True)
    skill_rows, add_rows = [], []
    for (name, kind), loss in L.items():
        p, lo, hi, _ = _intervals(unit_idx, (loss[seen] / count[seen])[:, None], b, level)
        skill_rows.append({"model": name, "kind": kind, "skill": 1 - p[0], "lo": 1 - hi[0],
                           "hi": min(1.0, 1 - lo[0])})  # fmt: skip
    for kind in ("additive", "any"):
        for what, name in (("identity", IDENTITY), ("description", DESCRIPTION)):
            if (name, kind) not in L:
                continue
            gain = (L[(WITHOUT, kind)] - L[(name, kind)])[seen] / count[seen]
            p, lo, hi, _ = _intervals(unit_idx, gain[:, None], b, level)
            add_rows.append({"adds": what, "kind": kind, "contribution": p[0], "lo": lo[0],
                             "hi": hi[0]})  # fmt: skip
    unit = shift.name.removeprefix("new ")
    return Ceiling(pd.DataFrame(skill_rows), pd.DataFrame(add_rows), unit, describe, tol)
