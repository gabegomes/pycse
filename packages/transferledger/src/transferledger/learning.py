"""What it would take: skill under a shift against the number of units trained on.

A verdict of "does not transfer" says what cannot be claimed. The next question is what would
change it. With twelve ligands nothing about a ligand carries to a new one: would twenty-four
do, or is the description of a ligand the limit? :func:`learning_curve` answers the measurable
part. It trains on 4, 6, 9, ... of the units in hand (whole molecules, sites, patients), scores
each model on units it has not seen, and reports the skill at each size with an interval.

What the curve is allowed to say was settled by experiment, not by taste. :func:`backtest`
measures a curve on a part of the units and then reveals the rest. On a set of 90 phosphines,
on alcohols, on housing regions and on made-up populations with known answers, it showed that:

- a curve that sits at zero does not say where it will leave zero. Curves fitted to 13
  phosphines put the skill with 39 at -0.17; it was 0.33. So in the state "nothing yet"
  nothing is said about more units, and :func:`~transferledger.detection.detectable` takes
  over;
- no way of extending a curve was right in both directions. Holding the last value is too low
  where skill keeps growing, and extending the rise overshoots where a threshold has just
  been found. So no number is forecast for more units;
- one statement held: once skill on a new unit is established, adding units did not lose it
  (the skill later measured stayed at or above the lower end of the interval in 98 of 102
  checks). :meth:`LearningCurve.floor` reports that lower bound.

The numbers are those of the example notebook ``KAN_Transfer_Planning``, which repeats the
check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import t as student_t
from sklearn.base import clone

from ._data import Design
from .curves import get_family
from .ledger import BASELINE, CURVES, EQUATION, _loss, _predict
from .model import PieceModel
from .shifts import NewGroups, Reference, Shift

__all__ = ["learning_curve", "LearningCurve", "backtest", "power_law"]

#: Exponents tried when a curve is extended. 1 is what a fitted model with a fixed number of
#: parameters gives (its excess loss falls as parameters / units).
EXPONENTS = (0.5, 0.75, 1.0, 1.5, 2.0)


def power_law(units, skill, exponents=EXPONENTS):
    """Fit ``skill = ceiling - gap * units**(-rate)`` to a measured curve.

    This is the textbook way to extend a learning curve, kept so that its failure can be
    reproduced: :func:`backtest` reports what it would have said next to what happened. The gap
    is not allowed to be negative and the ceiling not above 1.

    Returns:
        ``(ceiling, gap, rate)``.
    """
    m, s = np.asarray(units, float), np.asarray(skill, float)
    best = None
    for c in exponents:
        x = m ** (-c)
        vx = np.mean((x - x.mean()) ** 2)
        b = max(-np.mean((x - x.mean()) * (s - s.mean())) / vx if vx > 0 else 0.0, 0.0)
        a = s.mean() + b * x.mean()
        if a > 1.0:  # the ceiling cannot pass 1: refit the gap with it held there
            a, b = 1.0, max(np.mean(x * (1.0 - s)) / np.mean(x * x), 0.0)
        sse = float(np.mean((s - (a - b * x)) ** 2))
        if best is None or sse < best[0] - 1e-15:
            best = (sse, a, b, c)
    return best[1], best[2], best[3]


def _sizes(largest, smallest, n):
    smallest = min(smallest, largest)
    return [int(v) for v in np.unique(np.round(np.geomspace(smallest, largest, n)).astype(int))]


def _one(X, y, w, units, train_units, test_units, proto, extra, fam, inner):
    """Train on the rows of ``train_units``, score every rung on the rows of ``test_units``."""
    tr = np.flatnonzero(np.isin(units, train_units))
    te = np.flatnonzero(np.isin(units, test_units))
    Xtr, Xte, ytr, yte, wtr, wte = X.iloc[tr], X.iloc[te], y[tr], y[te], w[tr], w[te]
    out = {"rows": te}
    mu0 = np.full(len(te), np.average(ytr, weights=wtr))
    out[BASELINE] = _loss(fam, yte, mu0, wte)
    few = inner is not None and len(np.unique(units[tr])) < 2
    if np.ptp(ytr) == 0 or few:  # nothing to learn from: every rung is the baseline
        out.update({k: out[BASELINE] for k in (EQUATION, CURVES, *extra)})
        return out
    shift = None if inner is None else NewGroups(units[tr], inner)
    m = clone(proto).set_params(select_on=shift).fit(Xtr, ytr, sample_weight=wtr)
    out[EQUATION] = _loss(fam, yte, m.predict(Xte), wte)
    Xn_tr, Xn_te = m.design_.transform(Xtr), m.design_.transform(Xte)
    out[CURVES] = _loss(fam, yte, np.asarray(m.source_.predict(Xn_te), float), wte)
    for name, est in extra.items():
        e = clone(est).fit(Xn_tr, ytr if fam.name != "binomial" else ytr.astype(int))
        out[name] = _loss(fam, yte, _predict(e, Xn_te, fam), wte)
    return out


@dataclass
class LearningCurve:
    """Skill on unseen units against the number of units trained on.

    Attributes:
        curve: one row per rung and size: ``units``, ``skill``, ``lo``, ``hi`` (interval over the
            test units), ``spread`` (standard deviation of the skill across draws), ``draws``.
        unit: what a unit is called ("ligand").
        n_units: how many units the data have.
        level: confidence level of the intervals.
    """

    curve: pd.DataFrame
    unit: str
    n_units: int
    level: float
    family: str
    meta: dict = field(default_factory=dict)
    # rung -> (sizes, skill, skill with each unit left out)
    _loo: dict = field(default_factory=dict, repr=False)

    @property
    def rungs(self):
        return list(dict.fromkeys(self.curve["rung"]))

    @property
    def best(self):
        """The rung with the highest skill at the largest size measured."""
        top = self.curve[self.curve.units == self.curve.units.max()]
        return str(top.sort_values("skill", ascending=False)["rung"].iloc[0])

    def top(self, rung=None):
        """The row of the curve at the largest size measured."""
        rung = rung or self.best
        return self.curve[self.curve.rung == rung].sort_values("units").iloc[-1]

    def state(self, rung=None):
        """Where the curve stands: ``"nothing yet"`` or ``"established"``.

        *Nothing yet*: at the largest size measured the interval of the skill reaches zero, so
        no skill on a new unit is established. *Established*: it lies above zero.

        A finer reading was tried and dropped. Whether an established curve was still rising
        or had levelled off said nothing about what more units then added (0.10 of skill on
        average in both cases, over the 168 checks of an earlier run).
        """
        return "nothing yet" if self.top(rung)["lo"] <= 0 else "established"

    def floor(self, rung=None):
        """The skill that more units are expected to keep: the lower end of the interval at
        the largest size measured, or None while nothing is established.

        This is the one statement about more units that held when checked (see
        :func:`backtest`): with two or three times the units, the skill measured stayed at or
        above this value in 98 of 102 checks (the example notebook ``KAN_Transfer_Planning``)
        and in 95 % of the 168 checks of an earlier run on six populations.
        """
        row = self.top(rung)
        return float(row["lo"]) if row["lo"] > 0 else None

    def extend(self, units, rung=None):
        """What extending the fitted curve gives at ``units``. Not a forecast.

        The value of :func:`power_law` fitted to the measured sizes. Checked against units
        added later it was too low by 0.26 of skill on average where nothing was established
        yet, and off by 0.15 either way where something was; it is here so that the check can
        be repeated.
        """
        sizes, point, _ = self._loo[rung or self.best]
        a, b, c = power_law(sizes, point)
        return a - b * np.atleast_1d(np.asarray(units, float)) ** (-c)

    def advice(self, rung=None):
        """One sentence on what the curve supports."""
        rung = rung or self.best
        state, u, n = self.state(rung), self.unit, self.n_units
        if state == "nothing yet":
            return (
                f"Nothing yet: no skill on a new {u} is established with {n} {u}s. A curve at "
                f"zero does not say where it would leave zero; detectable() says what these "
                f"{u}s could have shown and how many a weaker property needs."
            )
        t = self.top(rung)
        return (
            f"Established: skill on a new {u} is {t.skill:.2f} with {int(t.units)} {u}s, and at "
            f"least {self.floor(rung):.2f} is expected to be kept with more. Whether more {u}s "
            f"add to it the curve does not say."
        )

    def __str__(self):
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            c = self.curve.assign(
                t=self.curve.apply(lambda r: f"{r.skill:.2f} [{r.lo:.2f}, {r.hi:.2f}]", axis=1)
            ).pivot(index="rung", columns="units", values="t")
            text = (
                f"Skill on a new {self.unit} against the number of {self.unit}s trained on "
                f"({self.n_units} in hand; {int(100 * self.level)} % intervals over test {self.unit}s)\n"
                + c.loc[self.rungs].to_string()
            )
        return text + f"\n\n{self.best!r}. {self.advice()}"

    def to_dict(self):
        return {
            "unit": self.unit, "n_units": self.n_units, "level": self.level, "family": self.family,
            "meta": self.meta, "curve": self.curve.to_dict("records"), "best": self.best,
            "state": {r: self.state(r) for r in self.rungs},
            "floor": {r: self.floor(r) for r in self.rungs}, "advice": self.advice(),
        }  # fmt: skip


def learning_curve(X, y, shift=None, *, model=None, models=None, family="gaussian",
                   sample_weight=None, sizes=None, n_sizes=6, smallest=4, test_fraction=0.2,
                   n_repeats=None, level=0.95, random_state=0, n_jobs=None):  # fmt: skip
    """Measure the skill on unseen units as a function of the number of units trained on.

    The units of ``shift`` are dealt into test folds; for each fold the remaining units are put
    in a random order and the model is trained on the first 4, 6, 9, ... of them, so that the
    sizes are compared on the same test units. The equation is chosen, and the curves tuned,
    for the shift inside each training set, as in the ledger.

    Args:
        X, y: the table and the target.
        shift: a :class:`~transferledger.shifts.NewGroups` (units are its groups) or None
            (units are rows: the ordinary learning curve).
        model: a :class:`~transferledger.model.PieceModel` to use as the template.
        models: ``{name: estimator}`` of other models to score at each size.
        sizes: numbers of training units to measure; default ``n_sizes`` sizes from
            ``smallest`` up to all units but one test fold, evenly spaced on a log scale.
        test_fraction: share of the units held out in each draw.
        n_repeats: how many times the units are dealt into test folds (default: enough for
            about twenty draws).
        level: confidence level of the intervals.

    Returns:
        A :class:`LearningCurve`.
    """
    fam = get_family(family)
    frame = Design()._frame(X)
    n = len(frame)
    y = np.asarray(y, float).ravel()
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
    if shift is None or isinstance(shift, Reference):
        units, inner, unit_name = np.arange(n), None, "row"
    elif isinstance(shift, NewGroups):
        units, inner = np.asarray(shift.groups), shift.name
        unit_name = shift.name[4:] if shift.name.startswith("new ") else "group"
    elif isinstance(shift, Shift):
        raise TypeError(
            "a learning curve needs units that can be counted: NewGroups, or None for rows"
        )
    else:
        raise TypeError("shift must be a NewGroups shift or None")
    if len(units) != n:
        raise ValueError(f"the shift covers {len(units)} rows but X has {n}")
    labels = np.unique(units)
    G = len(labels)
    n_test = max(1, int(round(test_fraction * G)))
    K = max(2, int(round(G / n_test)))
    largest = G - int(np.ceil(G / K))
    if largest < 3:
        raise ValueError(f"{G} units are too few for a learning curve")
    if sizes is None:
        sizes = _sizes(largest, smallest, n_sizes)
    sizes = sorted({int(s) for s in sizes if 2 <= int(s) <= largest})
    if len(sizes) < 2:
        raise ValueError("a learning curve needs at least two sizes")
    if n_repeats is None:
        n_repeats = max(1, int(np.ceil(20 / K)))
    proto = clone(model) if model is not None else PieceModel(family=family)
    if proto.family != fam.name:
        raise ValueError(
            f"the model's family is {proto.family!r} but family={fam.name!r} was asked"
        )
    extra = dict(models or {})

    rng = np.random.default_rng(random_state)
    tasks = []
    for _ in range(n_repeats):
        perm = rng.permutation(G)
        for fold in np.array_split(perm, K):
            rest = rng.permutation(np.setdiff1d(perm, fold))
            for s in sizes:
                tasks.append((s, labels[rest[:s]], labels[fold]))
    results = Parallel(n_jobs=n_jobs)(
        delayed(_one)(frame, y, w, units, tr, te, proto, extra, fam, inner) for _, tr, te in tasks
    )

    names = {EQUATION: EQUATION, CURVES: CURVES}
    if inner is not None:
        names = {
            EQUATION: f"{EQUATION}, chosen for {inner}",
            CURVES: f"{CURVES}, tuned for {inner}",
        }
    _, unit_idx = np.unique(units, return_inverse=True)
    rows, loo = [], {}
    tq = student_t.ppf(0.5 + level / 2, G - 1)
    for rung in [EQUATION, CURVES, *extra]:
        point, reps = [], []
        for s in sizes:
            mine = [r for (size, _, _), r in zip(tasks, results) if size == s]
            Q, B = np.zeros(G), np.zeros(G)
            per_draw = []
            for r in mine:
                g = unit_idx[r["rows"]]
                np.add.at(Q, g, r[rung])
                np.add.at(B, g, r[BASELINE])
                per_draw.append(1.0 - r[rung].sum() / r[BASELINE].sum())
            skill = 1.0 - Q.sum() / B.sum()
            left = 1.0 - (Q.sum() - Q) / (B.sum() - B)  # the skill with each unit left out
            se = np.sqrt((G - 1) / G * np.sum((left - left.mean()) ** 2))
            rows.append({"rung": names.get(rung, rung), "units": s, "skill": skill,
                         "lo": skill - tq * se, "hi": min(1.0, skill + tq * se),
                         "spread": float(np.std(per_draw)), "draws": len(mine)})  # fmt: skip
            point.append(skill)
            reps.append(left)
        loo[names.get(rung, rung)] = (np.array(sizes, float), np.array(point), np.array(reps).T)
    return LearningCurve(
        curve=pd.DataFrame(rows), unit=unit_name, n_units=G, level=level, family=fam.name,
        meta={"rows": n, "test units per draw": int(np.ceil(G / K)),
              "draws": len(tasks) // len(sizes), "sizes": sizes},
        _loo=loo,
    )  # fmt: skip


def backtest(X, y, shift, *, have, factors=(1.5, 2.0, 3.0), n_trials=6, n_acquisitions=6,
             model=None, family="gaussian", sample_weight=None, min_test=5, max_test=40,
             level=0.95, random_state=0, n_jobs=None, **curve_options):  # fmt: skip
    """Check what a learning curve says about more units, on a table with units to spare.

    In each trial ``have`` units are drawn and treated as all there is, and a learning curve is
    measured on them. Then ``factor * have`` units are really used (the ones in hand plus units
    drawn from the rest), and the skill is measured on units left over, which the curve has
    not seen.

    Args:
        have: the number of units pretended to be in hand.
        factors: multiples of ``have`` at which the skill is then measured.
        n_trials: independent draws of the units in hand.
        n_acquisitions: draws of the added units per trial and factor; their test rows are
            pooled.
        min_test, max_test: a factor is skipped when fewer than ``min_test`` units would be
            left to test on; at most ``max_test`` are used.
        **curve_options: passed to :func:`learning_curve` (``n_sizes``, ``n_repeats``, ...).

    Returns:
        DataFrame, one row per trial, factor and rung: ``units``; from the curve on the units
        in hand its ``state``, the skill at its largest size (``held``, ``lo``, ``hi``) and
        what the fitted curve extends to (``extended``); and the skill then ``realized``.
    """
    if not isinstance(shift, NewGroups):
        raise TypeError("backtest needs a NewGroups shift: its groups are the units")
    fam = get_family(family)
    frame = Design()._frame(X)
    y = np.asarray(y, float).ravel()
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
    units = np.asarray(shift.groups)
    labels = np.unique(units)
    proto = clone(model) if model is not None else PieceModel(family=family)
    rng = np.random.default_rng(random_state)
    rows = []
    for trial in range(n_trials):
        mine = rng.choice(labels, have, replace=False)
        sub = np.flatnonzero(np.isin(units, mine))
        L = learning_curve(
            frame.iloc[sub].reset_index(drop=True), y[sub], NewGroups(units[sub], shift.name),
            model=proto, family=family, sample_weight=w[sub], level=level,
            random_state=random_state + trial, n_jobs=n_jobs, **curve_options,
        )  # fmt: skip
        rest = np.setdiff1d(labels, mine)
        for factor in factors:
            m = int(round(factor * have))
            if m < have or len(rest) - (m - have) < min_test:
                continue
            jobs = []
            for _ in range(n_acquisitions):
                new = rng.choice(rest, m - have, replace=False)
                test = np.setdiff1d(rest, new)
                if len(test) > max_test:
                    test = rng.choice(test, max_test, replace=False)
                jobs.append((np.r_[mine, new], test))
            res = Parallel(n_jobs=n_jobs)(
                delayed(_one)(frame, y, w, units, tr, te, proto, {}, fam, shift.name)
                for tr, te in jobs
            )
            for key, rung in zip((EQUATION, CURVES), L.rungs):
                real = 1.0 - sum(r[key].sum() for r in res) / sum(r[BASELINE].sum() for r in res)
                t = L.top(rung)
                rows.append({"trial": trial, "have": have, "factor": factor, "units": m,
                             "rung": rung, "state": L.state(rung), "held": t["skill"],
                             "lo": t["lo"], "hi": t["hi"],
                             "extended": float(L.extend([m], rung)[0]), "realized": real})  # fmt: skip
    return pd.DataFrame(rows)
