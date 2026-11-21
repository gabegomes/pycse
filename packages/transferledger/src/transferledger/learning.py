"""What it would take: skill under a shift against the number of units trained on.

A verdict of "does not transfer" says what cannot be claimed. The next question is what would
change it. With twelve ligands nothing about a ligand carries to a new one: would twenty-four
do, or is the description of a ligand the limit? :func:`learning_curve` answers the measurable
part. It trains on 4, 6, 9, ... of the units in hand (whole molecules, sites, patients), scores
each model on units it has not seen, and reports the skill at each size with an interval.

The curve can then be extended a stated distance beyond the units in hand
(:meth:`LearningCurve.forecast`, :meth:`LearningCurve.needed`). An extrapolated learning curve
is a forecast, and forecasts of this kind are easy to make and often wrong, so the extension
is limited to a horizon, and the package's tests check, on populations where the answer is
known, that the stated intervals hold.
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

__all__ = ["learning_curve", "LearningCurve", "power_law"]

#: Exponents tried for the approach to the ceiling. 1 is what a fitted model with a fixed number
#: of parameters gives (its excess loss falls as parameters / units); slower and faster
#: approaches are allowed for.
EXPONENTS = (0.5, 0.75, 1.0, 1.5, 2.0)


def power_law(units, skill, weights=None, exponents=EXPONENTS):
    """Fit ``skill = ceiling - gap * units**(-rate)`` to a measured curve.

    The gap is not allowed to be negative (a curve that falls with more units is read as flat)
    and the ceiling not above 1.

    Returns:
        ``(ceiling, gap, rate)``.
    """
    m, s = np.asarray(units, float), np.asarray(skill, float)
    w = np.ones_like(s) if weights is None else np.asarray(weights, float)
    best = None
    for c in exponents:
        x = m ** (-c)
        xm, sm = np.average(x, weights=w), np.average(s, weights=w)
        vx = np.average((x - xm) ** 2, weights=w)
        b = -np.average((x - xm) * (s - sm), weights=w) / vx if vx > 0 else 0.0
        b = max(b, 0.0)
        a = sm + b * xm
        if a > 1.0:  # the ceiling cannot pass 1: refit the gap with it held there
            a = 1.0
            b = max(np.average(x * (1.0 - s), weights=w) / np.average(x * x, weights=w), 0.0)
        sse = float(np.average((s - (a - b * x)) ** 2, weights=w))
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
            test units), ``spread`` (standard deviation of the skill across draws).
        fits: per rung, the fitted ``ceiling``, ``gap`` and ``rate`` of
            ``skill = ceiling - gap * units**(-rate)``, with an interval for the ceiling.
        unit: what a unit is called ("ligand").
        n_units: how many units the data have.
        horizon: how far beyond ``n_units`` a forecast is given, as a multiple of it.
    """

    curve: pd.DataFrame
    fits: pd.DataFrame
    unit: str
    n_units: int
    horizon: float
    level: float
    family: str
    meta: dict = field(default_factory=dict)
    _loo: dict = field(
        default_factory=dict, repr=False
    )  # rung -> (sizes, leave-one-unit-out skills)

    @property
    def rungs(self):
        return list(dict.fromkeys(self.curve["rung"]))

    @property
    def best(self):
        """The rung with the highest skill at the largest size measured."""
        top = self.curve[self.curve.units == self.curve.units.max()]
        return str(top.sort_values("skill", ascending=False)["rung"].iloc[0])

    def _replicates(self, rung, units):
        """The fitted curve at ``units`` from all units, and with each unit left out in turn."""
        sizes, point, loo = self._loo[rung]
        units = np.atleast_1d(np.asarray(units, float))

        def at(s):
            a, b, c = power_law(sizes, s)
            return a - b * units ** (-c)

        return at(point), np.array([at(row) for row in loo])

    def state(self, rung=None):
        """Where the curve stands: ``"nothing yet"``, ``"rising"`` or ``"level"``.

        *Nothing yet*: at the largest size measured the interval of the skill reaches zero, so
        no skill on a new unit is established. *Rising*: skill is established and is higher
        at the largest size than at about half of it, by more than the uncertainty of that
        difference on the same test units. *Level*: established, and not rising.
        """
        rung = rung or self.best
        c = self.curve[self.curve.rung == rung].sort_values("units")
        if c["lo"].iloc[-1] <= 0:
            return "nothing yet"
        sizes, point, loo = self._loo[rung]
        half = int(np.argmin(np.abs(sizes[:-1] - sizes[-1] / 2.0)))
        d = loo[:, -1] - loo[:, half]
        g = len(d)
        se = np.sqrt((g - 1) / g * np.sum((d - d.mean()) ** 2))
        rise = point[-1] - point[half]
        return "rising" if rise - student_t.ppf(0.5 + self.level / 2, g - 1) * se > 0 else "level"

    def forecast(self, units, rung=None):
        """The fitted curve at ``units``, with an interval, for sizes up to the horizon.

        Nothing is forecast while the state is "nothing yet": a curve that sits at zero does
        not say where it would leave zero (on 90 phosphines, forecasts made from 13 of them
        missed the skill later measured with 39 by 0.4). Sizes beyond ``horizon * n_units``
        are missing too.

        Returns:
            DataFrame with ``units``, ``skill``, ``lo``, ``hi``.
        """
        rung = rung or self.best
        units = np.atleast_1d(np.asarray(units, float))
        if self.state(rung) == "nothing yet":
            nan = np.full(len(units), np.nan)
            return pd.DataFrame({"units": units, "skill": nan, "lo": nan, "hi": nan})
        point, reps = self._replicates(rung, units)
        g = len(reps)
        se = np.sqrt((g - 1) / g * np.sum((reps - reps.mean(axis=0)) ** 2, axis=0))
        half = student_t.ppf(0.5 + self.level / 2, g - 1) * se
        out = pd.DataFrame({"units": units, "skill": point, "lo": point - half,
                            "hi": np.minimum(point + half, 1.0)})  # fmt: skip
        out.loc[out.units > self.horizon * self.n_units, ["skill", "lo", "hi"]] = np.nan
        return out

    def needed(self, skill, rung=None):
        """How many units the fitted curve needs to reach ``skill``.

        Returns:
            dict with ``units`` (from the fitted curve), ``optimistic`` and ``pessimistic`` (from
            the two ends of its interval); a value is None when that curve does not reach the
            target within the horizon.
        """
        rung = rung or self.best
        last = int(np.floor(self.horizon * self.n_units))
        grid = np.arange(self.curve.units.min(), last + 1)
        f = self.forecast(grid, rung)

        def first(col):
            hit = np.flatnonzero(f[col].to_numpy() >= skill)
            return int(grid[hit[0]]) if len(hit) else None

        return {"rung": rung, "target": skill, "units": first("skill"),
                "optimistic": first("hi"), "pessimistic": first("lo"), "searched to": last}  # fmt: skip

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
            last = int(np.floor(self.horizon * self.n_units))
            state = self.state()
            if state == "nothing yet":
                text += (
                    f"\n\n{self.best!r}: nothing yet. No skill on a new {self.unit} is established "
                    f"with {self.n_units} {self.unit}s, and a curve at zero does not say where it "
                    f"would leave zero, so nothing is forecast. detectable() says what these "
                    f"{self.unit}s could have shown."
                )
            else:
                steps = sorted(
                    {int(round(k * self.n_units)) for k in (1.0, 1.5, 2.0, self.horizon)}
                )
                f = self.forecast([q for q in steps if q <= last], self.best)
                note = ("skill is established and still rising" if state == "rising" else
                        "skill is established and no longer rising with more units")  # fmt: skip
                text += (
                    f"\n\n{self.best!r}: {state} ({note}). Forecast from the fitted curve, not "
                    f"given beyond {last} {self.unit}s:\n" + f.round(3).to_string(index=False)
                )
        return text

    def to_dict(self):
        return {
            "unit": self.unit, "n_units": self.n_units, "horizon": self.horizon, "level": self.level,
            "family": self.family, "meta": self.meta,
            "curve": self.curve.to_dict("records"), "fits": self.fits.to_dict("records"),
        }  # fmt: skip


def learning_curve(X, y, shift=None, *, model=None, models=None, family="gaussian",
                   sample_weight=None, sizes=None, n_sizes=6, smallest=4, test_fraction=0.2,
                   n_repeats=None, horizon=2.0, level=0.95, random_state=0, n_jobs=None):  # fmt: skip
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
        horizon: forecasts are given up to this multiple of the units in hand.
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
    rungs = [EQUATION, CURVES, *extra]
    _, unit_idx = np.unique(units, return_inverse=True)
    rows, loo, fit_rows = [], {}, []
    tq = student_t.ppf(0.5 + level / 2, G - 1)
    for rung in rungs:
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
        point, reps = np.array(point), np.array(reps).T  # reps: (units, sizes)
        loo[names.get(rung, rung)] = (np.array(sizes, float), point, reps)
        a, b, c = power_law(sizes, point)
        ceilings = np.array([power_law(sizes, r)[0] for r in reps])
        se = np.sqrt((G - 1) / G * np.sum((ceilings - ceilings.mean()) ** 2))
        fit_rows.append({"rung": names.get(rung, rung), "ceiling": a, "lo": a - tq * se,
                         "hi": min(1.0, a + tq * se), "gap": b, "rate": c})  # fmt: skip
    return LearningCurve(
        curve=pd.DataFrame(rows), fits=pd.DataFrame(fit_rows), unit=unit_name, n_units=G,
        horizon=float(horizon), level=level, family=fam.name,
        meta={"rows": n, "test units per draw": int(np.ceil(G / K)), "draws": len(tasks) // len(sizes),
              "sizes": sizes},
        _loo=loo,
    )  # fmt: skip
