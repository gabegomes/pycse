"""What the units in hand could have shown: the detection floor of a table.

When nothing about a unit carries to a new unit, two readings are open. The description of the
units may hold nothing; or it may hold something that this many units cannot establish. The
learning curve cannot tell them apart, because a curve that sits at zero says nothing about
where it would leave zero.

:func:`detectable` separates them by experiment. It adds to the table a variable whose relation
to the units is known: a made-up property that captures a chosen share of what distinguishes
one unit from another (the unit's own mean of the target, blurred). The whole analysis is then
run as it would be on the real table, and one counts how often the planted property is admitted
to the equation chosen for the shift. Doing this at several shares and several numbers of units
gives

- the *floor*: the smallest share a property must capture to be established with the units in
  hand, so that "nothing transfers" can be read as "nothing stronger than this exists here";
- the units needed for a weaker property. A property that captures a share ``q`` of the
  unit-to-unit variance is a correlation of ``sqrt(q)`` over the units, and the evidence for a
  correlation over ``m`` units grows as ``sqrt((m - 2) q / (1 - q))``. The detection rate is
  fitted as a probit curve in that one quantity, which ties what more units would do to what a
  larger share is seen to do.

The target is never simulated: its noise, its rows and its other variables are the real ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import clone

from ._data import Design
from .model import PieceModel
from .shifts import NewGroups

__all__ = ["detectable", "Detection"]

PLANTED = "planted property"


def _plant(means, share, shape, rng):
    """A unit property that captures ``share`` of the variance of the unit ``means``.

    ``"trend"``: the target rises with the property. ``"peak"``: the target is high in the
    middle of the property's range and low on both sides, the harder case, since both sides
    have to be populated before a peak can be told from a trend.
    """
    t = (means - means.mean()) / (means.std() or 1.0)
    v = np.sqrt(share) * t + np.sqrt(1.0 - share) * rng.normal(size=len(t))
    if shape == "trend":
        return v
    if shape == "peak":
        return rng.choice([-1.0, 1.0], size=len(t)) * (v.max() - v)
    raise ValueError(f"shape must be 'trend' or 'peak', not {shape!r}")


def _trial(X, y, w, units, keep, share, shape, proto, name, seed):
    """Plant a property on the units ``keep`` and report whether the analysis admits it."""
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(np.isin(units, keep))
    u = units[rows]
    labels, idx = np.unique(u, return_inverse=True)
    means = np.bincount(idx, weights=w[rows] * y[rows]) / np.bincount(idx, weights=w[rows])
    Xs = X.iloc[rows].reset_index(drop=True).copy()
    Xs[PLANTED] = _plant(means, share, shape, rng)[idx]
    m = (
        clone(proto)
        .set_params(select_on=NewGroups(u, name))
        .fit(Xs, y[rows], sample_weight=w[rows])
    )
    return PLANTED in m.variables_


@dataclass
class Detection:
    """How often a planted property was established, by shape, number of units and share.

    Attributes:
        power: one row per shape, number of units and share: ``detected`` (share of trials in
            which the planted property entered the equation) and ``trials``.
        false_alarms: per shape and number of units, how often a property that captures
            nothing was admitted.
        law: per shape, the fitted ``intercept`` and ``slope`` of
            ``probit(detected) = intercept + slope * sqrt((units - 2) * share / (1 - share))``.
        unit, n_units, level: what a unit is called, how many the table has, the detection
            rate that counts as "established".
    """

    power: pd.DataFrame
    false_alarms: pd.DataFrame
    law: pd.DataFrame
    unit: str
    n_units: int
    level: float
    horizon: float
    meta: dict = field(default_factory=dict)

    def _law(self, shape):
        return self.law.set_index("shape").loc[shape]

    def rate(self, units, share, shape="trend"):
        """Detection rate the fitted law gives for ``units`` units and a property of ``share``."""
        k = self._law(shape)
        return norm.cdf(k["intercept"] + k["slope"] * _evidence(units, share))

    def floor(self, shape="trend", units=None):
        """The smallest share a property must capture to be established.

        For the units in hand this is read off the measured rates (interpolated between the
        shares that were planted); it equals the smallest share planted when even that one was
        established. For another number of ``units`` the fitted law is used. None when no
        share is enough.
        """
        if units is None or units == self.n_units:
            c = self.power[
                (self.power["shape"] == shape) & (self.power.units == self.power.units.max())
            ]
            c = c.sort_values("share")
            q = c["share"].to_numpy(float)
            r = np.maximum.accumulate(c["detected"].to_numpy(float))  # more is not found less often
            hit = np.flatnonzero(r >= self.level)
            if not len(hit):
                return None
            k = int(hit[0])
            if k == 0:
                return float(q[0])
            return float(q[k - 1] + (self.level - r[k - 1]) * (q[k] - q[k - 1]) / (r[k] - r[k - 1]))
        grid = np.linspace(0.02, 0.98, 97)
        ok = np.flatnonzero(self.rate(units, grid, shape) >= self.level)
        return float(grid[ok[0]]) if len(ok) else None

    def needed(self, share, shape="trend"):
        """Units needed to establish a property that captures ``share``, from the fitted law.

        None beyond the horizon (a multiple of the units in hand), and None when the planted
        rates did not rise with the evidence, so that there is no law to extend.
        """
        if not self._law(shape)["slope"] > 0:
            return None
        last = int(np.floor(self.horizon * self.n_units))
        grid = np.arange(4, last + 1)
        ok = np.flatnonzero(self.rate(grid, share, shape) >= self.level)
        return int(grid[ok[0]]) if len(ok) else None

    def table(self, shares=(0.3, 0.5, 0.7), shapes=None):
        """Units needed, per shape and share; "more than" the horizon where it is passed."""
        shapes = shapes or list(self.law["shape"])
        last = int(np.floor(self.horizon * self.n_units))
        rows = {}
        for s in shapes:
            need = {q: self.needed(q, s) for q in shares}
            rows[s] = {
                f"{int(100 * q)} %": f"more than {last}" if v is None else v
                for q, v in need.items()
            }
        out = pd.DataFrame(rows, dtype=object).T
        out.index.name = "shape"
        return out

    def __str__(self):
        p = self.power.assign(cell=self.power.detected.map("{:.2f}".format)).pivot(
            index=["shape", "units"], columns="share", values="cell"
        )
        lines = [
            f"How often a planted property of each share was established ({self.n_units} {self.unit}s "
            f"in hand; 'established' means in {int(100 * self.level)} % of trials)",
            p.to_string(),
            "",
            "A property that captures nothing was admitted in "
            + ", ".join(
                f"{r.rate:.0%} of trials ({r.shape}, {r.units} {self.unit}s)"
                for r in self.false_alarms.itertuples()
            ),  # fmt: skip
            "",
        ]
        smallest = float(self.power["share"].min())
        for s in self.law["shape"]:
            f = self.floor(s)
            if f is None:
                say = "more than any share planted"
            elif f <= smallest:
                say = f"{f:.0%} or less (the smallest share planted was established)"
            else:
                say = f"at least {f:.0%}"
            lines.append(
                f"{s}: with {self.n_units} {self.unit}s a property must capture {say} of the "
                f"differences between {self.unit}s to be established"
            )
        lines += ["", f"{self.unit.capitalize()}s needed for a weaker property:",
                  self.table().to_string()]  # fmt: skip
        return "\n".join(lines)

    def to_dict(self):
        return {
            "unit": self.unit, "n_units": self.n_units, "level": self.level, "horizon": self.horizon,
            "power": self.power.to_dict("records"), "false_alarms": self.false_alarms.to_dict("records"),
            "law": self.law.to_dict("records"), "meta": self.meta,
            "floor": {s: self.floor(s) for s in self.law["shape"]},
        }  # fmt: skip


def _evidence(units, share):
    """The strength of the evidence for a correlation of ``sqrt(share)`` over ``units`` units
    (its t statistic): ``sqrt((units - 2) * share / (1 - share))``."""
    m = np.maximum(np.asarray(units, float) - 2.0, 1.0)
    q = np.clip(np.asarray(share, float), 1e-6, 1 - 1e-6)
    return np.sqrt(m * q / (1.0 - q))


def _fit_law(cells):
    """``probit(rate) = intercept + slope * evidence``, by maximum likelihood on the counts."""
    x = _evidence(cells["units"].to_numpy(float), cells["share"].to_numpy(float))
    k = (cells["detected"] * cells["trials"]).to_numpy(float)
    n = cells["trials"].to_numpy(float)

    def nll(b):
        p = np.clip(norm.cdf(b[0] + b[1] * x), 1e-9, 1 - 1e-9)
        return -(k * np.log(p) + (n - k) * np.log(1 - p)).sum() + 1e-6 * float(b @ b)

    opts = {"xatol": 1e-7, "fatol": 1e-10, "maxiter": 4000}
    fits = [minimize(nll, np.array(x0), method="Nelder-Mead", options=opts)
            for x0 in ((0.0, 0.0), (-2.0, 0.5), (-4.0, 1.0))]  # fmt: skip
    return min(fits, key=lambda r: r.fun).x


def detectable(X, y, shift, *, model=None, family="gaussian", sample_weight=None,
               shares=(0.2, 0.35, 0.5, 0.7, 0.9), shapes=("trend", "peak"), sizes=None,
               n_trials=24, level=0.8, horizon=3.0, random_state=0, n_jobs=None):  # fmt: skip
    """Plant properties of known strength and count how often the analysis establishes them.

    Args:
        X, y: the table and the target, as analysed.
        shift: a :class:`~transferledger.shifts.NewGroups`; its groups are the units.
        model: a :class:`~transferledger.model.PieceModel` template.
        shares: the shares of the unit-to-unit variance of the target the planted property
            captures (its squared correlation with the unit means, before blurring by chance).
        shapes: ``"trend"`` and/or ``"peak"``.
        sizes: numbers of units to try (random subsets of the units); default: about a half,
            about three quarters, and all of them.
        n_trials: plantings per shape, size and share.
        level: the detection rate that counts as established.
        horizon: units needed are given up to this multiple of the units in hand.

    Returns:
        A :class:`Detection`.
    """
    if not isinstance(shift, NewGroups):
        raise TypeError("detectable needs a NewGroups shift: its groups are the units")
    frame = Design()._frame(X)
    n = len(frame)
    if PLANTED in frame.columns:
        raise ValueError(f"the table already has a column called {PLANTED!r}")
    y = np.asarray(y, float).ravel()
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
    units = np.asarray(shift.groups)
    if len(units) != n:
        raise ValueError(f"the shift covers {len(units)} rows but X has {n}")
    labels = np.unique(units)
    G = len(labels)
    if G < 6:
        raise ValueError(f"{G} units are too few to plant anything on")
    if sizes is None:
        sizes = sorted({max(6, int(round(f * G))) for f in (0.5, 0.75, 1.0)})
    sizes = sorted({int(s) for s in sizes if 4 <= int(s) <= G})
    proto = clone(model) if model is not None else PieceModel(family=family)
    unit_name = shift.name[4:] if shift.name.startswith("new ") else "group"

    rng = np.random.default_rng(random_state)
    tasks = []
    for shape in shapes:
        for s in sizes:
            for share in (0.0, *shares):
                for _ in range(n_trials):
                    keep = labels if s == G else rng.choice(labels, s, replace=False)
                    tasks.append((shape, s, float(share), keep, int(rng.integers(2**31))))
    hits = Parallel(n_jobs=n_jobs)(
        delayed(_trial)(frame, y, w, units, keep, share, shape, proto, shift.name, seed)
        for shape, _, share, keep, seed in tasks
    )
    raw = pd.DataFrame({"shape": [t[0] for t in tasks], "units": [t[1] for t in tasks],
                        "share": [t[2] for t in tasks], "hit": np.asarray(hits, float)})  # fmt: skip
    cells = (
        raw.groupby(["shape", "units", "share"], sort=False)["hit"]
        .agg(detected="mean", trials="size")
        .reset_index()
    )
    null = cells[cells.share == 0.0].rename(columns={"detected": "rate"})[
        ["shape", "units", "rate", "trials"]
    ]
    power = cells[cells.share > 0.0].reset_index(drop=True)
    law = []
    for shape in shapes:
        b = _fit_law(power[power["shape"] == shape])
        law.append({"shape": shape, "intercept": float(b[0]), "slope": float(b[1])})
    return Detection(
        power=power, false_alarms=null.reset_index(drop=True), law=pd.DataFrame(law),
        unit=unit_name, n_units=G, level=float(level), horizon=float(horizon),
        meta={"rows": n, "sizes": sizes, "trials per cell": n_trials},
    )  # fmt: skip
