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
- the units needed for a weaker property, from how detection grows with the number of units.

The target is never simulated: its noise, its rows and its other variables are the real ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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
        law: per shape, the fitted ``intercept``, ``per log unit`` and ``per logit share`` of
            ``probit(detected) = intercept + a * log(units) + b * logit(share)``.
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
        share = np.clip(np.asarray(share, float), 1e-6, 1 - 1e-6)
        z = (k["intercept"] + k["per log unit"] * np.log(np.asarray(units, float))
             + k["per logit share"] * np.log(share / (1 - share)))  # fmt: skip
        return norm.cdf(z)

    def floor(self, shape="trend", units=None):
        """The smallest share a property must capture to be established with ``units`` units
        (default: the units in hand). None when even a near-perfect property would not be."""
        units = self.n_units if units is None else units
        grid = np.linspace(0.02, 0.98, 97)
        ok = np.flatnonzero(self.rate(units, grid, shape) >= self.level)
        return float(grid[ok[0]]) if len(ok) else None

    def needed(self, share, shape="trend"):
        """Units needed to establish a property that captures ``share``. None beyond the
        horizon (a multiple of the units in hand): the law is not trusted that far."""
        last = int(np.floor(self.horizon * self.n_units))
        grid = np.arange(3, last + 1)
        ok = np.flatnonzero(self.rate(grid, share, shape) >= self.level)
        return int(grid[ok[0]]) if len(ok) else None

    def table(self, shares=(0.3, 0.5, 0.7), shapes=None):
        """Units needed, per shape and share; a dash where the horizon is passed."""
        shapes = shapes or list(self.law["shape"])
        last = int(np.floor(self.horizon * self.n_units))
        rows = {s: {f"{int(100 * q)} %": self.needed(q, s) for q in shares} for s in shapes}
        out = pd.DataFrame(rows).T
        out = out.astype(object).where(out.notna(), f"more than {last}")
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
        for s in self.law["shape"]:
            f = self.floor(s)
            lines.append(
                f"{s}: with {self.n_units} {self.unit}s a property must capture "
                + (f"at least {f:.0%}" if f is not None else "more than any tested share")
                + f" of the differences between {self.unit}s to be established"
            )
        lines += ["", f"{self.unit.capitalize()}s needed for a weaker property:",
                  self.table().to_string()]  # fmt: skip
        return "\n".join(lines)

    def to_dict(self):
        return {
            "unit": self.unit, "n_units": self.n_units, "level": self.level, "horizon": self.horizon,
            "power": self.power.to_dict("records"), "false_alarms": self.false_alarms.to_dict("records"),
            "law": self.law.to_dict("records"), "meta": self.meta,
        }  # fmt: skip


def _fit_law(cells):
    """Probit regression of the detection rate on log(units) and logit(share), by maximum
    likelihood on the counts (Newton steps; three parameters)."""
    m = np.log(cells["units"].to_numpy(float))
    q = cells["share"].to_numpy(float)
    X = np.column_stack([np.ones(len(m)), m, np.log(q / (1 - q))])
    k = (cells["detected"] * cells["trials"]).to_numpy(float)
    n = cells["trials"].to_numpy(float)
    beta = np.zeros(3)
    for _ in range(60):
        z = X @ beta
        p = np.clip(norm.cdf(z), 1e-6, 1 - 1e-6)
        d = norm.pdf(z)
        score = X.T @ (d * (k - n * p) / (p * (1 - p)))
        info = X.T @ (X * (n * d**2 / (p * (1 - p)))[:, None]) + 1e-6 * np.eye(3)
        step = np.linalg.solve(info, score)
        beta = beta + np.clip(step, -2.0, 2.0)
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


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
        law.append(
            {"shape": shape, "intercept": b[0], "per log unit": b[1], "per logit share": b[2]}
        )
    return Detection(
        power=power, false_alarms=null.reset_index(drop=True), law=pd.DataFrame(law),
        unit=unit_name, n_units=G, level=float(level), horizon=float(horizon),
        meta={"rows": n, "sizes": sizes, "trials per cell": n_trials},
    )  # fmt: skip
