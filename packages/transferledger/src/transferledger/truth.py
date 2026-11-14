"""Problems with known answers, to check the method itself.

Each problem draws data from a stated generator and names the facts a correct analysis must
recover: a step at a known place, a product that is there, a noise variable that must stay out,
a variable that must not be said to transfer. :func:`scorecard` runs every problem over several
seeds and reports the share of runs in which each fact was recovered. The package's tests hold
those shares above stated floors, so a change that makes the method quietly worse fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .forms import sigmoid
from .model import PieceModel
from .shifts import ForwardInTime, NewGroups

__all__ = ["Problem", "PROBLEMS", "scorecard", "FLOORS"]

R = 1.987e-3  # kcal / (mol K)
SIGMA = np.array([0.78, 0.66, 0.54, 0.45, 0.23, 0.06, 0.0, -0.17, -0.20, -0.27, -0.83])


@dataclass
class Problem:
    """A generator and the facts to recover.

    Attributes:
        name: short identifier.
        about: one sentence on what it tests.
        make: ``make(seed) -> (X, y, shift)``; ``shift`` may be None.
        options: keyword arguments for :class:`~transferledger.model.PieceModel`.
        checks: ``{fact: check(model, X, y) -> bool}`` on the model chosen under the shift.
    """

    name: str
    about: str
    make: Callable
    checks: dict
    options: dict = field(default_factory=dict)


def _term(model, variable):
    t = model.terms_
    rows = t[t.variable == variable]
    return rows.iloc[0] if len(rows) else None


def _near(model, variable, form, value, tol):
    row = _term(model, variable)
    return row is not None and row["form"] == form and abs(row["location"] - value) <= tol


# ---- 1, 2: a rate law, with and without an interaction ------------------------------------------
def _rate_law(seed, interaction, n=600):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "sigma": rng.choice(SIGMA, n), "T": rng.uniform(288, 368, n), "V": rng.uniform(22, 50, n),
        "E": rng.uniform(-30, 0, n), "unrelated": rng.normal(size=n),
    })  # fmt: skip
    s = sigmoid(X.V - 32.0)
    rho = (2.0 - 2.5 * s) if interaction else 1.5
    y = (-3 + rho * X.sigma - 1.5 * s - 15 / (2.303 * R) * (1 / X["T"] - 1 / 298.15)
         - 0.01 * (X.E + 15) ** 2 + 0.15 * rng.normal(size=n))  # fmt: skip
    return X, y.to_numpy(), None


def _barrier(model, X, y):
    row = _term(model, "T")
    return (
        row is not None
        and row["form"] == "inverse"
        and abs(-row["weight"] * X["T"].mean() * 2.303 * R - 15.0) <= 1.0
    )


def _slopes(model, X, y):
    t = model.terms_.set_index("variable")
    if "sigma" not in t.index or "V x sigma" not in t.index and "sigma x V" not in t.index:
        return False
    prod = t.loc["V x sigma"] if "V x sigma" in t.index else t.loc["sigma x V"]
    sd = X.sigma.std(ddof=0)
    below, above = t.loc["sigma", "weight"] / sd, (t.loc["sigma", "weight"] + prod["weight"]) / sd
    return abs(below - 2.0) <= 0.3 and abs(above + 0.5) <= 0.3


RATE_LAW = Problem(
    "rate law with an interaction",
    "A cliff, an optimum, an Arrhenius term, and a slope that changes sign across the cliff.",
    lambda seed: _rate_law(seed, True),
    {
        "step found at 32 (within 0.5)": lambda m, X, y: _near(m, "V", "step", 32.0, 0.5),
        "optimum found at -15 (within 0.7)": lambda m, X, y: _near(m, "E", "quadratic", -15.0, 0.7),
        "barrier of 15 recovered (within 1)": _barrier,
        "slopes on both sides of the cliff (within 0.3)": _slopes,
        "unrelated variable left out": lambda m, X, y: "unrelated" not in m.variables_,
    },
    {"forms": {"T": ["inverse"]}},
)
ADDITIVE = Problem(
    "rate law without an interaction",
    "The same terms, added. No product should be invented.",
    lambda seed: _rate_law(seed, False),
    {
        "no product invented": lambda m, X, y: len(m.products_) == 0,
        "unrelated variable left out": lambda m, X, y: "unrelated" not in m.variables_,
    },
    {"forms": {"T": ["inverse"]}},
)


# ---- 3, 4: group properties that are noise, or real -------------------------------------------
def _groups(seed, real, n_groups=12, per=40):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    x = rng.normal(size=len(g))
    Z = rng.normal(size=(n_groups, 3))
    offset = 2.0 * rng.normal(size=n_groups)
    if real:
        offset = 2.0 * Z[:, 0] + 0.7 * rng.normal(size=n_groups)
    y = 1.5 * x + offset[g] + 0.2 * rng.normal(size=len(g))
    X = pd.DataFrame(np.column_stack([x, Z[g]]), columns=["x", "z1", "z2", "z3"])
    return X, y, NewGroups(g, "new group")


GROUP_NOISE = Problem(
    "group properties that are noise",
    "Twelve groups with offsets of their own and three group properties unrelated to them.",
    lambda seed: _groups(seed, False),
    {
        "no noise property admitted": lambda m, X, y: not (set(m.variables_) & {"z1", "z2", "z3"}),
        "the real variable kept": lambda m, X, y: "x" in m.variables_,
    },
)
GROUP_SIGNAL = Problem(
    "one real group property",
    "Twelve groups whose offsets follow one of three group properties.",
    lambda seed: _groups(seed, True),
    {
        "the real property found": lambda m, X, y: "z1" in m.variables_,
        "no noise property admitted": lambda m, X, y: not (set(m.variables_) & {"z2", "z3"}),
    },
)


# ---- 5: a yes/no outcome with a threshold ----------------------------------------------------
def _threshold(seed, n=800):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "dose": rng.uniform(0, 10, n),
            "age": rng.uniform(20, 80, n),
            "unrelated": rng.normal(size=n),
        }
    )
    eta = -2.0 + 4.0 * sigmoid((X.dose - 6.0) / 0.3) + 0.03 * (X.age - 50)
    y = (rng.uniform(size=n) < sigmoid(eta)).astype(float)
    return X, y, None


THRESHOLD = Problem(
    "yes/no outcome with a threshold",
    "The odds of the outcome step up at a dose of 6.",
    _threshold,
    {
        "threshold found at 6 (within 0.6)": lambda m, X, y: _near(m, "dose", "step", 6.0, 0.6),
        "unrelated variable left out": lambda m, X, y: "unrelated" not in m.variables_,
        "probabilities returned": lambda m, X, y: bool(
            m.predict(X).min() >= 0 and m.predict(X).max() <= 1
        ),
    },
    {"family": "binomial"},
)


# ---- 6: counts ----------------------------------------------------------------------------------
def _counts(seed, n=600):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"exposure": rng.uniform(0, 4, n), "unrelated": rng.normal(size=n)})
    y = rng.poisson(np.exp(0.2 + 0.6 * X.exposure)).astype(float)
    return X, y, None


def _rate(model, X, y):
    row = _term(model, "exposure")
    return (
        row is not None
        and row["form"] == "linear"
        and abs(row["weight"] / X.exposure.std(ddof=0) - 0.6) <= 0.1
    )


COUNTS = Problem(
    "counts with a log-linear rate",
    "A count whose mean grows by a factor of exp(0.6) per unit of exposure.",
    _counts,
    {
        "rate of 0.6 per unit recovered (within 0.1)": _rate,
        "unrelated variable left out": lambda m, X, y: "unrelated" not in m.variables_,
    },
    {"family": "poisson"},
)


# ---- 7: a cycle that interpolates but does not extrapolate ---------------------------------------
def _cycle(seed, n=600):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    X = pd.DataFrame({"x": rng.normal(size=n), "run_number": t})
    y = 1.5 * X.x + 1.5 * np.sin(2 * np.pi * (t / 130.0 + rng.uniform())) + 0.2 * rng.normal(size=n)
    return X, y.to_numpy(), ForwardInTime(t, "later runs")


CYCLE = Problem(
    "a cycle over time",
    "The response cycles with the run number. A curve of run number fits the past, and holding its last "
    "level says nothing about later runs.",
    _cycle,
    {
        "run number left out when chosen for later runs": lambda m, X, y: (
            "run_number" not in m.variables_
        ),
        "the real variable kept": lambda m, X, y: "x" in m.variables_,
    },
)

#: Every problem, in the order the scorecard reports them.
PROBLEMS = [RATE_LAW, ADDITIVE, GROUP_NOISE, GROUP_SIGNAL, THRESHOLD, COUNTS, CYCLE]

#: The share of runs in which each fact must be recovered. A fact not listed needs ``DEFAULT``.
DEFAULT_FLOOR = 0.8
FLOORS = {
    ("group properties that are noise", "no noise property admitted"): 0.8,
    ("one real group property", "no noise property admitted"): 0.8,
    ("a cycle over time", "run number left out when chosen for later runs"): 0.7,
    ("rate law with an interaction", "slopes on both sides of the cliff (within 0.3)"): 0.7,
    ("one real group property", "the real property found"): 0.7,
}


def scorecard(seeds=range(10), problems=None):
    """Run every problem over ``seeds``; one row per problem and fact with its pass rate.

    Returns a DataFrame with columns ``problem``, ``fact``, ``passed`` (share of seeds),
    ``floor`` (the share the tests require) and ``ok``.
    """
    rows = []
    for prob in problems or PROBLEMS:
        hits = {k: [] for k in prob.checks}
        for seed in seeds:
            X, y, shift = prob.make(seed)
            model = PieceModel(select_on=shift, **prob.options).fit(X, y)
            for k, check in prob.checks.items():
                try:
                    hits[k].append(bool(check(model, X, y)))
                except Exception:
                    hits[k].append(False)
        for k, v in hits.items():
            floor = FLOORS.get((prob.name, k), DEFAULT_FLOOR)
            rows.append(
                {"problem": prob.name, "fact": k, "passed": float(np.mean(v)), "floor": floor}
            )
    out = pd.DataFrame(rows)
    out["ok"] = out["passed"] >= out["floor"]
    return out
