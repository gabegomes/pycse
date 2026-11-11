"""Pre-check: what the data can support, before anything is fitted.

Three questions decide in advance how much a ledger can say:

- how many independent values each variable has. A variable that is constant within a group (a
  property of the ligand, the patient, the site) has only as many values as there are groups,
  however many rows there are; a shape of it rests on that many points;
- how much of the target's variance lies between groups. That part can only be predicted for a
  new group through group-level variables;
- which variables are near-duplicates, so that an effect cannot be assigned to one of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._data import Design
from .shifts import NewGroups, Shift

__all__ = ["preflight", "Preflight"]


@dataclass
class Preflight:
    """Result of :func:`preflight`.

    Attributes:
        variables: one row per input column: its kind, its number of distinct values, and for
            each group shift the number of distinct values it takes across groups when it is
            constant within them (blank when it varies within groups).
        shifts: one row per shift: its units, how it will be split, and for a group shift the
            share of the target's variance that lies between groups.
        duplicates: pairs of numeric columns whose correlation is at least the threshold.
        warnings: plain-language notes on what limits the analysis.
    """

    variables: pd.DataFrame
    shifts: pd.DataFrame
    duplicates: pd.DataFrame
    warnings: list = field(default_factory=list)

    def __str__(self):
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            parts = ["Shifts", self.shifts.to_string(), "\nVariables", self.variables.to_string()]
            if len(self.duplicates):
                parts += ["\nNear-duplicate variables", self.duplicates.to_string(index=False)]
            if self.warnings:
                parts += ["\nNotes"] + [f"- {w}" for w in self.warnings]
        return "\n".join(parts)


def preflight(X, y, shifts=None, *, duplicate_r=0.9, few=8, n_splits=5):
    """Report what the data can support.

    Args:
        X: table or array of predictors.
        y: the target.
        shifts: a shift, or a list or dict of shifts, as for the ledger.
        duplicate_r: absolute correlation from which two numeric columns count as duplicates.
        few: a count of groups or distinct values below which a note is issued.
        n_splits: the folds the ledger will use, to say how a group shift will be split.

    Returns:
        :class:`Preflight`.
    """
    frame = Design()._frame(X)
    y = np.asarray(y, float).ravel()
    if shifts is None:
        shifts = {}
    elif isinstance(shifts, Shift):
        shifts = {shifts.name: shifts}
    elif not isinstance(shifts, dict):
        shifts = {s.name: s for s in shifts}
    notes = []

    rows = {}
    for c in frame.columns:
        col = frame[c]
        numeric = pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)
        k = int(col.nunique())
        kind = "text" if not numeric else "constant" if k <= 1 else "flag" if k == 2 else "number"
        rows[str(c)] = {"kind": kind, "distinct values": k}
    group_shifts = {n: s for n, s in shifts.items() if isinstance(s, NewGroups)}
    for name, s in group_shifts.items():
        g = pd.Series(s.groups)
        level = []
        for c in frame.columns:
            per_group = frame[c].groupby(g.values).nunique()
            if (per_group <= 1).all():  # a property of the group
                k = int(frame[c].groupby(g.values).first().nunique())
                level.append(k)
                if 1 < k < few and rows[str(c)]["kind"] == "number":
                    notes.append(
                        f"{c} is a property of the {name.removeprefix('new ')} and takes {k} values: "
                        f"any shape of it rests on {k} points."
                    )
            else:
                level.append(np.nan)
        for c, k in zip(frame.columns, level):
            rows[str(c)][f"values across groups ({name})"] = k
    variables = pd.DataFrame.from_dict(rows, orient="index")
    variables.index.name = "variable"
    for c in variables.columns[2:]:
        variables[c] = variables[c].astype("Int64")

    srows = {}
    total = float(np.sum((y - y.mean()) ** 2)) or 1.0
    for name, s in shifts.items():
        units = len(np.unique(s.units()))
        row = {"units": units}
        if isinstance(s, NewGroups):
            g = len(s.labels_)
            row["design"] = (
                "each group left out once" if g <= n_splits else f"{n_splits} folds of groups"
            )
            means = pd.Series(y).groupby(s.groups).transform("mean").to_numpy()
            share = float(np.sum((means - y.mean()) ** 2)) / total
            row["target variance between groups"] = round(share, 3)
            if g < few:
                notes.append(
                    f"{name}: only {g} groups. Each is left out once and the intervals will be wide; "
                    "a verdict of 'undetermined' is the likely and honest outcome."
                )
            if share > 0.5:
                notes.append(
                    f"{name}: {share:.0%} of the target's variance lies between groups. For a new group "
                    "that part can only come from variables that describe the group."
                )
        else:
            row["design"] = type(s).__name__
        srows[name] = row
    shift_table = pd.DataFrame.from_dict(srows, orient="index") if srows else pd.DataFrame()
    shift_table.index.name = "shift"

    num = frame.select_dtypes(include=[np.number, bool]).astype(float)
    num = num.loc[:, num.std() > 0]
    pairs = []
    if num.shape[1] > 1:
        r = np.corrcoef(num.to_numpy(), rowvar=False)
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                if abs(r[i, j]) >= duplicate_r:
                    pairs.append(
                        {
                            "variable": num.columns[i],
                            "duplicate of": num.columns[j],
                            "r": round(float(r[i, j]), 3),
                        }
                    )
    duplicates = pd.DataFrame(pairs, columns=["variable", "duplicate of", "r"])
    if len(duplicates):
        notes.append(
            f"{len(duplicates)} pair(s) of near-duplicate variables: an effect cannot be assigned to one "
            "member of a pair. Remove one of each, or read them as a block."
        )
    return Preflight(variables, shift_table, duplicates, notes)
