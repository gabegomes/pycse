"""Shifts: named ways in which the cases a model will meet differ from the cases it was fit on.

"Does this transfer?" has no answer until one says *to what*: a group that was never seen (a new
molecule, patient, site, batch), a later time, a value range beyond the training data. A
:class:`Shift` turns that into train/test splits in which every test case is new in the named
sense, and the same shift can be applied inside a training set, so that a model can be *chosen*
by how well it handles the shift, not only scored on it.

:class:`Reference` is the baseline every shift is compared with: random splits, in which the
test cases resemble the training cases.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

__all__ = ["Shift", "Reference", "NewGroups", "ForwardInTime", "BeyondRange", "Splitter"]


class Shift:
    """Base class. A shift is defined on a fixed set of rows (those of the data it was built for).

    Attributes:
        name: how the shift is called in tables ("new ligand", "later years").
        repeatable: whether different seeds give different splits (so repeats are informative).
    """

    name = "shift"
    repeatable = True

    def __len__(self):
        raise NotImplementedError

    def splits(self, n_splits=5, seed=0):
        """List of ``(train, test)`` index arrays into the shift's rows."""
        raise NotImplementedError

    def subset(self, rows):
        """The same shift, restricted to ``rows`` (and re-indexed from zero)."""
        raise NotImplementedError

    def units(self):
        """The resampling unit of each row: rows that stand or fall together share a label.
        Intervals are computed by resampling these units."""
        return np.arange(len(self))

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, {len(self)} rows)"


class Reference(Shift):
    """No shift: random folds over rows, stratified by ``stratify`` when given.

    Args:
        n: number of rows.
        stratify: optional class label per row.
        units: optional resampling unit per row (pass the groups when rows of the same group are
            not independent, so that intervals are not too narrow).
    """

    def __init__(self, n, stratify=None, units=None, name="reference"):
        self.n = int(n)
        self.stratify = None if stratify is None else np.asarray(stratify)
        self._units = None if units is None else np.asarray(units)
        self.name = name

    def __len__(self):
        return self.n

    def splits(self, n_splits=5, seed=0):
        k = max(2, min(n_splits, self.n))
        if (
            self.stratify is not None
            and np.min(np.unique(self.stratify, return_counts=True)[1]) >= k
        ):
            cv = StratifiedKFold(k, shuffle=True, random_state=seed)
            return list(cv.split(np.zeros(self.n), self.stratify))
        return list(KFold(k, shuffle=True, random_state=seed).split(np.zeros(self.n)))

    def subset(self, rows):
        rows = np.asarray(rows)
        return Reference(
            len(rows),
            None if self.stratify is None else self.stratify[rows],
            None if self._units is None else self._units[rows],
            self.name,
        )

    def units(self):
        return np.arange(self.n) if self._units is None else self._units


class NewGroups(Shift):
    """Whole groups are held out: every test row belongs to a group absent from training.

    With at least ``n_splits`` groups the groups are dealt into folds at random (different seeds
    give different deals). With fewer, each group is held out once (leave-one-group-out), which
    is the only honest design when a factor has three or four levels.

    Args:
        groups: the group label of each row.
        name: how to call it ("new ligand"); defaults to "new group".
    """

    def __init__(self, groups, name="new group"):
        self.groups = np.asarray(groups)
        self.name = name
        self.labels_ = np.unique(self.groups)
        if len(self.labels_) < 2:
            raise ValueError(f"{name}: holding out groups needs at least two groups")

    @classmethod
    def from_clusters(cls, X, k=10, name="new cluster", seed=0):
        """Groups from k-means clusters of the standardized columns of ``X``, for data without
        group labels: every test row then comes from a region of the variables not trained on."""
        from sklearn.cluster import KMeans

        X = np.asarray(X, float)
        sd = X.std(axis=0)
        Z = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
        return cls(KMeans(k, n_init=4, random_state=seed).fit_predict(Z), name)

    def __len__(self):
        return len(self.groups)

    @property
    def repeatable(self):
        return len(self.labels_) > 5

    def splits(self, n_splits=5, seed=0):
        g = len(self.labels_)
        if g <= n_splits:  # leave one group out
            return [
                (np.flatnonzero(self.groups != a), np.flatnonzero(self.groups == a))
                for a in self.labels_
            ]
        cv = GroupKFold(n_splits, shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(self)), groups=self.groups))

    def subset(self, rows):
        return NewGroups(self.groups[np.asarray(rows)], self.name)

    def units(self):
        return self.groups


class ForwardInTime(Shift):
    """Train on the past, test on what comes next.

    Rows are ordered by ``time`` and cut into ``n_splits + 1`` consecutive blocks; fold ``k``
    trains on the first ``k`` blocks and tests on block ``k + 1`` (an expanding window).

    Args:
        time: a sortable value per row (a date, a run number).
        name: how to call it.
        n_units: intervals resample this many consecutive time blocks.
    """

    repeatable = False

    def __init__(self, time, name="later in time", n_units=20):
        self.time = np.asarray(time)
        self.name = name
        self.n_units = n_units

    def __len__(self):
        return len(self.time)

    def splits(self, n_splits=5, seed=0):
        order = np.argsort(self.time, kind="stable")
        blocks = np.array_split(order, n_splits + 1)
        return [
            (np.sort(np.concatenate(blocks[: k + 1])), np.sort(blocks[k + 1]))
            for k in range(n_splits)
        ]

    def subset(self, rows):
        return ForwardInTime(self.time[np.asarray(rows)], self.name, self.n_units)

    def units(self):
        order = np.argsort(self.time, kind="stable")
        u = np.empty(len(self), int)
        for i, b in enumerate(np.array_split(order, min(self.n_units, len(self)))):
            u[b] = i
        return u


class BeyondRange(Shift):
    """Extrapolation in one variable: train on part of its range, test beyond it.

    Args:
        values: the variable's value per row.
        side: ``"high"`` (test on the top ``fraction``), ``"low"`` or ``"both"`` (two folds).
        fraction: the share of rows held out on a side.
        name: how to call it ("higher temperature").
    """

    repeatable = False

    def __init__(self, values, side="high", fraction=0.2, name="beyond the range"):
        if side not in ("high", "low", "both"):
            raise ValueError("side must be 'high', 'low' or 'both'")
        self.values = np.asarray(values, float)
        self.side, self.fraction, self.name = side, float(fraction), name

    def __len__(self):
        return len(self.values)

    def splits(self, n_splits=5, seed=0):
        out = []
        for s in ("high", "low") if self.side == "both" else (self.side,):
            q = np.quantile(self.values, 1 - self.fraction if s == "high" else self.fraction)
            test = self.values > q if s == "high" else self.values < q
            if test.any() and not test.all():
                out.append((np.flatnonzero(~test), np.flatnonzero(test)))
        if not out:
            raise ValueError(
                f"{self.name}: the variable has too few distinct values to hold out a range"
            )
        return out

    def subset(self, rows):
        return BeyondRange(self.values[np.asarray(rows)], self.side, self.fraction, self.name)


class Splitter(Shift):
    """Any scikit-learn splitter as a shift.

    Args:
        cv: an object with ``split(X, y, groups)``.
        n: number of rows.
        y, groups: passed to ``cv.split``.
        units: resampling unit per row (defaults to ``groups``, else rows).
    """

    repeatable = False

    def __init__(self, cv, n, y=None, groups=None, units=None, name="custom split"):
        self.cv, self.n, self.y, self.groups, self.name = cv, int(n), y, groups, name
        self._units = units if units is not None else groups

    def __len__(self):
        return self.n

    def splits(self, n_splits=5, seed=0):
        return [
            (np.asarray(a), np.asarray(b))
            for a, b in self.cv.split(np.zeros(self.n), self.y, self.groups)
        ]

    def subset(self, rows):
        raise NotImplementedError("a custom splitter cannot be applied inside a training set")

    def units(self):
        return np.arange(self.n) if self._units is None else np.asarray(self._units)
