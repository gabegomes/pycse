"""The transfer ledger: what a model, and each variable in it, is worth under each shift.

Everything is scored twice or more on held-out rows: under the *reference* (random folds, where
test cases resemble training cases) and under each named shift (a new group, a later time, a
range beyond the data). The unit of account is **skill**: the share of the baseline's loss a
model removes, where the baseline predicts the training mean. Skill 0 is the baseline; negative
is worse than knowing nothing.

Every number carries an interval, from leaving out the shift's units one at a time (whole
groups, when groups are held out), and verdicts are read from the intervals, so a variable whose evidence
is thin is reported as undetermined rather than assigned a side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from scipy.stats import t as student_t
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors

from ._data import Design
from .curves import get_family
from .model import PieceModel, fit_weights, linear_score
from .shifts import Reference, Shift

__all__ = ["transfer_ledger", "Ledger"]

BASELINE = "baseline (training mean)"
EQUATION = "equation"
CURVES = "curves"


def _loss(fam, y, mu, w):
    if fam.name == "binomial":  # keep one confident miss from dominating the account
        mu = np.clip(mu, 1e-3, 1 - 1e-3)
    return w * fam.loss(y, mu)


def _predict(model, Xn, fam):
    if fam.name == "binomial" and hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(Xn))[:, 1]
    return np.asarray(model.predict(Xn), float).ravel()


def _fold(X, y, w, tr, te, proto, variants, extra, fam, blocks):
    """Everything one train/test split contributes. ``variants`` maps a name to the shift the
    equation is chosen under, already restricted to the training rows (None: random folds)."""
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte, wtr, wte = y[tr], y[te], w[tr], w[te]
    out = {"rows": te, "loss": {}, "drop": {}, "selected": {}, "pred": {}}
    mu0 = np.full(len(te), np.average(ytr, weights=wtr))
    out["loss"][BASELINE] = _loss(fam, yte, mu0, wte)
    out["pred"][BASELINE] = mu0
    reference_model = None
    for name, shift in variants.items():
        m = clone(proto).set_params(select_on=shift).fit(Xtr, ytr, sample_weight=wtr)
        rung = EQUATION if name is None else f"{EQUATION}, chosen for {name}"
        mu = m.predict(Xte)
        out["loss"][rung], out["pred"][rung] = _loss(fam, yte, mu, wte), mu
        out["selected"][rung] = list(m.variables_)
        Xn_te = m.design_.transform(Xte)
        rung = CURVES if name is None else f"{CURVES}, tuned for {name}"
        mu = np.asarray(m.source_.predict(Xn_te), float)
        out["loss"][rung], out["pred"][rung] = _loss(fam, yte, mu, wte), mu
        if name is None:
            reference_model = m
    m = reference_model
    Xn_tr, Xn_te = m.design_.transform(Xtr), m.design_.transform(Xte)
    # what each variable of the reference equation adds: refit the weights without it
    C_tr, C_te, owners = m._columns(Xn_tr), m._columns(Xn_te), m._term_variables()
    full = out["loss"][EQUATION]

    def without(names):
        keep = [k for k, s in enumerate(owners) if not (s & names)]
        coef = fit_weights(C_tr[:, keep], ytr, wtr, fam)
        return _loss(fam, yte, fam.mean(linear_score(C_te[:, keep], coef)), wte) - full

    for v in m.variables_:
        out["drop"][v] = without({v})
    out["drop_block"] = {b: without(set(names)) for b, names in blocks.items()}
    for name, est in extra.items():
        e = clone(est).fit(Xn_tr, ytr if fam.name != "binomial" else ytr.astype(int))
        mu = _predict(e, Xn_te, fam)
        out["loss"][name], out["pred"][name] = _loss(fam, yte, mu, wte), mu
    # how far each test row is from the nearest training row, over all varying columns
    sd = Xn_tr.std(axis=0)
    use = sd > 0
    if use.any():
        Ztr = (Xn_tr[:, use] - Xn_tr[:, use].mean(axis=0)) / sd[use]
        Zte = (Xn_te[:, use] - Xn_tr[:, use].mean(axis=0)) / sd[use]
        d = NearestNeighbors(n_neighbors=1).fit(Ztr).kneighbors(Zte)[0][:, 0]
        out["dist"] = d / np.sqrt(use.sum())
    else:
        out["dist"] = np.zeros(len(te))
    return out


def _intervals(unit_idx, columns, base, level):
    """Intervals for ``sum(column) / sum(base)`` from a delete-one-unit jackknife.

    Each unit (a group, when groups are held out) is left out in turn; the spread of the
    resulting ratios gives a standard error, and the interval uses Student's t with one fewer
    degrees of freedom than there are units. With four groups that is wide, as it should be,
    and nothing here depends on a random seed.

    Args:
        unit_idx: integer unit of each row.
        columns: ``(n, k)`` per-row quantities.
        base: ``(n,)`` per-row baseline loss.

    Returns:
        ``(point, lo, hi, share)`` arrays of length ``k``; ``share`` is the fraction of units
        whose own total is positive.
    """
    g = int(unit_idx.max()) + 1
    Q = np.zeros((g, columns.shape[1]))
    np.add.at(Q, unit_idx, columns)
    b = np.bincount(unit_idx, weights=base, minlength=g)
    point = Q.sum(axis=0) / b.sum()
    if g < 2:
        return (
            point,
            np.full_like(point, -np.inf),
            np.full_like(point, np.inf),
            (Q > 0).mean(axis=0),
        )
    loo = (Q.sum(axis=0)[None, :] - Q) / (b.sum() - b)[:, None]
    se = np.sqrt((g - 1) / g * np.sum((loo - loo.mean(axis=0)) ** 2, axis=0))
    half = student_t.ppf(0.5 + level / 2, g - 1) * se
    return point, point - half, point + half, (Q > 0).mean(axis=0)


def _side(lo, hi):
    return "helps" if lo > 0 else "hurts" if hi < 0 else "unclear"


def _verdict(ref, shifted):
    if ref == "helps":
        return {"helps": "transfers", "hurts": "does not transfer", "unclear": "undetermined"}[
            shifted
        ]
    return "helps only under the shift" if shifted == "helps" else "not established"


@dataclass
class Ledger:
    """The result of :func:`transfer_ledger`.

    Attributes:
        skill: one row per model rung and scheme: its skill and interval.
        contributions: one row per variable and scheme: the skill the reference equation loses
            when the variable is removed (and its weights refitted), with its interval, the share
            of units (groups, when groups are held out) for which removing it made things worse,
            and how often the variable was selected.
        verdicts: one row per variable, one column per shift: ``transfers`` (helps on the
            reference and under the shift), ``does not transfer`` (helps on the reference, hurts
            under the shift), ``undetermined`` (helps on the reference; under the shift the
            interval spans zero), ``helps only under the shift``, or ``not established`` (no
            clear benefit even on the reference; the numbers say whether it hurts).
        blocks, block_verdicts: the same two tables for named blocks of variables removed
            together, which is the right question when variables are near-duplicates and each
            can stand in for another.
        memorized: per shift, the reference skill the equation gives up when its variables must
            earn their place under that shift: what it knew only by having seen the cases.
        selection: how often each variable was selected, per way of choosing the equation.
        distance: per shift, one row per held-out unit: its mean distance to the nearest training
            row and each rung's mean loss there.
        distance_trend: per shift and rung, the rank correlation of loss with distance.
        models: the equations fitted to all rows, per way of choosing.
        predictions: per scheme, the held-out prediction of every row by every rung.
        family, level, n_splits, n_repeats, schemes.
    """

    skill: pd.DataFrame
    contributions: pd.DataFrame
    verdicts: pd.DataFrame
    blocks: pd.DataFrame
    block_verdicts: pd.DataFrame
    memorized: pd.DataFrame
    selection: pd.DataFrame
    distance: dict
    distance_trend: pd.DataFrame
    models: dict
    predictions: dict
    family: str
    level: float
    n_splits: int
    n_repeats: int
    schemes: list = field(default_factory=list)

    @property
    def ladder(self):
        """Skill of every rung (rows) under every scheme (columns)."""
        rungs = list(dict.fromkeys(self.skill["rung"]))
        return self.skill.pivot(index="rung", columns="scheme", values="skill").loc[
            rungs, self.schemes
        ]

    def ladder_text(self, digits=3):
        """The ladder with intervals, as text cells."""
        s = self.skill.assign(
            cell=[f"{a:.{digits}f} [{b:.{digits}f}, {c:.{digits}f}]" for a, b, c in
                  zip(self.skill.skill, self.skill.lo, self.skill.hi)]
        )  # fmt: skip
        rungs = list(dict.fromkeys(s["rung"]))
        return s.pivot(index="rung", columns="scheme", values="cell").loc[rungs, self.schemes]

    def equation(self, shift=None):
        """The equation fitted to all rows, chosen on the reference (None) or for a shift."""
        return self.models[
            EQUATION if shift is None else f"{EQUATION}, chosen for {shift}"
        ].equation()

    def summary(self):
        pct = f"{100 * self.level:.0f} %"
        with pd.option_context(
            "display.width", 220, "display.max_columns", 30, "display.max_colwidth", 60
        ):
            c = self.contributions.assign(
                cell=[f"{a:+.3f} [{b:+.3f}, {d:+.3f}]" for a, b, d in
                      zip(self.contributions.contribution, self.contributions.lo, self.contributions.hi)]
            )  # fmt: skip
            order = list(dict.fromkeys(c["variable"]))
            wide = c.pivot(index="variable", columns="scheme", values="cell").loc[
                order, self.schemes
            ]
            parts = [
                f"Skill: share of the baseline's loss removed, on held-out rows ({self.n_splits} folds "
                f"x {self.n_repeats} repeats; {pct} intervals over units)",
                self.ladder_text().to_string(),
                "\nMemorized: reference skill given up when variables must earn their place under the shift",
                self.memorized.round(3).to_string(),
                "\nWhat each variable adds to the reference equation (skill lost when it is removed)",
                wide.to_string(),
                "\nVerdicts",
                self.verdicts.to_string(),
            ]
            if len(self.blocks):
                b = self.blocks.assign(
                    cell=[f"{a:+.3f} [{lo:+.3f}, {hi:+.3f}]" for a, lo, hi in
                          zip(self.blocks.contribution, self.blocks.lo, self.blocks.hi)]
                )  # fmt: skip
                names = list(dict.fromkeys(b["block"]))
                wide = b.pivot(index="block", columns="scheme", values="cell").loc[
                    names, self.schemes
                ]
                parts += ["\nBlocks of variables removed together", wide.to_string(),
                          self.block_verdicts.to_string()]  # fmt: skip
        return "\n".join(parts)

    def __str__(self):
        return self.summary()


def transfer_ledger(X, y, shifts, *, model=None, models=None, family="gaussian", sample_weight=None,
                    blocks=None, reference=None, n_splits=5, n_repeats=5, level=0.95,
                    random_state=0, n_jobs=None, verbose=False):  # fmt: skip
    """Account for what an equation, and each variable in it, is worth under each shift.

    Args:
        X: a table (DataFrame) or array of predictors. Text columns are allowed.
        y: the target.
        shifts: a :class:`~transferledger.shifts.Shift`, or a list or ``{name: Shift}`` of them,
            defined over the rows of ``X`` (for example ``NewGroups(ligand, "new ligand")``).
        model: a :class:`~transferledger.model.PieceModel` to use as the prototype (its
            ``select_on`` is set by the ledger). Default: ``PieceModel(family=family)``.
        models: ``{name: estimator}`` of other models to score as extra rungs (they are fitted
            on the numeric design; a classifier's ``predict_proba`` is used for a 0/1 target).
        family: ``"gaussian"``, ``"binomial"`` or ``"poisson"``; sets the loss.
        sample_weight: optional weight per row.
        blocks: ``{name: [columns]}`` of variables to also remove together (for example all the
            descriptors of one component).
        reference: the reference scheme. Default: random folds with rows as units. The
            reference asks about new rows from the *same* groups, so its intervals are
            conditional on the groups at hand; pass ``Reference(n, units=groups)`` to ask about
            a new sample of groups instead.
        n_splits: folds per scheme. A group shift with fewer groups leaves one out at a time.
        n_repeats: repeats of schemes whose splits depend on a seed.
        level: coverage of the intervals.
        random_state: seed for the folds.
        n_jobs: parallel workers over folds (joblib).

    Returns:
        :class:`Ledger`.
    """
    fam = get_family(family)
    frame = Design()._frame(X)
    n = len(frame)
    y = np.asarray(y, float).ravel()
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
    if isinstance(shifts, Shift):
        shifts = [shifts]
    if not isinstance(shifts, dict):
        shifts = {s.name: s for s in shifts}
    for name, s in shifts.items():
        if len(s) != n:
            raise ValueError(f"shift {name!r} covers {len(s)} rows but X has {n}")
    if reference is None:
        reference = Reference(n, stratify=y if fam.name == "binomial" else None)
    proto = clone(model) if model is not None else PieceModel(family=family)
    if proto.family != fam.name:
        raise ValueError(f"the model's family is {proto.family!r} but the ledger's is {fam.name!r}")
    extra = dict(models or {})
    blocks = {b: list(v) for b, v in (blocks or {}).items()}
    schemes = {"reference": reference, **shifts}

    def variants_for(tr):
        v = {None: None}
        for name, s in shifts.items():
            try:
                v[name] = s.subset(tr)
            except NotImplementedError:
                pass  # a custom splitter cannot be applied inside the training rows
        return v

    tasks = []
    for sname, s in schemes.items():
        for r in range(n_repeats if s.repeatable else 1):
            for tr, te in s.splits(n_splits, random_state + r):
                tasks.append((sname, tr, te))
    run = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)
    results = run(
        delayed(_fold)(frame, y, w, tr, te, proto, variants_for(tr), extra, fam, blocks)
        for _, tr, te in tasks
    )

    found = list(results[0]["loss"])
    rungs = (
        [BASELINE]
        + [k for k in found if k.startswith(EQUATION)]
        + [k for k in found if k.startswith(CURVES)]
        + [k for k in found if k in extra]
    )
    variables = list(dict.fromkeys(v for r in results for v in r["drop"]))
    skill_rows, contrib_rows, sides, distance, trend_rows, predictions = [], [], {}, {}, [], {}
    block_rows, block_sides = [], {}
    reference_boot = None
    for sname, s in schemes.items():
        mine = [r for (name, _, _), r in zip(tasks, results) if name == sname]
        count = np.zeros(n)
        L = {k: np.zeros(n) for k in rungs}
        Dv = {v: np.zeros(n) for v in variables}
        Db = {b: np.zeros(n) for b in blocks}
        dist = np.zeros(n)
        pred = {k: np.zeros(n) for k in rungs}
        for r in mine:
            rows = r["rows"]
            count[rows] += 1
            dist[rows] += r["dist"]
            for k in rungs:
                L[k][rows] += r["loss"][k]
                pred[k][rows] += r["pred"][k]
            for v, d in r["drop"].items():
                Dv[v][rows] += d
            for b, d in r["drop_block"].items():
                Db[b][rows] += d
        seen = count > 0
        c = count[seen]
        _, unit_idx = np.unique(np.asarray(s.units())[seen], return_inverse=True)
        base = L[BASELINE][seen] / c
        cols = np.column_stack(
            [L[k][seen] / c for k in rungs]
            + [Dv[v][seen] / c for v in variables]
            + [Db[b][seen] / c for b in blocks]
        )
        point, lo, hi, share = _intervals(unit_idx, cols, base, level)
        for i, k in enumerate(rungs):  # skill = 1 - loss / base, so the interval flips
            skill_rows.append(
                {
                    "rung": k,
                    "scheme": sname,
                    "skill": 1 - point[i],
                    "lo": 1 - hi[i],
                    "hi": 1 - lo[i],
                }
            )
        for j, v in enumerate(variables):
            i = len(rungs) + j
            contrib_rows.append(
                {
                    "variable": v,
                    "scheme": sname,
                    "contribution": point[i],
                    "lo": lo[i],
                    "hi": hi[i],
                    "units helped": share[i],
                }  # fmt: skip
            )
            sides[(v, sname)] = _side(lo[i], hi[i])
        for j, b in enumerate(blocks):
            i = len(rungs) + len(variables) + j
            block_rows.append(
                {
                    "block": b,
                    "scheme": sname,
                    "contribution": point[i],
                    "lo": lo[i],
                    "hi": hi[i],
                    "units helped": share[i],
                }  # fmt: skip
            )
            block_sides[(b, sname)] = _side(lo[i], hi[i])
        if sname == "reference":
            reference_boot = (unit_idx, {k: L[k][seen] / c for k in rungs}, base)
        predictions[sname] = pd.DataFrame(
            {k: np.where(seen, pred[k] / np.maximum(count, 1), np.nan) for k in rungs}
        )
        # loss against distance to the training rows, per unit
        per_unit = pd.DataFrame({"unit": np.asarray(s.units())[seen], "distance": dist[seen] / c,
                                 **{k: L[k][seen] / c for k in rungs}})  # fmt: skip
        per_unit = per_unit.groupby("unit").mean()
        distance[sname] = per_unit
        if len(per_unit) > 3 and per_unit["distance"].std() > 0:
            for k in rungs:
                trend_rows.append({"scheme": sname, "rung": k,
                                   "rank correlation": spearmanr(per_unit["distance"], per_unit[k]).statistic})  # fmt: skip

    skill = pd.DataFrame(skill_rows)
    contributions = pd.DataFrame(contrib_rows)
    # selection frequencies per way of choosing
    sel = {}
    for r in results:
        for rung, chosen in r["selected"].items():
            sel.setdefault(rung, []).append(set(chosen))
    names = list(dict.fromkeys(v for sets in sel.values() for s in sets for v in s))
    selection = pd.DataFrame({rung: [np.mean([v in s for s in sets]) for v in names] for rung, sets in sel.items()},
                             index=pd.Index(names, name="variable"))  # fmt: skip
    if len(contributions):
        contributions["selected"] = contributions["variable"].map(selection[EQUATION])
        order = contributions[contributions.scheme == "reference"].sort_values(
            "contribution", ascending=False
        )
        contributions["variable"] = pd.Categorical(
            contributions["variable"], list(order["variable"])
        )
        contributions = contributions.sort_values(["variable", "scheme"]).reset_index(drop=True)
        contributions["variable"] = contributions["variable"].astype(str)
        variables = list(order["variable"])
    verdicts = pd.DataFrame(
        {
            name: [_verdict(sides[(v, "reference")], sides[(v, name)]) for v in variables]
            for name in shifts
        },
        index=pd.Index(variables, name="variable"),
    )
    block_table = pd.DataFrame(
        block_rows, columns=["block", "scheme", "contribution", "lo", "hi", "units helped"]
    )
    block_verdicts = pd.DataFrame(
        {
            name: [_verdict(block_sides[(b, "reference")], block_sides[(b, name)]) for b in blocks]
            for name in shifts
        },
        index=pd.Index(list(blocks), name="block"),
    )
    # memorized: reference skill of the reference-chosen equation minus that of the shift-chosen one
    unit_idx, Lref, base = reference_boot
    mem = []
    for name in shifts:
        rung = f"{EQUATION}, chosen for {name}"
        if rung in Lref:
            p, lo, hi, _ = _intervals(unit_idx, (Lref[rung] - Lref[EQUATION])[:, None], base, level)
            mem.append({"shift": name, "memorized": p[0], "lo": lo[0], "hi": hi[0]})
    memorized = (
        pd.DataFrame(mem).set_index("shift")
        if mem
        else pd.DataFrame(columns=["memorized", "lo", "hi"])
    )

    # the equations on all rows
    final = {EQUATION: clone(proto).set_params(select_on=None).fit(frame, y, sample_weight=w)}
    for name, s in shifts.items():
        try:
            s.subset(np.arange(n))
        except NotImplementedError:
            continue
        final[f"{EQUATION}, chosen for {name}"] = (
            clone(proto).set_params(select_on=s).fit(frame, y, sample_weight=w)
        )
    return Ledger(
        skill=skill, contributions=contributions, verdicts=verdicts, blocks=block_table,
        block_verdicts=block_verdicts, memorized=memorized,
        selection=selection, distance=distance,
        distance_trend=pd.DataFrame(trend_rows, columns=["scheme", "rung", "rank correlation"]),
        models=final, predictions=predictions, family=fam.name, level=level, n_splits=n_splits,
        n_repeats=n_repeats, schemes=list(schemes),
    )  # fmt: skip
