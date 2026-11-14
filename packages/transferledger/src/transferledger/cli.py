"""Command line: a pre-check, an equation or a full ledger from a CSV file in one call.

    transferledger preflight data.csv --target yield --group ligand
    transferledger fit       data.csv --target yield --select-on group:ligand --save equation.json
    transferledger ledger    data.csv --target yield --group ligand --group base --report ledger.html
    transferledger predict   equation.json new_rows.csv --out predictions.csv
    transferledger truth

Columns that define a shift (a group label, a time) are not used as predictors unless named
with ``--keep``.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .ledger import transfer_ledger
from .model import PieceModel
from .preflight import preflight
from .shifts import BeyondRange, ForwardInTime, NewGroups

__all__ = ["main"]


def _read(path):
    if str(path).endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    sep = "\t" if str(path).endswith((".tsv", ".tab")) else ","
    return pd.read_csv(path, sep=sep)


def _column(df, name):
    if name not in df.columns:
        raise SystemExit(
            f"no column {name!r} in the data; columns are: {', '.join(map(str, df.columns))}"
        )
    return df[name]


def _shift(df, spec):
    """``group:COL``, ``time:COL`` or ``beyond:COL[:high|low|both]`` -> (shift, column)."""
    kind, _, rest = spec.partition(":")
    if kind == "group":
        return NewGroups(_column(df, rest).to_numpy(), f"new {rest}"), rest
    if kind == "time":
        return ForwardInTime(_column(df, rest).to_numpy(), f"later {rest}"), rest
    if kind == "beyond":
        col, _, side = rest.partition(":")
        side = side or "high"
        word = {"high": "higher", "low": "lower", "both": "beyond the range of"}[side]
        return BeyondRange(_column(df, col).to_numpy(float), side, name=f"{word} {col}"), None
    raise SystemExit(
        f"cannot read the shift {spec!r}; use group:COL, time:COL or beyond:COL[:high|low|both]"
    )


def _prepare(args):
    df = _read(args.data)
    y = _column(df, args.target).to_numpy(float)
    specs = [f"group:{g}" for g in getattr(args, "group", None) or []]
    specs += [f"time:{t}" for t in getattr(args, "time", None) or []]
    specs += [f"beyond:{b}" for b in getattr(args, "beyond", None) or []]
    if getattr(args, "select_on", None):
        specs = [args.select_on]
    shifts, shift_columns = {}, []
    for spec in specs:
        s, col = _shift(df, spec)
        shifts[s.name] = s
        if col is not None:
            shift_columns.append(col)
    weight = _column(df, args.weight).to_numpy(float) if getattr(args, "weight", None) else None
    drop = {args.target, *(getattr(args, "drop", None) or [])}
    if getattr(args, "weight", None):
        drop.add(args.weight)
    drop |= {c for c in shift_columns if c not in (getattr(args, "keep", None) or [])}
    X = df.drop(columns=[c for c in drop if c in df.columns])
    clusters = getattr(args, "clusters", None)
    if clusters:
        num = X.select_dtypes(include=[np.number, bool])
        s = NewGroups.from_clusters(num.to_numpy(float), clusters, "unlike anything seen")
        shifts[s.name] = s
    return X, y, shifts, weight


def _forms(items):
    out = {}
    for item in items or []:
        name, _, forms = item.partition("=")
        out[name] = [f.strip() for f in forms.split(",") if f.strip()]
    return out or None


def _model(args, select_on=None):
    return PieceModel(
        family=args.family, forms=_forms(getattr(args, "forms", None)),
        products=not getattr(args, "no_products", False), select_on=select_on,
    )  # fmt: skip


def cmd_preflight(args):
    X, y, shifts, _ = _prepare(args)
    print(preflight(X, y, shifts))
    return 0


def cmd_fit(args):
    X, y, shifts, weight = _prepare(args)
    select_on = next(iter(shifts.values())) if shifts else None
    model = _model(args, select_on).fit(X, y, sample_weight=weight)
    how = f"chosen for: {select_on.name}" if select_on is not None else "chosen on random folds"
    print(
        f"{len(X)} rows, {X.shape[1]} predictor columns; {how}; held-out skill {model.skill_:.3f}\n"
    )
    print(model.equation())
    if args.latex:
        print("\n" + model.to_latex())
    if args.save:
        model.save(args.save)
        print(f"\nequation written to {args.save}")
    if args.python:
        with open(args.python, "w", encoding="utf-8") as f:
            f.write(model.to_python())
        print(f"Python source written to {args.python}")
    return 0


def cmd_ledger(args):
    X, y, shifts, weight = _prepare(args)
    if not shifts:
        raise SystemExit(
            "a ledger needs at least one shift: --group COL, --time COL, --beyond COL or --clusters K"
        )
    blocks = {}
    for item in args.block or []:
        name, _, cols = item.partition("=")
        blocks[name] = [c.strip() for c in cols.split(",") if c.strip()]
    ledger = transfer_ledger(
        X, y, shifts, model=_model(args), family=args.family, sample_weight=weight, blocks=blocks,
        n_repeats=args.repeats, n_jobs=args.jobs, random_state=args.seed,
    )  # fmt: skip
    print(ledger.preflight)
    print()
    print(ledger)
    for name, m in ledger.models.items():
        print(f"\n{name}:\n{m.equation()}")
    if args.report:
        ledger.report(args.report, title=args.title or f"Transfer ledger: {args.target}")
        print(f"\nreport written to {args.report}")
    if args.json:
        ledger.to_json(args.json)
        print(f"tables written to {args.json}")
    return 0


def cmd_predict(args):
    model = PieceModel.load(args.equation)
    df = _read(args.data)
    out = df.copy()
    out["prediction"] = model.predict(df[list(model.feature_names_in_)])
    support = model.support(df[list(model.feature_names_in_)])
    out["outside_fitted_range"] = support["variables"].to_numpy()
    out["unseen_level"] = support["unseen_level"].to_numpy()
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"{len(out)} predictions written to {args.out}")
    else:
        out.to_csv(sys.stdout, index=False)
    n_out = int((support["outside"] > 0).sum())
    if n_out:
        print(
            f"{n_out} of {len(out)} rows lie outside the range the equation was fitted on",
            file=sys.stderr,
        )
    return 0


def cmd_truth(args):
    from .truth import scorecard

    card = scorecard(range(args.seeds))
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(card.to_string(index=False))
    failed = int((~card.ok).sum())
    print(
        f"\n{len(card) - failed} of {len(card)} facts recovered at or above their floor over {args.seeds} seeds"
    )
    return 1 if failed else 0


def _common(p, shifts=True):
    p.add_argument("data", help="CSV, TSV or Parquet file")
    p.add_argument("--target", required=True, help="the column to predict")
    p.add_argument(
        "--drop", action="append", help="a column to leave out of the predictors (repeatable)"
    )
    p.add_argument(
        "--keep", action="append", help="keep a shift's column as a predictor (repeatable)"
    )
    p.add_argument("--weight", help="a column of row weights")
    if shifts:
        p.add_argument(
            "--group", action="append", help="hold out whole groups of this column (repeatable)"
        )
        p.add_argument(
            "--time", action="append", help="train on the past of this column, test on the future"
        )
        p.add_argument(
            "--beyond",
            action="append",
            help="COL[:high|low|both]: test beyond the range of a column",
        )
        p.add_argument("--clusters", type=int, help="hold out K clusters of the numeric predictors")


def _model_args(p):
    p.add_argument("--family", default="gaussian", choices=["gaussian", "binomial", "poisson"])
    p.add_argument(
        "--forms", action="append", help="VAR=form[,form]: the forms to try for a variable"
    )
    p.add_argument("--no-products", action="store_true", help="do not admit products of pieces")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="transferledger",
        description="Readable equations from a table, and which parts of them hold under a named shift.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", help="what the data can support, before fitting")
    _common(p)
    p.set_defaults(run=cmd_preflight)

    p = sub.add_parser("fit", help="fit an equation and print it")
    _common(p, shifts=False)
    _model_args(p)
    p.add_argument(
        "--select-on",
        help="group:COL, time:COL or beyond:COL[:side]: choose variables under this shift",
    )
    p.add_argument("--save", help="write the equation to a JSON file")
    p.add_argument("--python", help="write numpy-only Python source of the equation")
    p.add_argument("--latex", action="store_true", help="also print the equation as LaTeX")
    p.set_defaults(run=cmd_fit)

    p = sub.add_parser("ledger", help="score the equation and each variable under each shift")
    _common(p)
    _model_args(p)
    p.add_argument(
        "--block", action="append", help="NAME=col,col: variables to also remove together"
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--jobs", type=int, default=None, help="parallel workers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", help="write a self-contained HTML report")
    p.add_argument("--json", help="write every table as JSON")
    p.add_argument("--title", help="title of the report")
    p.set_defaults(run=cmd_ledger)

    p = sub.add_parser("predict", help="apply a saved equation to new rows")
    p.add_argument("equation", help="JSON file written by 'fit --save'")
    p.add_argument("data", help="CSV, TSV or Parquet file with the equation's columns")
    p.add_argument("--out", help="write predictions here instead of standard output")
    p.set_defaults(run=cmd_predict)

    p = sub.add_parser("truth", help="run the known-answer problems and print the scorecard")
    p.add_argument("--seeds", type=int, default=10)
    p.set_defaults(run=cmd_truth)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
