"""Taking an equation out of the package: a dictionary (and JSON) that round-trips, LaTeX,
SymPy, and plain Python source with no dependency on this package.

The dictionary form is the durable one: it holds everything needed to predict (the pieces, the
weights, how text columns were encoded, the range each piece was fitted on), so an equation
fitted today can be reloaded and applied later without the data.
"""

from __future__ import annotations

import json

import numpy as np

from ._data import Design
from .curves import get_family
from .forms import FORMS, Piece, Scale

__all__ = ["to_dict", "from_dict", "to_latex", "to_sympy", "to_python"]

FORMAT = 1


# -----------------------------------------------------------------------------
# dictionary / JSON
# -----------------------------------------------------------------------------
def _piece_dict(p):
    return {
        "variable": p.feature, "group": p.group, "column": p.column, "form": p.form,
        "theta": [float(v) for v in p.theta],
        "scale": {"mean": p.scale.mean, "sd": p.scale.sd, "lo": p.scale.lo, "hi": p.scale.hi},
        "extrapolate": bool(p.extrapolate),
    }  # fmt: skip


def to_dict(model):
    """A JSON-serializable description of a fitted :class:`~transferledger.model.PieceModel`."""
    d = model.design_
    index = {j: k for k, j in enumerate(model.chosen_)}
    return {
        "format": FORMAT,
        "family": model.family,
        "intercept": float(model.intercept_),
        "weights": [float(c) for c in model.coef_],
        "pieces": [_piece_dict(model.pieces_[j]) for j in model.chosen_],
        "products": [[index[a], index[b]] for a, b in model.products_],
        "expression": getattr(model, "expression_", None),
        "limit": model.limit,
        "target_range": list(model.target_range_),
        "design": {
            "feature_names_in": list(d.feature_names_in_),
            "names": list(d.names_),
            "source": list(d.source_),
            "levels": d.levels_,
            "reference": d.reference_,
        },  # fmt: skip
        "equation": model.equation(),
    }


def from_dict(d):
    """Rebuild a predicting :class:`~transferledger.model.PieceModel` from :func:`to_dict`.

    The result predicts, writes its equation and reports ``support`` (without the distance to
    the training rows, which are not stored). It cannot be refitted without data.
    """
    from .model import PieceModel

    if d.get("format") != FORMAT:
        raise ValueError(f"unknown equation format {d.get('format')!r}")
    for p in d["pieces"]:
        if p["form"] != "flag" and p["form"] not in FORMS:
            raise ValueError(f"the equation uses form {p['form']!r}; register it before loading")
    m = PieceModel(family=d["family"], limit=d["limit"])
    design = Design()
    dd = d["design"]
    design.feature_names_in_ = list(dd["feature_names_in"])
    design.names_, design.source_ = list(dd["names"]), list(dd["source"])
    design.levels_, design.reference_ = dict(dd["levels"]), dict(dd["reference"])
    design.n_features_in_ = len(design.feature_names_in_)
    design.feature_names = list(design.feature_names_in_)
    m.design_ = design
    m.pieces_ = [
        Piece(p["variable"], p["form"], np.asarray(p["theta"], float), Scale(**p["scale"]),
              column=p["column"], group=p["group"], extrapolate=p["extrapolate"])
        for p in d["pieces"]
    ]  # fmt: skip
    m.chosen_ = list(range(len(m.pieces_)))
    m.products_ = [tuple(pair) for pair in d["products"]]
    m.expression_ = d.get("expression")
    m.intercept_, m.coef_ = float(d["intercept"]), np.asarray(d["weights"], float)
    m.target_range_ = tuple(d["target_range"])
    m._used = sorted({p.column for p in m.pieces_})
    m._train = None
    m.n_features_in_ = design.n_features_in_
    m.feature_names_in_ = np.asarray(design.feature_names_in_, dtype=object)
    return m


def save(model, path):
    with open(path, "w") as f:
        json.dump(to_dict(model), f, indent=1)


def load(path):
    with open(path) as f:
        return from_dict(json.load(f))


# -----------------------------------------------------------------------------
# LaTeX
# -----------------------------------------------------------------------------
def _g(v, digits=3):
    s = f"{v:.{digits}g}"
    if "e" in s:
        mant, exp = s.split("e")
        return f"{mant} \\times 10^{{{int(exp)}}}"
    return s


def _name(n):
    return (
        "\\mathrm{"
        + str(n).replace("\\", "").replace("_", "\\_").replace("%", "\\%").replace(" ", "\\ ")
        + "}"
    )


def _minus(x, v):
    return f"{x} - {_g(v)}" if v >= 0 else f"{x} + {_g(-v)}"


def _piece_latex(p):
    x, th, sc = _name(p.feature), p.theta, p.scale
    f = p.form
    if f == "flag":
        if "=" in p.feature:
            col, level = p.feature.split("=", 1)
            return f"[{_name(col)} = \\text{{{level}}}]"
        return f"[{x} = {_g(sc.hi)}]"
    if f == "linear":
        return f"\\frac{{{_minus(x, sc.mean)}}}{{{_g(sc.sd)}}}"
    if f == "inverse":
        return f"\\frac{{{_g(sc.mean)}}}{{{x}}}"
    if f == "log":
        return f"\\ln\\frac{{{x}}}{{{_g(sc.mean)}}}"
    if f == "quadratic":
        return f"\\left(\\frac{{{_minus(x, sc.x(th[0]))}}}{{{_g(sc.sd)}}}\\right)^2"
    if f == "exponential":
        ref = sc.zhi if th[0] > 0 else sc.zlo
        return f"\\exp\\left[{_g(th[0] / sc.sd)}\\,({_minus(x, sc.x(ref))})\\right]"
    if f == "saturation":
        return f"\\frac{{{x}}}{{{_g(th[0])} + {x}}}"
    if f == "vee":
        return f"\\frac{{|{_minus(x, sc.x(th[0]))}|}}{{{_g(sc.sd)}}}"
    if f == "step":
        return f"S\\!\\left(\\frac{{{_minus(x, sc.x(th[0]))}}}{{{_g(sc.sd * th[1])}}}\\right)"
    if f == "window":
        return f"W\\!\\left({x};\\ {_g(sc.x(th[0]))},\\ {_g(sc.x(th[1]))},\\ {_g(sc.sd * th[2])}\\right)"
    raise NotImplementedError(f"no LaTeX for the form {f!r}")


_LHS = {
    "gaussian": "y",
    "binomial": "\\operatorname{logit} P(y = 1)",
    "poisson": "\\ln \\mathbb{E}[y]",
}


def to_latex(model):
    """The equation as LaTeX: an ``align*`` body defining each piece, then the equation."""
    chosen = [model.pieces_[j] for j in model.chosen_]
    lines = [f"p_{{{k}}} &= {_piece_latex(p)} \\\\" for k, p in enumerate(chosen)]
    terms = [f"p_{{{k}}}" for k in range(len(chosen))] + [f"p_{{{a}}}\\,p_{{{b}}}" for a, b in
             (tuple(model.chosen_.index(i) for i in pair) for pair in model.products_)]  # fmt: skip
    expression = getattr(model, "expression_", None)
    if expression is not None:
        terms.append("g")
        tex = expression.replace("**", "^").replace("*", "\\,")
        for k in reversed(range(len(chosen))):
            tex = tex.replace(f"p{k}", f"p_{{{k}}}")
        lines.append(f"g &= {tex} \\\\")
    rhs = _g(model.intercept_, 4)
    for c, t in zip(model.coef_, terms):
        rhs += f" {'+' if c >= 0 else '-'} {_g(abs(c), 4)}\\,{t}"
    lines.append(f"{_LHS[get_family(model.family).name]} &= {rhs}")
    notes = []
    if any(p.form in ("step", "window") for p in chosen):
        notes.append("S(t) = 1 / (1 + e^{-t})")
    if any(p.form == "window" for p in chosen):
        notes.append("W(x; a, b, w) = S((x - a)/w) - S((x - b)/w)")
    body = "\n".join(lines)
    return body + ("\n% " + ", ".join(notes) if notes else "")


# -----------------------------------------------------------------------------
# SymPy
# -----------------------------------------------------------------------------
def to_sympy(model):
    """``(expression, symbols)``: the equation's value as a SymPy expression of the input
    variables. The limits that hold each piece at the edge of its fitted range are not part of
    the expression. Needs SymPy."""
    try:
        import sympy as sp
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("to_sympy needs SymPy: pip install sympy") from e
    syms = {}

    def sym(name):
        return syms.setdefault(name, sp.Symbol(name))

    def S(t):
        return 1 / (1 + sp.exp(-t))

    def piece(p):
        x, th, sc = sym(p.feature), [float(v) for v in p.theta], p.scale
        z = (x - sc.mean) / sc.sd
        f = p.form
        if f == "flag":
            return sp.Heaviside(x - 0.5 * (sc.lo + sc.hi))
        if f == "linear":
            return z
        if f == "inverse":
            return sc.mean / x
        if f == "log":
            return sp.log(x / sc.mean)
        if f == "quadratic":
            return (z - th[0]) ** 2
        if f == "exponential":
            return sp.exp(th[0] * (z - (sc.zhi if th[0] > 0 else sc.zlo)))
        if f == "saturation":
            return x / (th[0] + x)
        if f == "vee":
            return sp.Abs(z - th[0])
        if f == "step":
            return S((z - th[0]) / th[1])
        if f == "window":
            return S((z - th[0]) / th[2]) - S((z - th[1]) / th[2])
        raise NotImplementedError(f"no SymPy expression for the form {f!r}")

    exprs = [piece(model.pieces_[j]) for j in model.chosen_]
    index = {j: k for k, j in enumerate(model.chosen_)}
    cols = exprs + [exprs[index[a]] * exprs[index[b]] for a, b in model.products_]
    if getattr(model, "expression_", None) is not None:
        local = {f"p{k}": e for k, e in enumerate(exprs)}
        cols.append(sp.sympify(model.expression_, locals=local))
    expr = sp.Float(model.intercept_) + sum(float(c) * e for c, e in zip(model.coef_, cols))
    return expr, syms


# -----------------------------------------------------------------------------
# plain Python
# -----------------------------------------------------------------------------
def _piece_python(p, var):
    th, sc = [float(v) for v in p.theta], p.scale
    f = p.form
    if f == "flag":
        return f"({var} > {0.5 * (sc.lo + sc.hi)!r}).astype(float)"
    x = var if p.extrapolate else f"np.clip({var}, {sc.lo!r}, {sc.hi!r})"
    z = f"(({x}) - {sc.mean!r}) / {sc.sd!r}"
    if f == "linear":
        return z
    if f == "inverse":
        return f"{sc.mean!r} / ({x})"
    if f == "log":
        return f"np.log(({x}) / {sc.mean!r})"
    if f == "quadratic":
        return f"({z} - {th[0]!r}) ** 2"
    if f == "exponential":
        return f"np.exp({th[0]!r} * ({z} - {(sc.zhi if th[0] > 0 else sc.zlo)!r}))"
    if f == "saturation":
        return f"({x}) / ({th[0]!r} + ({x}))"
    if f == "vee":
        return f"np.abs({z} - {th[0]!r})"
    if f == "step":
        return f"_S(({z} - {th[0]!r}) / {th[1]!r})"
    if f == "window":
        return f"_S(({z} - {th[0]!r}) / {th[2]!r}) - _S(({z} - {th[1]!r}) / {th[2]!r})"
    raise NotImplementedError(f"no Python source for the form {f!r}")


def to_python(model, name="predict"):
    """Source of a function ``name(X)`` that reproduces ``model.predict`` with numpy alone.

    ``X`` is anything indexable by column name (a DataFrame, a dict of arrays).
    """
    d = model.design_
    lines = [
        "import numpy as np",
        "",
        "",
        "def _S(t):",
        "    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(t, float)))",
        "",
        "",
        f"def {name}(X):",
        f'    """{get_family(model.family).name} equation of {len(model.chosen_)} pieces; generated by transferledger."""',
    ]
    used = {}
    for j in model.chosen_:
        p = model.pieces_[j]
        col = d.names_[p.column]
        if col not in used:
            var = f"x{len(used)}"
            used[col] = var
            src = d.source_[p.column]
            if src in d.levels_:  # an indicator of one level of a text column
                level = col.split("=", 1)[1]
                lines.append(
                    f"    {var} = (np.asarray(X[{src!r}]).astype(str) == {level!r}).astype(float)"
                )
            else:
                lines.append(f"    {var} = np.asarray(X[{src!r}], float)")
    index = {j: k for k, j in enumerate(model.chosen_)}
    for k, j in enumerate(model.chosen_):
        p = model.pieces_[j]
        lines.append(f"    p{k} = {_piece_python(p, used[d.names_[p.column]])}")
    terms = [f"p{k}" for k in range(len(model.chosen_))] + [
        f"p{index[a]} * p{index[b]}" for a, b in model.products_
    ]
    if getattr(model, "expression_", None) is not None:
        lines.append(f"    g = {model.expression_}")
        terms.append("g")
    expr = repr(float(model.intercept_))
    for c, t in zip(model.coef_, terms):
        expr += f" + {float(c)!r} * {t}"
    lines.append(f"    value = {expr}")
    fam = get_family(model.family).name
    if fam == "binomial":
        lines.append("    return _S(value)")
    else:
        out = "np.exp(value)" if fam == "poisson" else "value"
        if model.limit is not None:
            lo, hi = model.target_range_
            pad = model.limit * (hi - lo)
            low = max(lo - pad, 0.0) if fam == "poisson" else lo - pad
            out = f"np.clip({out}, {low!r}, {hi + pad!r})"
        lines.append(f"    return {out}")
    return "\n".join(lines) + "\n"
