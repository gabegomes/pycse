"""Symbolic search over pieces, and safe evaluation of the expressions it returns.

PySR (optional) searches for an expression of the equation's pieces, using addition,
subtraction and multiplication only, that explains what the sum of pieces leaves over. It
never sees a raw variable: every symbol in its expression is a named piece.
"""

from __future__ import annotations

import ast

import numpy as np

__all__ = ["pysr_regressor", "knee", "evaluate"]


def pysr_regressor(**options):
    """A PySRRegressor restricted to ``+ - *``. Needs ``pip install "transferledger[pysr]"``
    (which installs Julia on first use)."""
    try:
        from pysr import PySRRegressor
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError('products="pysr" needs PySR: pip install "transferledger[pysr]"') from e
    opts = dict(
        niterations=60, maxsize=15, binary_operators=["+", "-", "*"], unary_operators=[],
        model_selection="best", populations=12, population_size=40, random_state=0,
        deterministic=True, parallelism="serial", progress=False, verbosity=0,
        temp_equation_file=True, delete_tempfiles=True,
    )  # fmt: skip
    opts.update(options)
    return PySRRegressor(**opts)


def knee(regressor, y_var, share=0.7, min_gain=0.01):
    """The equation to keep from a PySR front fitted to a residual, or None.

    The gain of an equation is the fraction of ``y_var`` it removes beyond the constant at the
    top of the front. Nothing is kept unless the best gain reaches ``min_gain``; then the
    smallest equation reaching ``share`` of the best gain is used. Past it, each extra node
    buys little and costs readability.
    """
    eq = regressor.equations_
    loss = eq["loss"].to_numpy()
    gain = (loss[0] - loss) / y_var
    if np.nanmax(gain) < min_gain:
        return None
    return int(eq.index[gain >= share * np.nanmax(gain)][0])


def evaluate(expression, values):
    """Evaluate an arithmetic expression of named columns.

    Only numbers, the names in ``values``, ``+ - *`` and whole-number powers are accepted, so
    an expression read from a file cannot do anything but arithmetic.
    """

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                return ev(node.left) + ev(node.right)
            if isinstance(node.op, ast.Sub):
                return ev(node.left) - ev(node.right)
            if isinstance(node.op, ast.Mult):
                return ev(node.left) * ev(node.right)
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
                k = node.right.value
                if isinstance(k, int) and 0 <= k <= 8:
                    return ev(node.left) ** k
        raise ValueError(f"not an arithmetic expression of pieces: {expression!r}")

    return ev(ast.parse(expression, mode="eval"))
