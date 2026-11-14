"""Form matching: known curves come back with the right form and location."""

import numpy as np
import pytest

from transferledger.forms import FORMS, Form, Piece, Scale, match_curve, register_form, sigmoid

RNG = np.random.default_rng(0)
V = RNG.uniform(22, 50, 600)


@pytest.mark.parametrize(
    "f, form, location",
    [
        (-1.5 * sigmoid(V - 32.0), "step", 32.0),
        (1.5 * V, "linear", None),
        (-0.01 * (V - 35.0) ** 2, "quadratic", 35.0),
        (np.abs(V - 33.0) + 0.3 * V, "vee", 33.0),
        (np.exp(-0.25 * (V - 22.0)), "exponential", None),
    ],
)
def test_known_curves_are_recovered(f, form, location):
    piece = match_curve("v", V, f)[0]
    assert piece.form == form and piece.r2 > 0.999
    if location is not None:
        assert piece.location == pytest.approx(location, abs=0.05)


def test_window_edges():
    f = sigmoid((V - 27) / 0.6) - sigmoid((V - 38) / 0.6)
    piece = match_curve("v", V, f)[0]
    assert piece.form == "window"
    assert piece.location == pytest.approx((27.0, 38.0), abs=0.05)


def test_raw_value_forms_are_opt_in():
    T = RNG.uniform(288, 368, 300)
    assert match_curve("T", T, -3300.0 / T)[0].form != "inverse"
    piece = match_curve("T", T, -3300.0 / T, forms=["inverse"])[0]
    assert piece.form == "inverse" and piece.r2 > 0.9999


def test_an_edge_needs_support_on_both_sides():
    """A jump carried by two of twelve values is a lookup, not a step."""
    x = np.repeat(np.arange(12.0), 10)
    f = np.where(x >= 9.5, 1.0, 0.0)
    assert match_curve("x", x, f, min_side=1)[0].form == "step"
    strict = match_curve("x", x, f, min_side=3)[0]
    assert strict.form not in ("step", "exponential") and strict.r2 < 0.95


def test_two_values_make_a_flag_and_a_vee_brings_its_line():
    flag = match_curve("f", np.repeat([0.0, 1.0], 20), np.repeat([0.0, 3.0], 20))
    assert [p.form for p in flag] == ["flag"]
    assert set(np.unique(flag[0].unit([0, 1, 1]))) == {0.0, 1.0}
    vee = match_curve("v", V, np.abs(V - 33.0) + 0.3 * V)
    assert [p.form for p in vee] == ["vee", "linear"] and vee[1].companion_of == "vee"


def test_pieces_clip_to_their_range_and_say_so():
    p = Piece("x", "linear", np.zeros(0), Scale(0.0, 1.0, -1.0, 1.0))
    assert p.unit(5.0) == pytest.approx(p.unit(1.0))
    assert list(p.outside([0.0, 5.0, -2.0])) == [False, True, True]


def test_a_registered_form_is_used():
    hill = Form(
        "hill2",
        shape=lambda z, th, sc: sc.x(z) ** 2 / (th[0] ** 2 + sc.x(z) ** 2),
        grid=lambda u, sc: np.geomspace(0.05 * sc.hi, 2 * sc.hi, 40)[:, None],
        valid=lambda th, u, sc, min_side: th[0] > 0,
        text=lambda n, th, sc: f"{n}^2 / ({th[0]:.3g}^2 + {n}^2)",
        location=lambda th, sc: float(th[0]),
        n_theta=1,
        positive_only=True,
    )
    register_form(hill)
    try:
        x = RNG.uniform(0.1, 10, 300)
        piece = match_curve("c", x, x**2 / (9.0 + x**2), forms=["hill2"])[0]
        assert piece.form == "hill2" and piece.location == pytest.approx(3.0, abs=0.05)
    finally:
        FORMS.pop("hill2")


def test_values_that_differ_in_the_fourth_decimal_are_one_value():
    """Nine values within 0.003 of each other and five spread out: a kink among the nine would
    have three values on each side by count, and be a lookup of them."""
    rng = np.random.default_rng(0)
    levels = np.r_[-0.694 + 0.0003 * np.arange(9), 0.94, 1.07, 1.47, 1.63, 1.83]
    x = np.repeat(levels, 20)
    f = np.where(x < -0.6935, 3.0, 0.0) + 0.5 * x + 0.01 * rng.normal(size=len(x))
    pieces = match_curve("x", x, f)
    for p in pieces:
        for edge in np.atleast_1d(p.location if p.location is not None else []):
            assert not (-0.6945 < edge < -0.6915), p.text()
