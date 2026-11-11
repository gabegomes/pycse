"""The ledger: skills, intervals, verdicts, blocks, memorized share."""

import numpy as np
import pandas as pd
import pytest

from transferledger import BeyondRange, NewGroups, preflight, transfer_ledger
from transferledger.ledger import _intervals


def grouped_data(n_groups=40, per=12, seed=0):
    """``x`` (per row) and ``z`` (a group property) carry real effects. Each group also has an
    offset of its own, unrelated to ``z``; ``label`` is a group property made of noise, which a
    flexible curve can use to look that offset up on rows of groups it has seen."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    x = rng.normal(size=len(g))
    z = rng.normal(size=n_groups)
    label = rng.normal(size=n_groups)
    offset = rng.normal(size=n_groups)
    offset = offset - np.polyval(np.polyfit(z, offset, 1), z)  # unrelated to z in this sample
    y = 1.5 * x + 1.5 * z[g] + 1.2 * offset[g] + 0.2 * rng.normal(size=len(g))
    X = pd.DataFrame({"x": x, "z": z[g], "label": label[g]})
    return X, y, g


@pytest.fixture(scope="module")
def ledger():
    X, y, g = grouped_data()
    return transfer_ledger(X, y, NewGroups(g, "new group"), blocks={"group properties": ["z", "label"]},
                           n_repeats=2)  # fmt: skip


def test_tables_have_every_rung_and_scheme(ledger):
    assert ledger.schemes == ["reference", "new group"]
    lad = ledger.ladder
    assert lad.index[0].startswith("baseline") and "equation, chosen for new group" in lad.index
    assert np.allclose(lad.iloc[0], 0.0, atol=1e-9)
    assert (ledger.skill.lo <= ledger.skill.skill + 1e-12).all() and (
        ledger.skill.skill <= ledger.skill.hi + 1e-12
    ).all()
    assert "Verdicts" in str(ledger) and list(ledger.memorized.index) == ["new group"]


def test_real_effects_transfer(ledger):
    v = ledger.verdicts["new group"]
    assert v["x"] == "transfers" and v["z"] == "transfers"
    assert ledger.ladder.loc["equation", "new group"] > 0.5


def test_memorizing_shows_up_and_choosing_under_the_shift_removes_it(ledger):
    lad = ledger.ladder
    # tuned on rows, the curves look the group offsets up through ``label``: best on the
    # reference, and they lose the most on new groups
    assert lad.loc["curves", "reference"] > lad.loc["equation", "reference"]
    assert lad.loc["curves", "reference"] - lad.loc["curves", "new group"] > 0.15
    assert (
        lad.loc["curves, tuned for new group", "new group"] > lad.loc["curves", "new group"] + 0.05
    )
    # the stand-in for group identity is not shown to transfer, and the equation chosen for
    # new groups leaves it out, at a cost on the reference: the memorized share
    assert ledger.verdicts.loc["label", "new group"] != "transfers"
    assert "label" not in ledger.models["equation, chosen for new group"].variables_
    assert ledger.memorized.loc["new group", "memorized"] > 0.02


def test_blocks_are_accounted_together(ledger):
    assert list(ledger.blocks.block.unique()) == ["group properties"]
    assert ledger.block_verdicts.shape == (1, 1)


def test_intervals_widen_with_fewer_units():
    rng = np.random.default_rng(0)
    cols, base = rng.normal(1.0, 1.0, size=(400, 1)), np.ones(400)
    _, lo_many, hi_many, _ = _intervals(np.arange(400) % 100, cols, base, 0.95)
    _, lo_few, hi_few, share = _intervals(np.arange(400) % 4, cols, base, 0.95)
    assert share[0] == 1.0
    assert (hi_few - lo_few)[0] > 1.3 * (hi_many - lo_many)[0]


def test_same_seed_same_ledger_and_extra_models_are_scored():
    from sklearn.linear_model import LinearRegression

    X, y, g = grouped_data(n_groups=12, per=10, seed=1)
    kw = dict(models={"linear": LinearRegression()}, n_repeats=1)
    a = transfer_ledger(X, y, NewGroups(g, "new group"), **kw)
    b = transfer_ledger(X, y, NewGroups(g, "new group"), **kw)
    pd.testing.assert_frame_equal(a.skill, b.skill)
    assert "linear" in a.ladder.index


def test_several_shifts_and_extrapolation_in_a_variable():
    X, y, g = grouped_data(n_groups=10, per=20, seed=2)
    L = transfer_ledger(X, y, {"new group": NewGroups(g, "new group"),
                               "larger x": BeyondRange(X.x, "high", 0.2, "larger x")}, n_repeats=1)  # fmt: skip
    assert L.schemes == ["reference", "new group", "larger x"]
    assert set(L.verdicts.columns) == {"new group", "larger x"}
    assert L.predictions["larger x"].notna().all(axis=1).mean() == pytest.approx(0.2, abs=0.01)


def test_binary_target():
    X, y, g = grouped_data(n_groups=20, per=10, seed=3)
    yb = (y > np.median(y)).astype(float)
    L = transfer_ledger(X, yb, NewGroups(g, "new group"), family="binomial", n_repeats=1)
    assert L.ladder.loc["equation", "reference"] > 0.2


def test_preflight_reports_group_properties_few_groups_and_duplicates():
    X, y, g = grouped_data(n_groups=5, per=20)
    X["x_copy"] = 2.0 * X.x + 1e-3
    pre = preflight(X, y, NewGroups(g, "new group"))
    assert pre.variables.loc["z", "values across groups (new group)"] == 5
    assert pd.isna(pre.variables.loc["x", "values across groups (new group)"])
    assert pre.shifts.loc["new group", "design"] == "each group left out once"
    assert len(pre.duplicates) == 1
    text = "\n".join(pre.warnings)
    assert "only 5 groups" in text and "rests on 5 points" in text and "near-duplicate" in text
