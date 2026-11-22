"""unit_ceiling: is there anything about a unit to explain?"""

import numpy as np
import pandas as pd
import pytest

from transferledger import NewGroups, unit_ceiling


def _table(effect, described, n_units=24, per=12, seed=0):
    """Units with a level of their own (``effect`` standard deviations) that a descriptor
    reproduces (``described``) or not."""
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), per)
    level = rng.normal(size=n_units)
    x = rng.normal(size=len(unit))
    desc = level + 0.05 * rng.normal(size=n_units) if described else rng.normal(size=n_units)
    y = 1.5 * x + effect * level[unit] + 0.3 * rng.normal(size=len(unit))
    return pd.DataFrame({"x": x, "descriptor": desc[unit]}), y, unit


def test_units_that_do_not_differ():
    X, y, unit = _table(effect=0.0, described=True)
    c = unit_ceiling(X, y, NewGroups(unit, "new site"))
    assert c.state() == "nothing to explain"
    assert c.captured() is None
    assert "No description of a site can be tested" in str(c)


def test_described_units():
    X, y, unit = _table(effect=1.0, described=True)
    c = unit_ceiling(X, y, NewGroups(unit, "new site"))
    assert c.state() == "units differ"
    assert c.described_by == ["descriptor"]
    assert c.ceiling().contribution > 0.2
    assert c.captured() > 0.8


def test_units_differ_but_the_description_misses_it():
    X, y, unit = _table(effect=1.0, described=False)
    c = unit_ceiling(X, y, NewGroups(unit, "new site"))
    assert c.state() == "units differ"
    assert c.captured() < 0.5
    assert set(c.skill.kind) == {"additive", "any"}


def test_yes_no_target_and_no_description():
    X, y, unit = _table(effect=1.5, described=True)
    yes = (y > np.median(y)).astype(float)
    c = unit_ceiling(X[["x"]], yes, NewGroups(unit, "new site"), family="binomial")
    assert c.state() == "units differ"
    assert c.described_by == [] and c.captured() is None


def test_needs_groups():
    X, y, unit = _table(effect=1.0, described=True)
    with pytest.raises(TypeError):
        unit_ceiling(X, y, None)
