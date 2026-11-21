"""The detection floor: planted properties of known strength, and how often they are found."""

import numpy as np
import pandas as pd
import pytest

from transferledger import NewGroups, detectable
from transferledger.detection import PLANTED, _plant


def table(n_groups=18, per=4, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    offset = rng.normal(size=n_groups)
    X = pd.DataFrame({"x": rng.normal(size=len(g)), "z": rng.normal(size=n_groups)[g]})
    y = X.x + offset[g] + 0.3 * rng.normal(size=len(g))
    return X, y.to_numpy(), g


def test_planted_property_has_the_stated_share():
    rng = np.random.default_rng(0)
    means = rng.normal(size=4000)
    for share in (0.2, 0.7):
        v = _plant(means, share, "trend", rng)
        assert abs(np.corrcoef(v, means)[0, 1] ** 2 - share) < 0.03
    # a peak: the target is highest where the property is near zero, on either side
    p = _plant(means, 0.9, "peak", rng)
    assert abs(np.corrcoef(p, means)[0, 1]) < 0.1 and np.corrcoef(np.abs(p), means)[0, 1] < -0.9


def test_strong_properties_are_found_and_absent_ones_are_not():
    X, y, g = table()
    D = detectable(X, y, NewGroups(g, "new site"), shapes=("trend",), sizes=[9, 18],
                   shares=(0.3, 0.95), n_trials=8)  # fmt: skip
    p = D.power.set_index(["units", "share"])["detected"]
    assert p.loc[(18, 0.95)] >= 0.8 and p.loc[(18, 0.95)] >= p.loc[(9, 0.3)]
    assert (D.false_alarms["rate"] <= 0.25).all()
    assert D.unit == "site" and D.n_units == 18
    # the fitted law: more units and a larger share both raise the detection rate
    assert D.rate(36, 0.5) > D.rate(18, 0.5) > D.rate(18, 0.2)
    floor = D.floor()
    assert floor is None or 0.02 <= floor <= 0.98
    strong, weak = D.needed(0.95), D.needed(0.3)
    assert strong is not None and (weak is None or weak >= strong)
    assert "site" in str(D) and set(D.to_dict()) >= {"power", "law", "false_alarms"}


def test_bad_input():
    X, y, g = table()
    with pytest.raises(TypeError, match="NewGroups"):
        detectable(X, y, None)
    with pytest.raises(ValueError, match="already has a column"):
        detectable(X.assign(**{PLANTED: 0.0}), y, NewGroups(g))
    with pytest.raises(ValueError, match="too few"):
        detectable(X.iloc[:16], y[:16], NewGroups(g[:16]))
