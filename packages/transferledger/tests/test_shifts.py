"""Shifts produce splits in which the test rows are new in the named sense."""

import numpy as np
import pytest

from transferledger import BeyondRange, ForwardInTime, NewGroups, Reference


def test_new_groups_share_no_group_between_train_and_test():
    g = np.repeat(np.arange(20), 5)
    s = NewGroups(g, "new site")
    for seed in (0, 1):
        for tr, te in s.splits(5, seed):
            assert not set(g[tr]) & set(g[te])
    assert s.splits(5, 0)[0][1].tolist() != s.splits(5, 1)[0][1].tolist() and s.repeatable
    inner = s.subset(s.splits(5, 0)[0][0])
    assert len(inner) == 80 and len(np.unique(inner.units())) == 16


def test_few_groups_are_left_out_one_at_a_time():
    g = np.repeat(["a", "b", "c"], 10)
    s = NewGroups(g)
    folds = s.splits(5, 0)
    assert len(folds) == 3 and not s.repeatable
    assert sorted(len(te) for _, te in folds) == [10, 10, 10]


def test_forward_in_time_never_trains_on_the_future():
    t = np.random.default_rng(0).permutation(100)
    for tr, te in ForwardInTime(t).splits(4):
        assert t[tr].max() < t[te].min()


def test_beyond_range_tests_outside_the_training_values():
    v = np.random.default_rng(1).normal(size=200)
    ((tr, te),) = BeyondRange(v, "high", 0.2).splits()
    assert v[tr].max() < v[te].min() and len(te) == 40
    assert len(BeyondRange(v, "both").splits()) == 2
    with pytest.raises(ValueError):
        BeyondRange(np.ones(10)).splits()


def test_reference_is_random_folds():
    folds = Reference(50).splits(5, 0)
    assert len(folds) == 5 and sorted(np.concatenate([te for _, te in folds])) == list(range(50))
    with pytest.raises(ValueError):
        NewGroups(np.zeros(5))
