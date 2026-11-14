"""The method against problems with known answers."""

import pytest

from transferledger.truth import PROBLEMS, scorecard


def test_every_problem_runs_once():
    card = scorecard(seeds=[0], problems=[p for p in PROBLEMS if "yes/no" not in p.name])
    assert set(card.columns) == {"problem", "fact", "passed", "floor", "ok"} and len(card) >= 10


@pytest.mark.slow
def test_known_answers_are_recovered_at_their_floors():
    card = scorecard(seeds=range(8))
    failed = card[~card.ok]
    assert failed.empty, "\n" + failed.to_string(index=False)
