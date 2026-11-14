"""The command line, on a small CSV."""

import numpy as np
import pandas as pd
import pytest

from transferledger.cli import main
from transferledger.forms import sigmoid


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    rng = np.random.default_rng(0)
    site = np.repeat(np.arange(12), 20)
    df = pd.DataFrame({
        "site": [f"s{k}" for k in site], "dose": rng.uniform(0, 10, 240), "temp": rng.uniform(15, 35, 240),
        "altitude": rng.uniform(0, 2000, 12)[site], "operator": rng.choice(["ana", "ben"], 240),
    })  # fmt: skip
    df["response"] = (2.0 * sigmoid((df.dose - 6.0) / 0.4) - 0.08 * df.temp + 0.5 * (df.operator == "ben")
                      + 0.6 * rng.normal(size=12)[site] + 0.2 * rng.normal(size=240))  # fmt: skip
    path = tmp_path_factory.mktemp("cli") / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_preflight(csv, capsys):
    assert main(["preflight", str(csv), "--target", "response", "--group", "site"]) == 0
    out = capsys.readouterr().out
    assert "new site" in out and "altitude" in out


def test_fit_save_and_predict(csv, capsys, tmp_path):
    eq, src, preds = tmp_path / "eq.json", tmp_path / "eq.py", tmp_path / "preds.csv"
    assert main(["fit", str(csv), "--target", "response", "--select-on", "group:site", "--save", str(eq),
                 "--python", str(src), "--latex"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "chosen for: new site" in out and "dose" in out and "&=" in out
    assert "site" not in pd.read_json(eq, typ="series")["design"]["feature_names_in"]
    assert main(["predict", str(eq), str(csv), "--out", str(preds)]) == 0
    table = pd.read_csv(preds)
    assert {"prediction", "outside_fitted_range", "unseen_level"} <= set(table.columns)
    assert np.corrcoef(table.prediction, table.response)[0, 1] > 0.8


def test_ledger_with_report(csv, capsys, tmp_path):
    report, tables = tmp_path / "ledger.html", tmp_path / "ledger.json"
    code = main(["ledger", str(csv), "--target", "response", "--group", "site", "--beyond", "dose:high",
                 "--repeats", "1", "--block", "site properties=altitude", "--report", str(report),
                 "--json", str(tables)])  # fmt: skip
    assert code == 0
    out = capsys.readouterr().out
    assert "Verdicts" in out and "higher dose" in out
    assert report.read_text(encoding="utf-8").startswith("<!doctype html>") and tables.exists()


def test_errors_are_plain(csv):
    with pytest.raises(SystemExit, match="no column 'nope'"):
        main(["preflight", str(csv), "--target", "nope"])
    with pytest.raises(SystemExit, match="needs at least one shift"):
        main(["ledger", str(csv), "--target", "response"])
