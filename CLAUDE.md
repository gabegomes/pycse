# CLAUDE.md - Project Guide for AI Assistants

This file provides context for Claude and other AI assistants working with this codebase.

## Project Overview

**pycse** (Python Computations in Science and Engineering) is a Python library for scientific computing and engineering calculations. It provides:

- Regression and curve fitting with confidence intervals
- ODE solving with units support
- Machine learning models (sklearn-compatible)
- JAX-based neural networks
- Design of experiments utilities

**Version**: see `version` in `pyproject.toml` (single source of truth; `pycse.__version__` reads the installed metadata)
**Python**: >=3.12
**License**: GPL-3.0-or-later

## Directory Structure

```
pycse/
├── src/pycse/           # Main source code
│   ├── PYCSE.py         # Core regression functions (regress, nlinfit, etc.)
│   ├── cli.py           # `pycse` CLI (launch, mcp, skill)
│   ├── skill/           # Claude Code skill: SKILL.md + reference/*.md
│   ├── beginner.py      # Beginner-friendly utilities
│   ├── utils.py         # General utilities
│   ├── sklearn/         # sklearn-compatible ML models
│   │   ├── kan.py       # Kolmogorov-Arnold Networks
│   │   ├── jax_*.py     # JAX-based neural networks
│   │   ├── dpose.py     # Shallow ensemble with uncertainty
│   │   ├── zenn/        # Zentropy-enhanced neural networks
│   │   ├── tests/       # active_learning and zenn tests (also in testpaths)
│   │   └── ...
│   ├── tests/           # Main test suite
│   └── examples/        # Example notebooks
├── pycse-jb/            # Jupyter Book documentation (primary docs)
├── pycse-channel/       # YouTube companion materials
├── archive/             # Legacy org-mode documentation
├── data/                # Example data files
├── docker/              # Docker development environment
└── .github/workflows/   # CI configuration
```

## Development Setup

```bash
# Create virtual environment and install
uv venv
uv pip install -e ".[test]"

# Or with pip
pip install -e ".[test]"
```

## Testing

See `TESTING.md` for comprehensive testing documentation.

### Quick Commands

```bash
# Fast tests only (recommended for development, ~10 seconds)
pytest -m "not slow"

# All tests
pytest

# Slow tests only (ML/training, ~40 minutes)
pytest -m "slow"

# With coverage
pytest -m "not slow" --cov=src/pycse --cov-report=term-missing

# Parallel execution
pytest -m "not slow" -n auto

# Specific test file
pytest src/pycse/tests/test_pycse.py -v
```

### Test Markers

- `@pytest.mark.slow` - ML/training tests (>1 second)
- Use `pytestmark = pytest.mark.slow` for entire modules

### CI Workflows

- **pycse-tests.yaml** - Fast tests on every push/PR (~2 min)
- **pycse-tests-slow.yaml** - Slow tests on release tags (`v*`) or manual dispatch (~40 min)

## Code Style

- **Linter**: Ruff (configured in pyproject.toml)
- **Line length**: 100 characters
- **Target**: Python 3.12+

```bash
# Run linter
ruff check src/pycse

# Fix auto-fixable issues
ruff check --fix src/pycse
```

## Key Modules

### Core (`src/pycse/PYCSE.py`)
- `regress()` - Linear regression with confidence intervals
- `polyfit()` / `polyval()` - Polynomial fits with intervals
- `predict()` - Prediction intervals for linear fits
- `nlinfit()` - Nonlinear curve fitting
- `nlpredict()` - Prediction intervals for nonlinear fits
- `ivp()` - ODE initial value problems (`solve_ivp` with defaults)
- Prediction intervals (`yint`) have shape `(n, 2)`: lower, upper

### sklearn Models (`src/pycse/sklearn/`)
All models follow sklearn's estimator API (fit/predict/score). Public names are
lazily exported from `pycse.sklearn` (the `_LAZY` table in `sklearn/__init__.py`).

- `KAN`, `KANLLPR` - Kolmogorov-Arnold Networks
- `JAXMonotonicRegressor` - Monotonic neural networks (JAX)
- `JAXPeriodicRegressor` - Periodic neural networks (JAX)
- `JAXICNNRegressor` - Input-convex neural networks (JAX)
- `DPOSE` - Shallow ensemble with uncertainty
- `KfoldNN` - K-fold neural network ensemble
- `LLPRRegressor`, `NNBR`, `NNGMM`, `LeafModelRegressor`, `LinearRegressionUQ`
- `SISSO`, `SISSOEnsemble` - Symbolic regression (optional `sisso` extra)
- `ZENNRegressor`, `ZENNClassifier` - Zentropy-enhanced neural networks
- `ActiveLearner` - Active learning loop (`active_learning.py`)

### Interpretation pipeline (`src/pycse/sklearn/kan_interpret.py`)
Shared by the chemistry example notebooks; imported directly (not via the `_LAZY` table,
which is for estimators).
- `cv_compare()` - several models on identical repeated folds, scaler refit per fold, with a
  paired Wilcoxon test of one model against the best of the others; `groups=` holds out whole
  molecules per fold (does it work on a ligand it never saw?)
- `prune_collinear()` - keep one descriptor per group correlated above a threshold, so an
  effect is not split between interchangeable columns
- `accumulated_local_effects()` / `accumulated_local_effects_cv()` - the default curve
  reading: moves a descriptor only within reactions already near each value, so correlated
  descriptors do not make the model extrapolate; plus the envelope of per-fold refits
- `partial_effects()` / `partial_effects_cv()` - the older reading (other descriptors at their
  mean), kept for comparison
- `accumulated_local_effects_2d()` / `plot_interaction()` - the interaction of a descriptor
  pair alone (second-order ALE, double-centred; exactly zero for an additive model)
- `transitions()` - where a curve rises to and falls from its maximum, interpolated between
  grid points (the location of a step or of a window's edges)
- `classify_shape()` / `shape_table()` - Linear, Step, Saturating, Monotone, U-shaped,
  Inverted U, Window, Gap, Constant or Complex, by template fit against an explicit criterion
- `threshold_rules()` - best single cut per descriptor, class-balanced, scored by balanced
  accuracy against the majority baseline
- `sensitivity()`, `plot_partial_effect()`, `plot_fold_scores()`

### transferledger (`packages/transferledger/`)
A standalone, field-neutral package (own `pyproject.toml`, README and tests; core depends only on
numpy, scipy, pandas, scikit-learn; no import of pycse except the optional KAN curve source). It
replaced `pycse.sklearn.kan_symbolic`. Install with `pip install -e packages/transferledger`
(extras: `kan`, `pysr`, `sympy`); it is not a dependency of pycse.
- `PieceModel` - the equation of named pieces. Curve source `"spline"` (default: penalized
  additive splines, exact and repeatable), `"kan"`, or any object with `fit`/`curves`. Families
  gaussian, binomial, poisson. Tables in, text columns, sample weights; missing values are an
  error. `select_on=<Shift>` chooses variables under that shift; shapes are refit inside each
  selection fold and a variable needs `min_evidence` standard errors over the shift's units.
  A product may bring in its second variable; with a product in, its two variables' forms are
  chosen again by held-out loss (`rematch`); a simpler form replaces a variable's form when it
  costs less than `min_gain` of held-out skill (`simplify`). `products="pysr"` adds a PySR
  expression of the pieces. `equation()`, `terms_`, `components(X)`, `support(X)`; `save`/`load`, `to_latex`,
  `to_python` (numpy-only source), `to_sympy`
- shifts (`shifts.py`): `NewGroups` (leave-one-out when few; `.from_clusters`), `ForwardInTime`,
  `BeyondRange`, `Splitter`, and `Reference` (random folds)
- `transfer_ledger(X, y, shifts, ...)` - repeated folds; skill = share of the baseline's loss
  removed; delete-one-unit jackknife intervals; per-variable and per-block verdicts (transfers,
  does not transfer, undetermined, ...); memorized share; several shifts at once
  `ledger.report(path)` writes a self-contained HTML page; `to_json` every table
  `rule=True` adds a "one rule" rung below the equation (the best cut or window in one
  variable, chosen and scored like the equation; `ledger.rule(shift)`). Opt-in: it changes no
  existing table
  `ceiling=True` runs `unit_ceiling` for each `NewGroups` shift and puts it in the summary,
  the report ("Do the units differ at all?") and the JSON (`ledger.ceilings`). Opt-in too
- `preflight()` - distinct values, group-level variables, variance between groups, duplicates
- `unit_ceiling(X, y, NewGroups(...))` (`ceiling.py`) - on random folds (every unit seen), what
  the unit's identity adds over the variables that change within a unit (additive curves, and
  gradient boosting where a unit may change how the others act), and how much of that the
  unit's description reaches. `state()` is "units differ" or "nothing to explain"; the second
  means no description of the unit can be tested on the table (the Ni C-N couplings of the
  outside test: +0.001). Run it before reading a "does not transfer"
- The iterated fit (binomial, poisson) in `curves.py` and `model.fit_weights` shortens a step
  that would raise the penalized deviance. Full steps alternated between two bad fits on
  nearly separated yes/no targets (found on the outside test); do not remove the step control.
  `tests/test_curves.py::test_separated_yes_no_fit_does_not_overshoot` holds it
- `learning_curve(X, y, NewGroups(...))` (`learning.py`) - skill on unseen units against the
  number of units trained on; `state()` is "nothing yet" or "established", `floor()` the skill
  more units are expected to keep. It forecasts nothing: `backtest()` (measure on part of the
  units, then reveal the rest) showed that a curve at zero cannot be extended and that no
  extension is right in both directions; `extend()` exists only to repeat that check. Do not
  add a forecast without a backtest that supports it
- `detectable(X, y, NewGroups(...))` (`detection.py`) - plants a unit property that captures a
  known share of the unit-to-unit variance (trend or peak; the target is never simulated) and
  counts how often the analysis admits it: `floor(shape)` (read off the measured rates) and
  `needed(share, shape)` (from a probit law in `sqrt((units - 2) share / (1 - share))`, the
  evidence for a correlation of that strength; a three-parameter law in log units was
  unstable and extrapolated worse). `needed` is an estimate, not a bound: checked on the
  phosphines it asked for 0.7 to 1.2 times the ligands that were enough for a trend and 1.5
  to 2.3 times for a peak; on an outside table it was enough in 8 of 15 comparisons; fitted
  on 13 ligands it was of no use. Choosing the next
  units by farthest-point selection was tried and was no better than chance, so no such
  helper is offered
- `forms.py`: `register_form(Form(...))` adds a form; matching is grid plus exact linear solve.
  A rule that accepted a form within 1.5 standard errors of the curve was tried and dropped: on
  a yes/no target it let a V stand in for a window (the curve's noise says nothing about
  whether a shape predicts); held-out rows decide instead
- `cli.py`: `transferledger preflight|fit|ledger|plan|predict|truth` on CSV/TSV/Parquet files
- `truth.py`: eight known-answer problems; `scorecard(seeds)` gives the share of runs in which
  each fact is recovered, and the slow test holds each above its floor in `FLOORS`. Rerun
  `transferledger truth --seeds 20` after any change to forms, selection or curve sources
Tests: `packages/transferledger/tests` (in the repo's `testpaths`; also run by
`.github/workflows/transferledger-tests.yaml` with the package installed alone).

### Chemistry example notebooks (`src/pycse/examples/`)
Seven notebooks share that pipeline, one per dataset, each ending in a written reading:
`KAN_Chemistry_Yield_Prediction` and `KAN_vs_MLR_Sigman_Descriptors` (the first two; their
closing section repeats the reading with the current pipeline), `KAN_Known_Truth_Benchmark`
(synthetic data with known generating equations), `KAN_Ligand_Reactivity_Cliff`,
`KAN_Buchwald_DFT_Descriptors`, `KAN_Direct_Arylation_Conditions` and
`KAN_Deoxyfluorination`. The five dataset notebooks prune near-duplicate descriptors, fit the
KAN with L1 and entropy penalties of 1e-3, read curves as accumulated local effects, and score
every model a second time on folds that hold out whole molecules. `KAN_Transfer_Ledger` runs
`transferledger` on a calibration set with a known interaction, on the five datasets (each
loaded as in its own notebook, every component held out in turn) and on a housing table:
equations, ladder with intervals, verdicts per descriptor block and shift.
`KAN_Transfer_Planning` asks what would change a "does not transfer": whether the molecules
differ at all (`unit_ceiling`), learning curves in molecules, the forecast-then-reveal check of
what such a curve can say, the detection floor of each table and a check of its law on the 90
phosphines (about three hours on four cores; `PLAN_FAST=1` runs it with a handful of trials).
It was rerun with transferledger 0.2; its Reading ends with what the first run said and no
longer holds (only the phosphine table changed).
`KAN_Prospective_Test` scores nine predictions written beforehand in `PROSPECTIVE_TEST.md`
(same folder; do not edit its predictions) on outside tables: the Cernak C-N couplings
(Open Reaction Database, needs `ord-schema`), a 24-ligand C-H arylation, 8 new alcohols and a
Ni borylation (about three and a half hours on four cores; 5 held, 4 failed).
All of them download their data at run time and are not executed in CI.

### Utilities
- `hashcache.py` - Disk caching for expensive computations
- `beginner.py` - Simplified interfaces for teaching
- `mcp.py` - MCP server (`pycse_mcp`, optional `mcp` extra)
- `sklearn/mip.py` - MIP export of piecewise-linear models to Pyomo (optional `mip` extra).
  Models implement `mip_network(**options)` (a `MIPNetwork`, `TreeModel` or `SymbolicModel`)
  and get `to_pyomo()` from `MIPExportMixin`; tests in `test_sklearn_mip.py` solve with HiGHS
  and compare to `predict()` (std objectives and SISSO tests need gurobipy and skip without it).
  Walkthrough: `pycse-jb/.../docs/mip.ipynb`

### Claude Code skill (`src/pycse/skill/`)
`pycse skill install` copies this directory to `~/.claude/skills/pycse/`. Keep the
examples in it in sync with the API when signatures change; `test_cli.py` checks
the frontmatter and that its links resolve.

## Important Notes

### When Modifying Tests
- Mark slow tests (training, >1 sec) with `@pytest.mark.slow`
- Use small datasets (30-50 samples) for fast tests
- Use few epochs (5-10) for training tests
- See TESTING.md for detailed guidelines

### When Adding sklearn Models
- Follow sklearn estimator conventions
- Implement `fit()`, `predict()`, `score()`
- Add `get_params()` and `set_params()` for hyperparameters
- Include tests in `src/pycse/tests/` (`test_sklearn_<model>.py`); both test
  directories are in `testpaths` in pyproject.toml

### Pre-commit Hooks
Pre-commit runs fast tests only. Ensure `pytest -m "not slow"` passes before committing.

## Git Conventions

- Use conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `perf:`, `test:`
- Never use `--no-verify` with git commands
- Run fast tests before committing

## Building and Publishing

```bash
uv build      # Build wheel and sdist
uv publish    # Publish to PyPI
```

## Resources

- **Documentation**: `pycse-jb/` (Jupyter Book)
- **Testing Guide**: `TESTING.md`
- **PyPI**: https://pypi.org/project/pycse/
- **GitHub**: https://github.com/jkitchin/pycse
