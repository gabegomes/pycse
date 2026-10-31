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

### Readable equations and the transfer ledger (`src/pycse/sklearn/kan_symbolic.py`)
Imported directly. An additive KAN (`layers=(d, 1)`) is one curve per descriptor; each curve is
matched to the simplest form that reproduces it (line, 1/x, log, exponential, quadratic, V,
step, window), and a step or window edge needs `min_side` (3) distinct values on each side, or it
is a lookup of a few molecules.
- `snap()` - a `Formula` with every coefficient refit to the data (recovers slopes, step
  positions, volcano optima; `per_feature={"T_K": ["inverse"]}` gives an Arrhenius barrier)
- `extract_pieces()` / `select_pieces()` - unit shapes of one descriptor, clipped to their fit
  range; forward selection by CV, on random folds or (`groups=`) on held-out molecules
- `PieceModel` - estimator: sum of selected pieces; `select_on="groups"` keeps only pieces that
  help new molecules; `products=True` hands the residual to PySR with `+ - *` only (optional
  `pysr` extra, installs Julia on first use); `equation()` writes it out
- `transfer_ledger()` - every rung (mean, pieces, pieces chosen on new molecules, products,
  additive KAN, then any `models=`) scored on random folds and on grouped folds; per-descriptor
  drop-one ledger with verdicts (transfers, memorizes, ...); error per held-out molecule against
  its distance to the nearest training molecule; `memorized` = random-fold score lost when
  pieces must help new molecules
Tests in `test_sklearn_kan_symbolic.py` use an exactly additive stand-in model (fast); the KAN
and PySR end-to-end tests are marked slow (the PySR one skips without `pysr`).

### Chemistry example notebooks (`src/pycse/examples/`)
Seven notebooks share that pipeline, one per dataset, each ending in a written reading:
`KAN_Chemistry_Yield_Prediction` and `KAN_vs_MLR_Sigman_Descriptors` (the first two; their
closing section repeats the reading with the current pipeline), `KAN_Known_Truth_Benchmark`
(synthetic data with known generating equations), `KAN_Ligand_Reactivity_Cliff`,
`KAN_Buchwald_DFT_Descriptors`, `KAN_Direct_Arylation_Conditions` and
`KAN_Deoxyfluorination`. The five dataset notebooks prune near-duplicate descriptors, fit the
KAN with L1 and entropy penalties of 1e-3, read curves as accumulated local effects, and score
every model a second time on folds that hold out whole molecules. `KAN_Transfer_Ledger` runs
`kan_symbolic` on a calibration set with a known interaction and on the five datasets (each
loaded as in its own notebook): equations, ladder, descriptor verdicts, error against distance.
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
