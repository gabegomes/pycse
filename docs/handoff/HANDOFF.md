# Handoff: KAN chemistry notebooks, dataviz restyle

Delete `docs/handoff/` before anything is merged. It exists only so the next session can read it.

Owner: Gabe Gomes (gabegomes@cmu.edu). Repo: `gabegomes/pycse` (fork of `jkitchin/pycse`).
Prior session: https://claude.ai/code/session_019V2DnX1TAPGNuaNLJXPJ9y

## Working preferences

Concise replies, tables and bullets, casual tone, no emoji ever. Lead with the solution. When
changing a file in a back-and-forth, show only the changed parts. Do not push to `master` or
`main` of this fork, and do not push to upstream. PRs are drafts unless asked otherwise.
Commit messages end with the attribution trailers your session instructions give.

## Branches and PRs

| Branch | Head when handed off | What it is |
|---|---|---|
| `claude/improve-kan-visualizations-vkP9h` | `756a4fc` | Designated notebook branch. Contains the upstream 2.13.0 merge and both notebooks re-executed against 2.13.0 (constant descriptors dropped, dashboard picks best model across all models, ruff clean). |
| `claude/kan-vectorize-forward` | `7175127` | PR #4 (draft), base = the notebooks branch. `kan.py` only: vectorized forward pass and B-spline basis (L-BFGS compile went from over 30 min to 8 s) and a unit-scale fix for constant input features (`feature_std`), with two regression tests. CI was green on this head. |
| `claude/kan-viz-style` | `55e3517` | PR #5 (draft), base = `claude/kan-vectorize-forward`. Library plot restyle (`KAN.plot`, `plot_activations`, `plot_network`) plus the restyled notebook sources. CI was running on this head at handoff. |

Continue on **`claude/kan-viz-style`** (PR #5). Do not use `master`/`main`. If your session is
assigned its own branch name, create it from `claude/kan-viz-style` and open the PR against it,
or push to `claude/kan-viz-style` directly if you have write access to the fork.

PR event subscriptions belong to one session at a time. The previous session unsubscribed from
#4 and #5 at handoff; call `subscribe_pr_activity` for both.

## What is done

1. Upstream `jkitchin/pycse` master (2.13.0) merged into the notebooks branch.
2. Both notebooks re-executed. Metrics now (test split):
   - Buchwald-Hartwig, R2 / MAE: KAN 0.7025 / 10.83%, Gradient Boosting 0.6808 / 11.11%,
     Random Forest 0.6270 / 11.76%, MLR 0.5400 / 15.18%. KAN is best.
   - Sigman thioetherification, accuracy / AUC: Gradient Boosting 0.935 / 0.973,
     Random Forest 0.935 / 0.914, Logistic Regression 0.774 / 0.836, KAN 0.742 / 0.777.
3. Found and fixed: `KAN`/`KANLLPR` normalized inputs by `std + 1e-8`, so 7 constant Buchwald
   descriptors made finite-difference sensitivities explode (about 1e8). Fixed in the library
   (PR #4) and the notebook now drops zero-variance descriptors (32 to 25).
4. Dataviz audit of all 34 figures (17 per notebook) against the `dataviz` skill, then:
   - Shared style block in **cell 2 of both notebooks** (validated reference palette, hairline
     chrome, thin bars with rounded data end via `bars()` + `finish(fig)`, entity color maps
     `COMPONENT_COLORS` / `DESCRIPTOR_COLORS` / `PARTNER_COLORS` / `MODEL_COLORS`, `stat_tile`,
     `DIV_CMAP`, `SEQ_CMAP`). This cell is the style API; read it first.
   - Every figure cell converted (commit `55e3517`): no dual axes, no green/amber/red for
     non-states, no colored text, no black outlines, solid hairline grid, frameless legends,
     stat tiles instead of bordered cards and the 2-slice pie, 4-regime scatter as small
     multiples (the palette supports only 3 categories in scatter forms), full wrapped labels,
     emoji removed from printed text. Ruff format and check pass.
   - Library plots restyled (commit `b964048`), including the `plot_network` layer labels that
     were drawn on top of the nodes.

## What is NOT done (your job)

1. **Re-execute both notebooks and commit the outputs.** The notebooks on `claude/kan-viz-style`
   hold the new source but the OLD cell outputs, so figures are stale. A clean run was started
   in the previous container and is lost with it.
2. **Verify numbers did not change.** Compare every stream output against the notebooks branch
   (`claude/improve-kan-visualizations-vkP9h`) with a small script: the only expected
   differences are removed emoji (`Note:` instead of the warning sign) and figure titles.
3. **Look at every figure** (extract the PNGs from the executed notebook, open them). The
   workers did not re-check the final renders of Buchwald cells 18, 37, 52, the cell 45
   swatch change, or Sigman cell 44. Check against the acceptance list in `viz_brief.md`.
4. **Gates:** `uvx ruff@0.16.7 format --check src/pycse` and `uvx ruff@0.16.7 check src/pycse`
   (CI pins 0.16.7 and checks notebooks too), then `pytest -m "not slow"` (805 passed before).
5. Commit and push to `claude/kan-viz-style`; update PR #5's title and body (it still says WIP
   and "not runnable"); also copy the two finished notebooks onto
   `claude/improve-kan-visualizations-vkP9h` (the designated notebook branch):
   `git checkout claude/improve-kan-visualizations-vkP9h && git checkout claude/kan-viz-style -- src/pycse/examples/KAN_Chemistry_Yield_Prediction.ipynb src/pycse/examples/KAN_vs_MLR_Sigman_Descriptors.ipynb`,
   commit, push. Cells 20/21 outputs there come from the new library plot code that is not on
   that branch yet; say so in the commit message.
6. Delete `docs/handoff/`.

## Setup in a fresh container

```bash
git fetch origin && git checkout claude/kan-viz-style
git remote add upstream https://github.com/jkitchin/pycse.git   # only if you need upstream
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install -e ".[colab]" seaborn rdkit openpyxl jupyter nbconvert ipykernel pytest pyomo highspy
echo ".venv/" >> .git/info/exclude        # never commit .venv
python -m ipykernel install --user --name python3
# Sigman dataset: the notebook skips its download if this file exists.
curl -sSL -o /tmp/thioetherification_dataset.xlsx \
  https://raw.githubusercontent.com/SigmanGroup/Thioetherification-modeling/main/notebooks/dataset.xlsx
# (github.com/.../raw/... returned 403 through the proxy; raw.githubusercontent.com and
#  `git clone` both worked. Check the file is ~68 KB, not a 378-byte error page.)
cd src/pycse/examples
jupyter nbconvert --to notebook --execute --inplace KAN_Chemistry_Yield_Prediction.ipynb --ExecutePreprocessor.timeout=1800 &
jupyter nbconvert --to notebook --execute --inplace KAN_vs_MLR_Sigman_Descriptors.ipynb --ExecutePreprocessor.timeout=1800 &
wait
```

Runtime is roughly 15-30 minutes in parallel. Do not run them in the foreground of a session
that may restart; use a background task.

## Gotchas

- Python must be 3.12+ (upstream requires it).
- Without the PR #4 `kan.py`, `optimizer="bfgs"` (real L-BFGS since 2.12) compiles for 30+
  minutes on 38 features and the cells time out. `claude/kan-viz-style` already includes it.
- CI is `Fast Tests - Python 3.12` and `3.13` plus the ruff steps. Ruff covers notebook cells.
- `nbconvert --inplace` writes the notebook only at the end of the run.
- Interaction (hover, tooltips), dark mode and the texture option from the dataviz skill do not
  apply to static notebook figures.
- `docs/handoff/nbkernel.py` keeps a notebook kernel alive so one plot cell can be re-executed
  in seconds (`start`, `status`, `exec`, `code`, `stop`; see its docstring). Optional.
- `docs/handoff/viz_brief.md` is the brief the two restyle workers used: the rules and the
  per-figure acceptance check. The dataviz skill itself is bundled with Claude Code
  (`dataviz`); its references are `anti-patterns.md`, `marks-and-anatomy.md`,
  `choosing-a-form.md`, `color-formula.md`, `palette.md`.

## Decisions for Gabe

| Question | Default taken |
|---|---|
| Palette: the old notebook hues actually pass the skill's numeric palette checks; the problems were role/usage. I switched to the skill's validated reference palette anyway. | Reference palette. Reverting means changing the hex values in cell 2 only. |
| Library plot changes in PR #5 together with the notebooks, or separate? | Together, because cells 20/21 outputs depend on them. |
| Send the `kan.py` fixes (PR #4: vectorization, constant-feature scale) upstream to `jkitchin/pycse`? | Not done. They are general-purpose. |

## Content issues noticed, not fixed (styling pass only touched styling)

- Buchwald: single-parameter "threshold" rules all score the same accuracy (0.699, probably
  the majority-class rate), so those lollipops carry no information. "Nonlinear share" is
  100%, so the MLR-validity verdict is trivial. "Rate-limiting step" rests on mean KAN
  sensitivity, not kinetics. Regime R1 overlaps the others heavily in PCA space.
- Buchwald/Sigman cell 20: `plot_activations` hard-caps at 16 subplots, so it shows only the
  first two inputs; titles now say so. The 25- and 38-input `plot_network` renders are not
  legible at notebook size (library limitation).
- Sigman cell 44: the fourth cliff feature has maximum sensitivity about 0.03 and still prints
  "decreases by 0.00" and a decision-boundary warning. Printed headers are just "max"/"min".
- Sigman cell 50: "OPTIMAL: 15 features" rests on one split with a tiny test set; the
  `Current` architecture AUC (0.777, maxiter 300) and the retrain (maxiter 150) differ.
- Buchwald cell 50: the "All 25 features" reference (0.7025) differs slightly from the "Top 25"
  point (0.7036); they come from different fits.
- Other model classes' `plot()` methods (KANLLPR, DPOSE, KfoldNN, ...) still use old colors.
