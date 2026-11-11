# transferledger

Readable equations from tabular data, and an account of which parts of a model still hold when
the cases change: a new group, a later time, a value range not seen in training.

It answers three questions about any table with a target column:

1. **What is the relationship?** An equation made of named one-variable shapes (a line, a step
   at a threshold, a window, an optimum), each in the variable's own units.
2. **Does it transfer?** Every model, and every variable in the equation, is scored on held-out
   rows twice: when the test cases resemble the training cases, and when they are new in a sense
   you name. Each number comes with an interval, and a verdict is only given when the interval
   supports it.
3. **Is this prediction inside what was fitted?** Each prediction can report which variables lie
   outside the range the equation saw.

The core needs only numpy, scipy, pandas and scikit-learn.

```
pip install -e packages/transferledger
```

## A worked example

Thirty sites, twenty measurements each. The response steps up at a dose of 6, falls with
temperature, depends on the operator, and each site has an offset of its own. `altitude` is a
property of the site that has nothing to do with the response.

```python
import numpy as np, pandas as pd
from transferledger import PieceModel, NewGroups, transfer_ledger

rng = np.random.default_rng(0)
site = np.repeat(np.arange(30), 20)
df = pd.DataFrame({
    "dose": rng.uniform(0, 10, 600),
    "temperature": rng.uniform(15, 35, 600),
    "altitude": rng.uniform(0, 2000, 30)[site],
    "operator": rng.choice(["ana", "ben"], 600),
})
y = (2.0 / (1 + np.exp(-(df.dose - 6.0) / 0.4)) - 0.08 * df.temperature
     + 0.5 * (df.operator == "ben") + 0.6 * rng.normal(size=30)[site] + 0.2 * rng.normal(size=600))

model = PieceModel().fit(df, y)
print(model.equation())
```

Output (footnotes on the notation left out here):

```
pieces:
  p0 = S((dose - 6.04) / 0.446)   [0 below 6.04, 1 above]
  p1 = (temperature - 25) / 5.72
  p2 = [operator=ana]  (0 or 1)
  p3 = W(altitude; 686, 1.77e+03, 21.2)   [1 inside 686 to 1.77e+03]
y = -1.752 +1.948*p0 -0.4392*p1 -0.5182*p2 +0.4216*p3
```

The step is found at 6.04 with the right height. But `altitude` is in the equation too: on
rows from sites it has seen, a window of altitudes stands in for those sites' offsets. The
ledger shows that this does not carry to a new site:

```python
ledger = transfer_ledger(df, y, NewGroups(site, "new site"))
print(ledger.ladder.round(2))
print(ledger.verdicts)
print(ledger.equation("new site"))
```

```
scheme                         reference  new site
baseline (training mean)           -0.00      0.00
equation                            0.80      0.72
equation, chosen for new site       0.77      0.76
curves                              0.81      0.64
curves, tuned for new site          0.79      0.75

                 new site
dose            transfers
temperature     transfers
operator        transfers
altitude     undetermined

pieces:
  p0 = S((dose - 6.04) / 0.547)   [0 below 6.04, 1 above]
  p1 = (temperature - 25) / 5.72
  p2 = [operator=ana]  (0 or 1)
y = -1.543 +2.012*p0 -0.4363*p1 -0.5231*p2
```

Chosen by how well it predicts sites it has not seen, the equation leaves `altitude` out, gives
up 0.03 of skill on the reference (the part that was memorized), and is the better model for a
new site.

## The ideas

**Pieces.** A curve source (by default penalized additive splines, an exact and repeatable fit)
gives one curve per variable. Each curve is matched to the simplest named form that reproduces
it. A form's edge must have at least three distinct values on each side; otherwise it would be a
lookup of one or two cases, not a shape. Text columns become 0/1 indicators. Products of two
pieces are admitted when the evidence supports them: a product is one variable changing the
effect of another.

**Shifts.** "Does this transfer?" has no answer until one says to what.

| Shift | Test cases are | Example |
|---|---|---|
| `NewGroups(groups, name)` | from groups absent in training (left out one at a time when there are few) | a new ligand, patient, site, batch |
| `ForwardInTime(time)` | later than every training case | next quarter |
| `BeyondRange(values, side)` | beyond the training range of one variable | higher temperature |
| `NewGroups.from_clusters(X, k)` | from a region of the variables not trained on | no group labels available |
| `Splitter(cv, n)` | whatever a scikit-learn splitter says | a custom design |

Several can be given at once; the ledger then has one column per shift. A variable can transfer
under one shift and not under another, so a verdict always names its shift.

**Skill.** The share of the baseline's loss a model removes, on held-out rows, where the baseline
predicts the training mean. 0 is the baseline; negative is worse than knowing nothing. The loss
is the deviance of the family (`"gaussian"`, `"binomial"` for a 0/1 target, `"poisson"` for
counts).

**Intervals and verdicts.** Intervals come from leaving out the shift's units one at a time (whole
groups, when groups are held out), with a small-sample correction, so four groups give wide
intervals. For each variable, the ledger removes it from the equation, refits, and records the
skill lost under the reference and under each shift:

| Verdict | On the reference | Under the shift |
|---|---|---|
| transfers | helps | helps |
| does not transfer | helps | hurts |
| undetermined | helps | interval spans zero |
| helps only under the shift | no clear benefit | helps |
| not established | no clear benefit | anything else |

Near-duplicate variables hide each other when removed one at a time; pass `blocks={name: [columns]}`
to also remove them together.

**Honest selection.** Variables enter the equation while they improve the held-out loss under
the chosen shift. Two details matter. Inside each selection fold the shapes are fitted on that
fold's training rows only, because a shape fitted on all rows has already seen the held-out
cases. And a variable is admitted only when its gain is clear of its own uncertainty over the
shift's units: with twelve groups, a chance association is real within the sample, and only
its uncertainty gives it away. In simulation that rule cuts the admission of pure-noise group
properties from 50 % of runs to 3 %, while a real one is still found 97 % of the time.

**Memorized share.** The reference skill an equation gives up when its variables must earn their
place under the shift: what it knew only by having seen the cases.

## Before fitting

```python
from transferledger import preflight
print(preflight(df, y, NewGroups(site, "new site")))
```

reports, per variable, how many distinct values it has and whether it is a property of the group
(then it has only as many values as there are groups); per shift, how many units there are and
how much of the target's variance lies between groups; and which variables are near-duplicates.

## At prediction time

```python
model.support(new_rows)
```

returns, per row, how many of the equation's variables lie outside the fitted range (their pieces
are held at the edge value), which ones, whether a text value was never seen, and the distance to
the nearest training row. Predictions are also kept within the range of the training target,
widened by half on each side (`limit=None` removes this).

## Reference

| Object | Purpose |
|---|---|
| `PieceModel(source, family, forms, select_on, products, ...)` | the equation; scikit-learn estimator with `fit`, `predict`, `equation()`, `terms_`, `components(X)`, `support(X)` |
| `transfer_ledger(X, y, shifts, model, models, blocks, ...)` | the ledger: `.ladder`, `.skill`, `.contributions`, `.verdicts`, `.blocks`, `.memorized`, `.selection`, `.distance`, `.models`, `.predictions` |
| `preflight(X, y, shifts)` | what the data can support |
| `SplineAdditive`, `KANAdditive`, `ModelCurves` | curve sources (the KAN needs `pip install "transferledger[kan]"`) |
| `register_form(Form(...))` | add a form of your own (a Hill curve, a power law) |
| `forms={"T": ["inverse"]}` | restrict a variable to forms the field already names (an Arrhenius term) |

## Status

Version 0.1. Not yet built: shrunken (rather than selected) equation weights, export of equations
to LaTeX and SymPy, an HTML report, a command line, and a suite of known-answer problems run in
continuous integration.
