# transferledger

Readable equations from tabular data, and an account of which parts of a model still hold when
the cases change: a new group, a later time, a value range not seen in training.

It answers four questions about any table with a target column:

1. **What is the relationship?** An equation made of named one-variable shapes (a line, a step
   at a threshold, a window, an optimum), each in the variable's own units.
2. **Does it transfer?** Every model, and every variable in the equation, is scored on held-out
   rows twice: when the test cases resemble the training cases, and when they are new in a sense
   you name. Each number comes with an interval, and a verdict is only given when the interval
   supports it.
3. **Is this prediction inside what was fitted?** Each prediction can report which variables lie
   outside the range the equation saw.
4. **What would it take?** Where nothing about a group carries to a new group: how strong would a
   property of the group have to be to show up with the groups in hand, and how many groups
   would a weaker one need?

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
df = pd.DataFrame(
    {
        "dose": rng.uniform(0, 10, 600),
        "temperature": rng.uniform(15, 35, 600),
        "altitude": rng.uniform(0, 2000, 30)[site],
        "operator": rng.choice(["ana", "ben"], 600),
    }
)
y = (
    2.0 / (1 + np.exp(-(df.dose - 6.0) / 0.4))
    - 0.08 * df.temperature
    + 0.5 * (df.operator == "ben")
    + 0.6 * rng.normal(size=30)[site]
    + 0.2 * rng.normal(size=600)
)

model = PieceModel().fit(df, y)
print(model.equation())
```

Output (footnotes on the notation left out here):

```
pieces:
  p0 = S((dose - 6.04) / 0.446)   [0 below 6.04, 1 above]
  p1 = (temperature - 25) / 5.72
  p2 = [operator=ana]  (0 or 1)
  p3 = |altitude - 1.63e+03| / 492
  p4 = (altitude - 946) / 492
y = -0.4663 +1.959*p0 -0.4337*p1 -0.522*p2 -0.7286*p3 -0.5371*p4
```

The step is found at 6.04 with the right height. But `altitude` is in the equation too: on
rows from sites it has seen, a bend in altitude stands in for those sites' offsets. The
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
lookup of one or two cases, not a shape. Values closer than 2 % of the variable's range count
as one: nine molecules that differ in the fourth decimal are one value. A form also has to earn its place over a simpler one:
for each variable in the equation the simpler forms are tried, and the simplest is taken that
costs less held-out skill than a variable must add. A noisy curve may be reproduced best by a
window; if a line predicts as well, the equation says a line. Text columns become 0/1
indicators.

**Products.** A product of two pieces is one variable changing the effect of another, and is
admitted by the same rule as a variable. One of the two must already be in the equation; the
other may enter with the product, so a variable that acts mostly through another is not lost
for having a weak curve of its own. With a product in, the forms of its two variables are
chosen again by the held-out loss of the whole equation: an interaction that a sum of curves
cannot express blurs exactly those two curves, and a step can pass for a bend until the product
carries it. `products="pysr"` also asks PySR for an expression of the chosen pieces.

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
its uncertainty gives it away. In simulation (twelve groups, three group properties, 60 runs)
that rule cuts the admission of a pure-noise property from 68 % of runs to 10 % when none is
real, and from 53 % to 3 % when one is, while the real one is still found 92 % of the time.

**Memorized share.** The reference skill an equation gives up when its variables must earn their
place under the shift: what it knew only by having seen the cases.

**One rule.** `transfer_ledger(..., rule=True)` adds a rung below the equation: the best
statement about one variable, a cut or a window ("active between 25 and 31"), chosen and scored
like the equation. `ledger.rule("new site")` prints it. An equation can keep a variable's
threshold and still lose skill on new groups through its other terms; when the rule rung keeps
under a shift what the equation loses, the rule is the thing to carry to a new case.

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

## What it would take

"Nothing about a site carries to a new site" leaves two readings open: the description of a site
holds nothing, or it holds something that this many sites cannot establish.

```python
from transferledger import unit_ceiling, learning_curve, detectable, backtest

sites = NewGroups(site, "new site")
top = unit_ceiling(df, y, sites)  # what knowing the site adds, on sites already seen
print(top)
top.state()  # "units differ" or "nothing to explain"
top.captured()  # share of that which the site's description reaches

curve = learning_curve(df, y, sites)  # skill on a new site, trained on 4, 6, 9, ... sites
print(curve)
curve.state()  # "nothing yet" or "established"
curve.floor()  # the skill more sites are expected to keep, or None

found = detectable(df, y, sites)  # plant properties of known strength, count how often found
print(found)
found.floor("trend")  # smallest share of the site-to-site variance that shows up
found.needed(0.3, "peak")  # sites needed for a property that captures 30 %
```

**The ceiling** comes first. On random folds, where every group is seen in training,
`unit_ceiling` scores three models: the variables that change within a group, the same plus the
group's description, and the same plus the group's identity (one shrunken level per group, and
a gradient-boosting model in which a group may also change how the other variables act). What
identity adds is the most any description of the group could add. If that cannot be told from
zero the state is "nothing to explain": the table cannot test a description of the group,
however many groups it has, and "does not transfer" says nothing about the description. This
is what happened on an outside table of Ni-catalysed C-N couplings, where every active
reaction also ran without the metal: knowing the ligand added +0.001 [-0.006, +0.008].
Otherwise `captured()` is the share of the ceiling that the description reaches on groups
already seen; whether it carries to a new group is the ledger's verdict.
`transfer_ledger(..., ceiling=True)` runs it for every shift that holds out groups and puts the
answer in the ledger's summary, its report ("Do the units differ at all?") and its JSON, next
to the verdicts it qualifies.

**The learning curve** trains on a part of the groups and scores on groups held out, size by
size. What it may say about *more* groups was settled by `backtest`, which measures a curve on a
part of the groups and then reveals the rest. On 90 phosphines, 37 alcohols, 160 housing regions
and made-up populations with known structure:

- a curve that sits at zero does not say where it will leave zero (extended to 1.5, 2 and 3
  times the groups it was too low by 0.11, 0.23 and 0.30; curves measured on 13 phosphines and
  extended to 39 said 0.18, and 0.38 was then measured);
- no way of extending a curve was right in both directions;
- skill, once established, was kept when groups were added (in 110 of 114 checks).

So the curve reports a state and a floor, and forecasts nothing. `curve.extend()` gives the
textbook extension for anyone repeating the check.

**The detection floor** answers the case the curve cannot. `detectable` adds to the table a made-up
property that captures a chosen share of what distinguishes one group from another (the group's
own mean of the target, blurred), runs the whole analysis as on the real table, and counts how
often the property is admitted. The target is never simulated. Two shapes are planted: a trend,
and a peak (high in the middle, low on both sides), which needs both sides populated. The floor
is the smallest share found in 80 % of plantings: "nothing transfers" then reads "no variable in
this table captures that much". Groups needed for a weaker property come from a law fitted to
the counts, up to three times the groups in hand.

That law was checked in the same forecast-then-reveal way (notebook `KAN_Transfer_Planning`).
Fitted on 20 of the phosphines, its detection rates for 26 to 60 were low by 0.00 to 0.15; it
asked for 0.7 to 1.2 times the ligands that were then enough for a trend, and 1.5 to 2.3 times
for a peak. Fitted on 13 it was of no use. So `needed()` leans toward more groups, most for a
peak; it is an estimate, not a bound, and it wants about 20 groups in hand. The real
descriptor behind the ligand set's threshold captures a third of the differences between
ligands; it was admitted from none of the draws of 13 ligands, 28 % of draws of 30 and 91 % of
draws of 60.

One more thing was tried and is not offered: choosing the next groups to cover the descriptor
space (farthest-point selection). On the phosphines it was worse than choosing at random, and
on the alcohols no different.

**Checked on outside tables.** Predictions written before the yields were read were scored on
tables from other laboratories (notebook `KAN_Prospective_Test`). Fitted on 20 of 52 ligands of a
Ni C-N coupling table, the law's rates were again too low (by 0.03 to 0.05 for a trend, 0.08 to
0.38 for a peak), but the ligands it asked for were enough in only 8 of 15 comparisons. On 22
ligands of a C-H arylation table the floor was 49 % (trend) and 67 % (peak), as the phosphines
had implied for 24. Where the ligand descriptors carried nothing (that Ni table, whose activity
is the metal-free background), the ligand block was given 0.000 and the chosen equation kept no
ligand descriptor.

Two further predictions concerned the threshold itself (notebook `KAN_Prospective_Followup`).
The window found on 90 phosphines (minimum buried volume between 25.1 and 31.4 %) was scored on
23 phosphines it had never seen, screened later in two of the same reactions: active in 74 %
of their reactions inside the window against 17 % outside (p = 0.0047), balanced accuracy 0.73.
And in 4 of 6 Pd-catalysed reactions the active ligands lay above a cut, the opposite side, as
the ligation-state argument says.

## Taking the result elsewhere

```python
model.save("equation.json")  # and PieceModel.load("equation.json")
print(model.to_latex())  # for a paper
open("equation.py", "w").write(model.to_python())  # numpy only, no transferledger needed
model.to_sympy()  # needs the sympy extra

ledger.report("ledger.html", title="Yield across sites")  # one self-contained page
ledger.to_json("ledger.json")  # every table
```

The report is a single HTML file with no scripts and no external resources: the pre-check, the
ladder with intervals, each variable under each shift with its verdict, and the equations.

## Command line

```
transferledger preflight data.csv --target yield --group site
transferledger fit       data.csv --target yield --select-on group:site --save equation.json
transferledger ledger    data.csv --target yield --group site --time date --report ledger.html
transferledger plan      data.csv --target yield --group site --floor
transferledger predict   equation.json new_rows.csv --out predictions.csv
transferledger truth
```

`--group`, `--time` and `--beyond COL[:high|low|both]` each add a shift and may be repeated;
`--clusters K` holds out regions of the numeric columns when there are no labels. A column that
defines a shift is not used as a predictor unless named with `--keep`. `predict` adds, per row,
which variables lie outside the fitted range. `ledger --rule` also scores the best one-variable
rule; `ledger --ceiling` also asks whether the units of each `--group` differ at all. CSV, TSV and Parquet files are read.

## Known answers

`transferledger truth` (or `transferledger.scorecard()`) runs eight problems drawn from stated
generators and reports how often each fact a correct analysis must recover was recovered:

| Problem | Facts |
|---|---|
| rate law with an interaction | step at 32, optimum at -15, a barrier of 15, the slope on each side of the step, an unrelated variable left out |
| the same without the interaction | no product invented |
| group properties that are noise | none admitted |
| one real group property | found, and the noise ones left out |
| yes/no outcome with a threshold | threshold at 6, probabilities returned |
| yes/no outcome with a window | both edges of the window |
| counts with a log-linear rate | the rate per unit |
| a cycle over time | the time variable left out when chosen for later runs |

Over 20 seeds all 20 facts are recovered in at least 85 % of runs. The package's slow tests
hold each above a stated floor, so a change that makes the method quietly worse fails.

## Reference

| Object | Purpose |
|---|---|
| `PieceModel(source, family, forms, select_on, products, ...)` | the equation; scikit-learn estimator with `fit`, `predict`, `equation()`, `terms_`, `components(X)`, `support(X)` |
| `transfer_ledger(X, y, shifts, model, models, blocks, rule, ceiling, ...)` | the ledger: `.ladder`, `.skill`, `.contributions`, `.verdicts`, `.blocks`, `.memorized`, `.selection`, `.distance`, `.models`, `.predictions`, `.rule(shift)`, `.ceilings` |
| `preflight(X, y, shifts)` | what the data can support |
| `unit_ceiling(X, y, shift)` | what knowing the group adds on groups already seen, and how much of it the group's description reaches: `.state()`, `.ceiling()`, `.captured()`, `.skill`, `.adds` |
| `learning_curve(X, y, shift)` | skill on unseen groups against the number of groups trained on: `.curve`, `.state()`, `.floor()`, `.advice()` |
| `backtest(X, y, shift, have=...)` | measure a curve on part of the groups, then reveal the rest |
| `detectable(X, y, shift)` | planted properties of known strength: `.power`, `.floor(shape)`, `.needed(share, shape)`, `.rate(units, share, shape)` |
| `ledger.report(path)`, `ledger.to_json(path)` | the HTML report; every table as JSON |
| `model.save`, `PieceModel.load`, `to_latex`, `to_python`, `to_sympy` | the equation as a file, LaTeX, numpy source, SymPy |
| `scorecard(seeds)` | the known-answer problems |
| `SplineAdditive`, `KANAdditive`, `ModelCurves` | curve sources (the KAN needs `pip install "transferledger[kan]"`) |
| `register_form(Form(...))` | add a form of your own (a Hill curve, a power law) |
| `forms={"T": ["inverse"]}` | restrict a variable to forms the field already names (an Arrhenius term) |

## Status

Version 0.2. The equation's shape parameters (where a step sits, how wide it is) come from the
curve and are not refined against the data afterwards. Missing values are an error, not
handled. Families beyond gaussian, binomial and poisson are not built.

Changes in 0.2:

- Fixed: the iterated fit for yes/no and count targets took full steps, which on a nearly
  separated outcome overshoot and then alternate between two bad fits. The curves rung could
  then score far below the training mean (on a table of 6560 yes/no reactions: -2.8 on random
  folds, 0.56 after the fix), and curves fitted on few groups were affected most. Steps are now
  shortened until the penalized deviance does not rise (`SplineAdditive.damped_` says whether
  that was needed). Fits that had converged are unchanged; Gaussian targets are untouched.
- Added `unit_ceiling`, and `transferledger plan` prints it first.
- Added the one-rule rung: `transfer_ledger(..., rule=True)`, `ledger.rule(shift)`, and
  `transferledger ledger --rule`.
- Added `transfer_ledger(..., ceiling=True)` and `transferledger ledger --ceiling`: the ceiling
  in the ledger's summary, report and JSON (`ledger.ceilings`). Opt-in; no other table changes.
