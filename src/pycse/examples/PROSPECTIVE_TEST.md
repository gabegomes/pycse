# Prospective test: predictions written before the outside data were analysed

Written 2026-11-21, before any yield of the outside table below was read. The numbers this file
predicts from come from `KAN_Transfer_Ledger` and `KAN_Transfer_Planning` as executed and
committed on this branch (commit 37bfde7a). This file is committed before the analysis, so its
git timestamp precedes the results.

## The outside table

Cernak group, *J. Am. Chem. Soc.* **2026**, *148*, 26401 (DOI 10.1021/jacs.6c05959): 50,688
C-N couplings of 4 aryl halides with 2 amines, run with 33 Pd, Ni and Cu complexes, 164
ligands, 17 bases, several solvents and three temperatures; assay yields by UPLC-MS. Open
Reaction Database `ord_dataset-805ad863feef48579d95d86a728035f4`. Published in July 2026,
after every table this series was built on, and by a different laboratory.

Ligand descriptors: the kraken library (`kraken/kraken.csv`, 1560 phosphines), matched by
canonical SMILES. Only monodentate phosphines found in kraken enter the tests; how many there
are is reported, not chosen.

Definitions fixed in advance:

- *Active*: assay yield of at least 10 %, as in `KAN_Ligand_Reactivity_Cliff`.
- *Percent buried volume, minimum*: kraken `vbur_vbur_min` divided by the 3.5 A sphere volume
  (179.6 A^3), as in that notebook.
- A ligand's *activity* under a metal: the share of its reactions with that metal that are
  active.
- Ni, Pd and Cu are analysed separately. Reactions without a ligand, or with a ligand that is
  not a monodentate phosphine in kraken, are left out of H1 to H4.

## Predictions

**H1. The target transfers to Ni.** The window found on 90 phosphines in a Ni-catalysed
Suzuki coupling, minimum buried volume between 25.1 and 31.4 %, marks the active ligands in the
Ni-catalysed C-N couplings: ligands inside it have a higher activity than ligands outside it
(one-sided Mann-Whitney test over ligands, p < 0.05), and the share of active reactions is
higher inside than outside.

**H2. The cliff, not the window, is the general statement.** For Ni and for Pd separately, a
single cut of minimum buried volume between 28 and 34 % separates ligands of high and low
activity with a balanced accuracy above 0.65, and the side that is active is the one the
ligation state predicts: below the cut for Ni (bis-ligated), above it for Pd (mono-ligated).
For Pd this direction is a guess from the mechanism, not a result of this series; it is
written down so that it can fail.

**H3. With this many ligands the gate gives a verdict for the steric block.** Held out by
ligand (`NewGroups`), the equation chosen for a new ligand on the Ni reactions keeps a steric
descriptor and has skill above zero (interval excluding zero), if the table has at least 45
monodentate phosphines in kraken with Ni data. This follows from the detection floor of the
phosphine table: a property capturing a third of the ligand-to-ligand differences was found in
72 % of draws of 45 ligands and 91 % of draws of 60.

**H4. The detection law errs toward more ligands.** Fitted on 20 ligands of this table, the
detection law predicts the rates measured on 40 or more ligands too low or about right: mean
error at most +0.05 (predicted minus measured), and the ligands it asks for are at least the
number that turns out to be enough.

**H5. Established skill is kept.** Where the learning curve in ligands is established at half
the ligands, the skill measured with all of them is at or above the floor reported at half,
in at least 90 % of draws.

**H6. Spreading the ligands out does not help.** Training ligands chosen by farthest-point
selection in descriptor space are not better than random ones (better in at most 60 % of
draws).

Each hypothesis is reported as held, failed or not testable, with the reason. None is dropped
or redefined after the analysis.
