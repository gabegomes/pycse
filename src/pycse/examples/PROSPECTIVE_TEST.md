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

### Analysis details for H2 to H6, fixed before the Cernak yields were read

- H2: "high" and "low" activity split the ligands of each metal at the median activity. The
  cut is searched from 28 to 34 % in steps of 0.1; the direction is the one stated in H2, and
  the balanced accuracy is that of the best cut in that direction.
- H3 to H6 use the target of `KAN_Ligand_Reactivity_Cliff`: active (yield at least 10 %) as a
  yes/no outcome, family binomial. Rows: Ni reactions with a kraken ligand. Variables: the
  twelve kraken descriptors of that notebook, pruned at |r| > 0.85 with the minimum buried
  volume kept first, plus indicators for catalyst, base, solvent, aryl halide, amine and
  temperature. Steric descriptors are the buried volumes and the Sterimol values.
- If the Ni rows are too many for the planning routines in a few hours on four cores, rows are
  thinned at random within each ligand, never ligands.

## A second outside table, added before its yields were read

Wang, J. Y.; Stevens, J. M.; Kariofillis, S. K.; ... Doyle, A. G. *Nature* **2024**, *626*,
1025 ("Identifying general reaction conditions by bandit optimization"): Pd-catalysed C-H
arylation of imidazoles, 24 ligands with 8 aryl bromides and 8 imidazoles (1536 reactions), the
reaction family of the 12-ligand direct arylation table (Shields et al. 2021). File
`deebo/aryl-scope-ligand.csv` of the Doyle lab data repository; 22 of the 24 ligands are in
kraken.

**H7. Twenty-four ligands are not enough for a property of the window's strength.** The plan
said that for the direct arylation ligands 24 would not be enough to establish a property
capturing a third of the differences between ligands (found in 16 to 25 % of draws on the
phosphines), and that 45 to 60 were. On this table, held out by ligand, the detection floor
for a trend is above 33 %.

## A third outside table: new alcohols

Żurański, Gandhi, Doyle, *J. Am. Chem. Soc.* **2023**, *145*, 7898 (DOI 10.1021/jacs.2c13093)
ran a validation set after the 37-alcohol table: 8 alcohols that table never had (mostly
primary benzylic), with PyFluor and PBSF and the same four bases, 116 reactions. Data and
M06-2X descriptors for all 45 alcohols: `data/` of github.com/doyle-lab-ucla/deoxyfluorination_modeling.
Disclosure: while checking the file layout, the first 7 rows (alcohol 1al with PyFluor and
PBSF) were printed before this prediction was written. No other yield of this set was read.

Fixed procedure: train on the 740 reactions of the 37 alcohols, with that paper's alcohol
descriptors plus the substructure flags of the validation file, and score on the new alcohols
against the training mean. Rungs: conditions only (base and sulfonyl fluoride), the equation
chosen on random folds, the equation chosen for a new alcohol, the curves tuned for a new
alcohol, gradient boosting with everything.

**H8. Alcohol descriptors do not carry to the new alcohols.** The ledger's verdict for the
alcohol block, "does not transfer", holds outside the table: no rung that uses alcohol
descriptors beats the conditions-only rung by more than 0.10 of skill on the new alcohols, and
the equation chosen on random folds scores below the conditions-only rung. (The 2023 paper's
own model, with interaction terms, reached a mean absolute error of 13 % against 18 % for a
random forest, so this can fail.)

## A fourth outside table: Ni-catalysed borylation

Stevens, Li, Simmons, ... Jackson, *Organometallics* **2022**, *41*, 1847 (DOI
10.1021/acs.organomet.2c00089): Ni-catalysed borylation of 33 aryl (pseudo)halides with 24
ligands in ethanol and methanol, plus an external set of 12 further ligands with 2 substrates.
Files `NiB/rxns/inchi.csv` and `NiB/rxns/inchi_val.csv` of the Doyle lab data repository; 19 of
the 24 ligands and all 12 external ones are in kraken. Not one yield of this table was read
before this was written.

**H9. The window marks the active ligands in Ni-catalysed borylation too.** Over the kraken
ligands of both files, ligands with a minimum buried volume between 25.1 and 31.4 % have a
higher activity (share of reactions with at least 10 % yield) than ligands outside it
(one-sided Mann-Whitney over ligands, p < 0.05).

Each hypothesis is reported as held, failed or not testable, with the reason. None is dropped
or redefined after the analysis.

## A fifth outside table, added 2026-11-22 before its yields were read

LeSueur, Tao, Doyle, Sigman, *Eur. J. Org. Chem.* **2024**, e202400428 (DOI
10.1002/ejoc.202400428): 28 further monophosphines screened on the bench in two of the
Ni-catalysed Suzuki-Miyaura couplings ("reactions 2 and 4") from which the window came.
Supporting information, Tables S8 and S9, as archived at Zenodo record 15652094. This is the
closest test of the window there is: the same reaction family, ligands screened later.

Disclosure. Only the article's main text was read: it says its own two-threshold rule (minimum
buried volume below 57.277 A^3, which is 31.9 %, and a second cut on the electrostatic
potential) classified these ligands with 83 % accuracy, and that the bench reactions gave
about 15 % more yield than the plates. So an upper cut near ours is known to do reasonably;
the lower edge of the window (25.1 %) and its exact upper edge (31.4 %) are what can fail.
Tables S8 and S9 were not opened before this was written.

Fixed in advance: active is a yield of at least 10 %; a ligand's activity is the share of its
reactions in these tables that are active; ligands are matched to kraken by identifier if the
tables give one, otherwise by structure. The test is run on the ligands that are not among
the 90 phosphines of `KAN_Ligand_Reactivity_Cliff` if there are at least 10 of them, and
reported for all 28 as well.

**H10. The window marks the active ligands among new ligands in the reactions it came from.**
Ligands inside the window (25.1 to 31.4 %) have a higher activity than ligands outside it
(one-sided Mann-Whitney over ligands, p < 0.05), and the rule "inside the window is active"
has a balanced accuracy of at least 0.65 over the ligand-reaction pairs.

## A sixth set of tables, added 2026-11-22 before their yields were read

The same supporting information reprints yields for Pd-catalysed reactions VI to XIII
(Tables S3 to S7: four cross-couplings, a Suzuki-Miyaura coupling, two Heck reactions and a
C(sp3)-H arylation, from several laboratories), each with kraken monophosphines. Only the
table captions were read before this was written.

**H11. For Pd the active side is above the cut.** H2 guessed this from the ligation state and
the Cernak table agreed with it on a thin base (Pd active above 29.5 %, balanced accuracy
0.75, on 2 % of reactions). Here: in more than half of the reactions VI to XIII that have at
least 5 active and 5 inactive kraken ligands, a single cut of minimum buried volume between 28
and 34 % (steps of 0.1), with the active ligands above it, has a balanced accuracy over
ligands above 0.65. Active is a yield of at least 10 %. Reactions with fewer than 5 ligands on
either side are reported as not testable.

## Outcome (added after the analysis; the predictions above are unchanged)

Scored in `KAN_Prospective_Test.ipynb`: H4, H5, H6, H7 and H9 held; H1, H2, H3 and H8 failed.
Two points the notebook's reading spells out:

- H4 does not say in how many comparisons the law must ask for enough ligands. The notebook
  used at least half; that rule was set before the run but was not committed here. Under
  "every comparison" H4 fails.
- H5 rests on one check: the curve was established at half the ligands in 1 of 16 draws.

Post hoc, the Ni activity of the Cernak table is the metal-free background (0 of 3577 Ni
reactions with a kraken phosphine active where the metal-free controls give no product), so
H1 to H3 could not have found a ligand effect on Ni there.

H10 and H11, scored in `KAN_Prospective_Followup.ipynb`, both held:

- H10: of the 28 ligands, 23 are not among the 90 phosphines. Their activity is 0.74 inside
  the window and 0.17 outside (p = 0.005); balanced accuracy over the 46 ligand-reaction pairs
  0.73. With the buried volumes as printed in the tables instead of the kraken file (they
  differ for one ligand): p = 0.016, balanced accuracy 0.67.
- H11: six of the eight Pd reactions are testable; "active above the cut" has a balanced
  accuracy over 0.65 in four of them (VII, VIII, IX, XI; 0.85, 0.88, 0.92, 0.66), not in VI
  (0.63) and XIII (0.57).

A defect found afterwards. H3 to H6 were scored with `transferledger` 0.1.0, whose yes/no fit
overshot on nearly separated outcomes (the curves rung on the Ni table stood at -2.8 on random
folds). Version 0.2 repairs it. `KAN_Prospective_Followup.ipynb` repeats H3 to H6 as
registered with the repaired fit and shows both results; the verdicts above are the registered
ones.
