# Brief: restyle one notebook's figures to the dataviz guidelines

You restyle the plotting code of ONE example notebook so every figure follows the
dataviz skill. Content, models, metrics and printed numbers must not change.

## Read first (5 minutes)

The `dataviz` skill (bundled with Claude Code; load it with the Skill tool). Its references:
- `anti-patterns.md` (what goes wrong; check every figure against it)
- `marks-and-anatomy.md` (mark specs, spacers, labels)
- `choosing-a-form.md`, `color-formula.md` (color jobs)

The notebook's cell 2 already contains the style block (tokens, `plt.rcParams`, helpers).
Read it first: it IS the style API. The old names `COLORS`, `SHAPE_COLORS`,
`MECHANISM_COLORS`, `FONT_*`, `create_card_style_box`, `format_value` no longer exist; remove
every use. Entity maps (`COMPONENT_COLORS` / `DESCRIPTOR_COLORS` / `PARTNER_COLORS` /
`MODEL_COLORS`) assign one fixed categorical slot per entity.

## Rules (a figure that breaks one of these is wrong)

Color
1. Color has one job. Identity -> `SERIES` slots in fixed order via the entity maps (never
   cycled, never by rank). Magnitude -> `SEQ_CMAP` (one hue). Polarity -> `DIV_CMAP` with a
   symmetric `vmin/vmax`, or two bars/lines: positive `SERIES[0]`, negative `SERIES[7]`.
   State -> `STATUS` tokens, always with a marker shape AND a text label.
2. Never use green/amber/red for things that are not states (shapes, regimes, rate-limiting
   steps, "high/low" classes, value-coded bars). Never viridis/RdYlGn/Reds/Blues/coolwarm/
   RdBu, never `'b-'`, `color="red"`, `edgecolor="black"`, tab10 defaults.
3. A nominal category with ONE measure (features, shapes, regimes, mechanism steps) takes ONE
   color (`ACCENT`); do not color bars by value or by category. Exception: when the category
   is an entity with a fixed map (component, descriptor type, partner, model), use that map.
4. Scatter / small multiples coloured by category are "all-pairs" forms: at most the first
   THREE slots. A 4th category -> facet into small multiples (focus group in `ACCENT`, the
   rest in `DEEMPHASIS`), or fold into "Other".
5. Text never wears a series color. Titles, labels, tick labels, legend text, annotations use
   `INK` / `INK_2` / `MUTED`; identity comes from a colored mark next to the text. Only text
   inside a filled mark may differ: use `ink_on(fill)`.
6. No dual axes. Never `twinx`/`secondary_y`. Two quantities with different scales -> two
   stacked panels sharing x (`sharex=True`).

Marks and chrome
7. Lines `LINE_W` solid; markers >= `MARKER` with a surface ring via `dots()` (dense scatter,
   > 300 points: smaller dots, `alpha` ~0.45, no ring is fine); areas are a 10-15% wash.
8. Bars via `bars()` and call `finish(fig)` (it caps thickness at 24 px, rounds the data
   end, keeps a 2 px gap between touching bars). Error bars: thin `INK_2` hairlines, small caps.
9. Chrome: `style_axes(ax, grid=...)` on every axes (hairline solid grid on the value axis
   only, no top/right spines). No dashed gridlines. Reference lines (identity, threshold, zero,
   mean) use `hline`/`vline` (solid hairline `MUTED`) plus a direct label or legend entry.
10. No borders around marks, no black outlines, no heavy frames.

Labels and legends
11. 2+ series -> a frameless `legend(ax)` (the mark is the key); <= 4 series are also direct
    labeled; one series -> no legend, the title names it. Never a number on every point:
    label the tip/extreme/endpoint only, <= ~8 bar values.
12. Nothing clipped or colliding. Truncated tick labels (e.g. "Frontier Orbita") must be
    fully readable (horizontal bars with room, or wrapped text). Legends must not sit on data.
13. Titles: sentence case, descriptive, no exclamation marks, no emoji or symbol glyphs
    (no check marks, warning signs, stars; write the word). Panel letters ("A.") are fine.
14. Pie/donut with 2 slices -> a `stat_tile`. Dashboard "cards" -> `stat_tile()` (label,
    value, note; the accent bar carries identity, text stays in ink).
15. Tables drawn with `ax.table`: light header (no dark navy fill), hairline row rules, ink
    text, numbers right-aligned; monospace (`FONT_MONO`) only for columns of numbers.
16. Heatmaps: counts -> `SEQ_CMAP`; z-scores -> `DIV_CMAP` with `vmin=-vmax`; annotate cells
    with `ink_on(cell color)`; thin labeled colorbar. If seaborn is used, pass the cmap
    explicitly and `linewidths=GAP, linecolor=SURFACE`.
17. Every figure's numbers must still be reachable as text (most cells already print a table;
    never delete a print, add one if a figure shows numbers that are printed nowhere).

Static notebook figures: hover/tooltips, dark mode and the texture option do not apply.

## Hard constraints

- Edit ONLY plotting code in the notebook you are assigned. Do not change models, data,
  seeds, metrics, computed variables, or printed text (except emoji removal and wording of
  figure titles). If a plotting cell also computes variables used later (`dominant_by_sens`,
  `regime_df_sorted`, `cliff_insights`, ...), keep those computations exactly.
- Do not touch the other notebook, `kan.py` or any other file. No git commands at all.
- No new dependencies. Keep the existing library plot calls (`kan.plot_activations()`,
  `kan.plot_network()`); they are already restyled, just verify they look right in context.
- Do not re-run the whole notebook (the lead does a final clean run).

## Workflow (fast loop)

A kernel with the notebook's complete state is already running. Helper:
`.venv/bin/python docs/handoff/nbkernel.py <cmd> ...`
- `status NAME` -> must say `ready` (if it says `running i/N`, read the code meanwhile)
- `exec NAME NOTEBOOK_PATH IDX [IDX...]` re-executes those notebook cells (source read from
  the notebook file as it is NOW) in the live kernel, prints text, and saves each figure to a
  PNG whose path it prints. Open the PNG with the Read tool and LOOK at it.
- `code NAME 'python...'` runs an ad-hoc snippet in that kernel (e.g. to inspect variables).
- First run `exec ... 2` to load the new style block into the kernel.
Edit cells by rewriting `nb["cells"][i]["source"]` with a small Python script
(json.load -> modify -> `json.dumps(nb, indent=1, ensure_ascii=False) + "\n"`), keep all other
fields untouched. Work cell by cell: edit -> exec -> Read PNG -> fix -> next. Finish with
`uvx -q ruff@0.16.7 format <notebook>` and `uvx -q ruff@0.16.7 check <notebook>` (both must pass;
they are CI gates).

## Per-figure acceptance check (run it on every PNG you produce)

[ ] colors are from the tokens; same entity = same color everywhere in this notebook
[ ] no dual axis, no dashed grid, no black outlines, no colored text
[ ] bars <= 24 px thick, square at the baseline, rounded at the data end
[ ] legend present for 2+ series, frameless, not covering data; no legend for 1 series
[ ] no label collisions or clipping; tick labels readable; nothing overflows the figure
[ ] no emoji/symbol glyphs; titles neutral
[ ] numbers unchanged vs the previous render (compare text output of the cell)

## Report back (concise)

Per cell: what was wrong, what you changed. List every deviation from the rules and why,
any figure you could not make compliant, any number/text that changed (should be none besides
emoji removal), and anything in the notebook content (not styling) that looks wrong or
misleading, without fixing it.
