"""A self-contained HTML report of a ledger: one file, no scripts, no network.

Charts are inline SVG drawn here, so the report opens anywhere and prints. Every chart has the
table it was drawn from next to it.
"""

from __future__ import annotations

import html
import math

import numpy as np

__all__ = ["render"]

_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#8a63d2", "#c9a227", "#d03b7a"]
_VERDICT_CLASS = {
    "transfers": "good", "does not transfer": "bad", "undetermined": "warn",
    "helps only under the shift": "info", "not established": "mute",
}  # fmt: skip

_CSS = """
:root { --bg:#f7f8fa; --panel:#ffffff; --fg:#14161a; --fg2:#4b5160; --rule:#d9dde5; --mute:#868c9a;
  --good:#0f7a3d; --goodbg:#e2f4e8; --bad:#b3261e; --badbg:#fbe4e2; --warn:#8a5a00; --warnbg:#fbf0d5;
  --info:#1d5fae; --infobg:#e1edfb; --mutebg:#eceef2; }
@media (prefers-color-scheme: dark) { :root { --bg:#121417; --panel:#1b1e24; --fg:#e8e9ec; --fg2:#b4b8c2;
  --rule:#2d313a; --mute:#7f8594; --good:#6fd39a; --goodbg:#163524; --bad:#f19a94; --badbg:#3d1b19;
  --warn:#e8c26a; --warnbg:#3a2f12; --info:#8fbcf2; --infobg:#172a44; --mutebg:#262a32; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 28px 16px 60px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 19px; margin: 34px 0 8px; } h3 { font-size: 15px; margin: 20px 0 6px; }
p { margin: 6px 0 10px; max-width: 72ch; } .small { color: var(--fg2); font-size: 13px; }
.wrap { overflow-x: auto; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; margin: 8px 0 14px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th { font-weight: 600; color: var(--fg2); } tr:last-child td { border-bottom: 0; }
td.num { font-variant-numeric: tabular-nums; }
.chip { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; }
.good { color: var(--good); background: var(--goodbg); } .bad { color: var(--bad); background: var(--badbg); }
.warn { color: var(--warn); background: var(--warnbg); } .info { color: var(--info); background: var(--infobg); }
.mute { color: var(--mute); background: var(--mutebg); }
pre { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 10px 12px; overflow-x: auto;
  font: 12.5px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
svg { display: block; max-width: 100%; height: auto; } svg text { fill: var(--fg2); font: 12px system-ui, sans-serif; }
.legend span { display: inline-block; margin-right: 14px; font-size: 13px; color: var(--fg2); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 5px; margin-right: 5px; }
ul { margin: 6px 0 10px 18px; padding: 0; max-width: 72ch; } li { margin: 3px 0; }
"""


def _e(s):
    return html.escape(str(s))


def _num(v, d=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return f"{v:.{d}f}"


def _table(columns, rows, numeric=()):
    head = "".join(f"<th>{_e(c)}</th>" for c in columns)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f'<td class="num">{c}</td>' if i in numeric else f"<td>{c}</td>" for i, c in enumerate(r)
        ) + "</tr>"  # fmt: skip
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _legend(names):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{_SERIES[i % len(_SERIES)]}"></i>{_e(n)}</span>' for i, n in enumerate(names)
    ) + "</div>"  # fmt: skip


def _whiskers(labels, series, lo_clip, hi_clip, xlabel, zero=True, label_width=250):
    """Dot-and-whisker chart. ``series`` is a list of ``(name, [(point, lo, hi) per label])``."""
    width, plot = 900, 900 - label_width - 30
    row = 12 * len(series) + 12
    height = row * len(labels) + 46
    span = hi_clip - lo_clip

    def x(v):
        return label_width + plot * (min(max(v, lo_clip), hi_clip) - lo_clip) / span

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_e(xlabel)}">']
    ticks = np.linspace(lo_clip, hi_clip, 6)
    for t in ticks:
        out.append(
            f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="6" y2="{height - 34}" stroke="var(--rule)"/>'
        )
        label = f"{t:.2f}".replace("-0.00", "0.00")
        out.append(f'<text x="{x(t):.1f}" y="{height - 18}" text-anchor="middle">{label}</text>')
    if zero and lo_clip < 0 < hi_clip:
        out.append(
            f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="6" y2="{height - 34}" stroke="var(--mute)" stroke-dasharray="3 3"/>'
        )
    out.append(
        f'<text x="{label_width + plot / 2:.1f}" y="{height - 3}" text-anchor="middle">{_e(xlabel)}</text>'
    )
    for i, lab in enumerate(labels):
        y0 = 8 + i * row
        text = lab if len(lab) <= 40 else lab[:38] + "..."
        out.append(
            f'<text x="{label_width - 10}" y="{y0 + row / 2 + 2:.1f}" text-anchor="end">{_e(text)}</text>'
        )
        for k, (_, values) in enumerate(series):
            p, lo, hi = values[i]
            if p is None or not math.isfinite(p):
                continue
            c = _SERIES[k % len(_SERIES)]
            yy = y0 + 9 + 12 * k
            out.append(
                f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{yy}" y2="{yy}" stroke="{c}" stroke-width="2"/>'
            )
            out.append(f'<circle cx="{x(p):.1f}" cy="{yy}" r="4" fill="{c}"/>')
            if p < lo_clip:  # off the axis: say so rather than pile dots on the edge
                out.append(
                    f'<text x="{label_width + 8}" y="{yy + 4}" style="font-size:11px">{p:.1f}</text>'
                )
    out.append("</svg>")
    return "".join(out)


def render(ledger, title="Transfer ledger", preflight=None):
    """The ledger as one HTML page (a string)."""
    schemes = ledger.schemes
    shifts = schemes[1:]
    skill = ledger.skill.set_index(["rung", "scheme"])
    rungs = list(dict.fromkeys(ledger.skill["rung"]))
    pct = f"{100 * ledger.level:.0f} %"
    meta = getattr(ledger, "meta", {}) or {}
    parts = [f"<h1>{_e(title)}</h1>"]
    bits = []
    if meta:
        bits.append(f"{meta.get('rows', '?')} rows, {meta.get('columns', '?')} predictor columns")
        bits.append(
            "units per scheme: "
            + ", ".join(f"{_e(k)} {v}" for k, v in meta.get("units", {}).items())
        )
    bits.append(
        f"family {ledger.family}; {ledger.n_splits} folds x {ledger.n_repeats} repeats; {pct} intervals"
    )
    parts.append(f'<p class="small">{"; ".join(bits)}.</p>')
    parts.append(
        "<p>Every number is measured on held-out rows. <strong>Skill</strong> is the share of the baseline's "
        "loss a model removes, where the baseline predicts the training mean: 0 is the baseline, 1 is perfect, "
        "negative is worse than knowing nothing. The <strong>reference</strong> column uses random folds, where "
        "test cases resemble training cases; each other column is a named shift, where every test case is new in "
        "that sense. Intervals come from leaving out the shift's units one at a time.</p>"
    )

    # ---- ladder
    parts.append("<h2>Models under each shift</h2>")
    lo_clip = max(-1.0, min(-0.1, float(np.floor(10 * min(ledger.skill.skill.min(), 0)) / 10)))
    series = [
        (s, [tuple(skill.loc[(r, s), ["skill", "lo", "hi"]]) for r in rungs]) for s in schemes
    ]
    parts.append(_legend(schemes))
    parts.append(_whiskers(rungs, series, lo_clip, 1.0, "skill"))
    rows = [[_e(r)] + [f"{_num(skill.loc[(r, s), 'skill'])} [{_num(skill.loc[(r, s), 'lo'])}, {_num(skill.loc[(r, s), 'hi'])}]"
                       for s in schemes] for r in rungs]  # fmt: skip
    parts.append(_table(["model"] + schemes, rows, numeric=range(1, len(schemes) + 1)))
    parts.append(
        '<p class="small">"equation" is chosen on the reference; "equation, chosen for X" admits a variable only if '
        'it helps cases that are new in the sense of X. "curves" is the additive model the pieces were read from, '
        "before any form is imposed.</p>"
    )

    # ---- memorized
    if len(ledger.memorized):
        parts.append("<h2>What was memorized</h2>")
        parts.append(
            "<p>The reference skill the equation gives up when its variables must earn their place under the shift: "
            "what it knew only by having seen the cases.</p>"
        )
        rows = [
            [_e(k), _num(r.memorized), f"[{_num(r.lo)}, {_num(r.hi)}]"]
            for k, r in ledger.memorized.iterrows()
        ]
        parts.append(_table(["shift", "memorized", f"{pct} interval"], rows, numeric=(1, 2)))

    # ---- verdicts
    def chips(table, first):
        rows = []
        for name, r in table.iterrows():
            rows.append(
                [_e(name)]
                + [f'<span class="chip {_VERDICT_CLASS.get(v, "mute")}">{_e(v)}</span>' for v in r]
            )
        return _table([first] + list(table.columns), rows)

    def forest(frame, key, shift):
        f = frame.set_index([key, "scheme"])
        names = list(dict.fromkeys(frame[key]))[:24]
        vals = frame[frame[key].isin(names) & frame.scheme.isin(["reference", shift])]
        lo = max(-1.0, float(np.floor(10 * min(vals.lo.min(), -0.05)) / 10))
        hi = min(1.0, float(np.ceil(10 * max(vals.hi.max(), 0.05)) / 10))
        series = [
            (s, [tuple(f.loc[(n, s), ["contribution", "lo", "hi"]]) for n in names])
            for s in ("reference", shift)
        ]
        return _legend(["reference", shift]) + _whiskers(
            names, series, lo, hi, "skill lost when removed"
        )

    if len(ledger.blocks):
        parts.append("<h2>Blocks of variables</h2>")
        parts.append(
            "<p>Variables removed together. This is the right question when variables are near-duplicates, because "
            "each can stand in for another when removed alone.</p>"
        )
        parts.append(chips(ledger.block_verdicts, "block"))
        for s in shifts:
            parts.append(f"<h3>{_e(s)}</h3>" + forest(ledger.blocks, "block", s))
    if len(ledger.verdicts):
        parts.append("<h2>Variables</h2>")
        parts.append(
            "<p>Each variable of the reference equation is removed, the weights are refitted, and the skill lost is "
            "recorded under the reference and under each shift.</p>"
        )
        parts.append(chips(ledger.verdicts, "variable"))
        parts.append(
            "<ul class='small'><li><strong>transfers</strong>: helps on the reference and under the shift.</li>"
            "<li><strong>does not transfer</strong>: helps on the reference, hurts under the shift.</li>"
            "<li><strong>undetermined</strong>: helps on the reference; under the shift the interval spans zero.</li>"
            "<li><strong>helps only under the shift</strong>; <strong>not established</strong>: no clear benefit even "
            "on the reference.</li></ul>"
        )
        for s in shifts:
            parts.append(f"<h3>{_e(s)}</h3>" + forest(ledger.contributions, "variable", s))
        c = ledger.contributions
        rows = [[_e(r.variable), _e(r.scheme), _num(r.contribution), f"[{_num(r.lo)}, {_num(r.hi)}]",
                 _num(r["units helped"], 2), _num(r.selected, 2)] for _, r in c.iterrows()]  # fmt: skip
        parts.append("<h3>All numbers</h3>")
        parts.append(_table(["variable", "scheme", "skill lost when removed", f"{pct} interval",
                             "share of units helped", "how often selected"], rows, numeric=(2, 3, 4, 5)))  # fmt: skip

    # ---- equations
    parts.append("<h2>Equations, fitted to all rows</h2>")
    for name, m in ledger.models.items():
        parts.append(f"<h3>{_e(name)}</h3><pre>{_e(m.equation())}</pre>")

    # ---- pre-check
    if preflight is not None and preflight.warnings:
        parts.append(
            "<h2>What the data can support</h2><ul>"
            + "".join(f"<li>{_e(w)}</li>" for w in preflight.warnings)
            + "</ul>"
        )
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{_e(title)}</title>'
        f"<style>{_CSS}</style></head><body><main>{''.join(parts)}</main></body></html>"
    )
