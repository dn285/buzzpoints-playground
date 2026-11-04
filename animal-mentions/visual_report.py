"""
Visual follow-up: how monkey/ape mentions are distributed inside sets, and how
"perceived" monkey density changes once clustered mentions are collapsed.

Reads mentions.csv + questions.csv (count.py) and writes monkey_visuals.html,
a self-contained page with inline SVG:

1. Sparklines per set: mentions per packet and monkey questions per packet
   (one shared scale per column, so bar heights compare across sets).
2. Heat grids for the season's top sets: packet x question slot, read order
   T1 B1 T2 B2 ..., so clusters are visible.
3. A cooldown curve: hits per packet when a monkey question only counts if at
   least k other questions were read since the last counted one (within a
   packet). k = 0 is "monkey questions"; the ends are "every mention" and
   "packets with any monkey". Motivated by Hintzman (1969): judged frequency
   rises with the spacing of repetitions and levels off after 3-4 intervening
   items.

Usage:
    python visual_report.py [--focus chicago-open-2026]
        [--season-start 2025-09-01] [--season-end 2026-08-31] [--min-packets 10]
"""

import argparse
import html
import json

import pandas as pd

from monkey_claims import set_stats, with_mentions

COOLDOWNS = [1, 2, 3, 4]  # intervening questions required between counted hits
LEVELS = (["Every mention", "Monkey questions"]
          + [f"{k} between" for k in COOLDOWNS] + ["Any in packet"])


def load(args):
    q = pd.read_csv("questions.csv", keep_default_na=False)
    m = pd.read_csv("mentions.csv")
    for d in (q, m):
        d["key"] = d["set"] + "|" + d["kind"] + "|" + d["id"].astype(str)
    q = with_mentions(q, m)
    q["mon"] = q["mon"].astype(int)
    q["round"] = q["round"].astype(int)
    q["num"] = q["id"].astype(str).str.split("-").str[1].astype(int)
    # Read order within a packet: tossup n, then bonus n.
    q["slot"] = (q["num"] - 1) * 2 + (q["kind"] == "bonus").astype(int)
    m["round"] = m["id"].astype(str).str.split("-").str[0].astype(int)
    return q, m


def cooldown_hits(g: pd.DataFrame, k: int) -> int:
    """Monkey questions counted only when >= k questions separate them (per packet)."""
    hits = 0
    for _, p in g[g["mon"] > 0].groupby("round"):
        last = None
        for s in sorted(p["slot"]):
            if last is None or s - last > k:
                hits += 1
                last = s
    return hits


def profile(g: pd.DataFrame) -> dict:
    rounds = sorted(g["round"].unique())
    by = g.groupby("round")
    ment = [int(by.get_group(r)["mon"].sum()) for r in rounds]
    ques = [int((by.get_group(r)["mon"] > 0).sum()) for r in rounds]
    n = len(rounds)
    levels = ([sum(ment) / n, sum(ques) / n]
              + [cooldown_hits(g, k) / n for k in COOLDOWNS]
              + [sum(1 for v in ment if v) / n])
    return {"rounds": rounds, "ment": ment, "ques": ques, "levels": levels}


# ------------------------------------------------------------------ SVG bits
def esc(s) -> str:
    return html.escape(str(s), quote=True)


def sparkline(values, vmax, rounds, unit, other=None, other_unit=None) -> str:
    bw, gap, h = 6, 2, 30
    w = len(values) * (bw + gap) - gap
    parts = [f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
             f'role="img" aria-label="{esc(unit)} per packet: {esc(", ".join(map(str, values)))}">']
    for i, v in enumerate(values):
        x = i * (bw + gap)
        tip = f"Packet {rounds[i]}: {v} {unit if v != 1 else unit.rstrip('s')}"
        if other is not None:
            o = other[i]
            tip += f" ({o} {other_unit if o != 1 else other_unit.rstrip('s')})"
        if v:
            bh = max(3, v / vmax * (h - 2))
            parts.append(f'<rect class="bar" x="{x}" y="{h - bh:.1f}" width="{bw}" '
                         f'height="{bh:.1f}" rx="1.5" data-tip="{esc(tip)}"/>')
        else:
            parts.append(f'<rect class="zero" x="{x}" y="{h - 1.5}" width="{bw}" height="1.5" '
                         f'data-tip="{esc(tip)}"/>')
    parts.append("</svg>")
    return "".join(parts)


def heat_grid(g: pd.DataFrame, m_set: pd.DataFrame) -> str:
    rounds = sorted(g["round"].unique())
    nslots = int(g["slot"].max()) + 1
    cell, gap, left, top = 10, 2, 26, 16
    w = left + nslots * (cell + gap)
    h = top + len(rounds) * (cell + gap)
    words = m_set.groupby(["kind", "id"])["match"].apply(lambda s: ", ".join(s)).to_dict()
    out = [f'<svg class="grid" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
           f'aria-label="Monkey mentions by packet and question slot">']
    for t in range(0, nslots // 2, 5):  # tossup-number ticks every 5
        x = left + (t * 2) * (cell + gap)
        out.append(f'<text class="tick" x="{x}" y="{top - 5}">T{t + 1}</text>')
    lookup = g.set_index(["round", "slot"])
    for ri, r in enumerate(rounds):
        y = top + ri * (cell + gap)
        out.append(f'<text class="tick" x="{left - 5}" y="{y + cell - 1}" '
                   f'text-anchor="end">{r}</text>')
        for s in range(nslots):
            x = left + s * (cell + gap)
            if (r, s) not in lookup.index:
                continue  # slot not in this packet (e.g. no tiebreaker)
            row = lookup.loc[(r, s)]
            v = int(row["mon"])
            label = f"{'Tossup' if s % 2 == 0 else 'Bonus'} {r}-{s // 2 + 1}"
            if v:
                lvl = 1 if v == 1 else 2 if v == 2 else 3 if v <= 4 else 4
                hit = words.get((row["kind"], row["id"]), "")
                tip = f"{label}: {v} mention{'s' if v > 1 else ''} ({hit})"
                out.append(f'<rect class="h{lvl}" x="{x}" y="{y}" width="{cell}" height="{cell}" '
                           f'rx="2" data-tip="{esc(tip)}"/>')
            else:
                out.append(f'<rect class="h0" x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2"/>')
    out.append("</svg>")
    return "".join(out)


def cooldown_chart(curves: list[dict]) -> str:
    """curves: {name, values, cls (series class or None for muted)}."""
    W, H = 720, 340
    ml, mr, mt, mb = 44, 150, 16, 46
    pw, ph = W - ml - mr, H - mt - mb
    ymax = max(max(c["values"]) for c in curves)
    ymax = (int(ymax * 4) + 1) / 4  # round up to 0.25
    x = lambda i: ml + i * pw / (len(LEVELS) - 1)
    y = lambda v: mt + ph - v / ymax * ph
    out = [f'<svg class="cool" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Perceived monkey hits per packet at each cooldown level">']
    t = 0.0
    while t <= ymax + 1e-9:
        out.append(f'<line class="gridline" x1="{ml}" x2="{ml + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        out.append(f'<text class="tick" x="{ml - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
        t += 0.25
    for i, lab in enumerate(LEVELS):
        out.append(f'<text class="tick" x="{x(i):.1f}" y="{mt + ph + 18}" text-anchor="middle">{esc(lab)}</text>')
    out.append(f'<text class="axis-note" x="{ml + pw / 2:.1f}" y="{H - 6}" text-anchor="middle">'
               f'stricter cooldown &#8594;</text>')
    # Muted series first, highlighted on top.
    for c in sorted(curves, key=lambda c: c["cls"] is not None):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(c["values"]))
        cls = c["cls"] or "muted"
        out.append(f'<polyline class="line {cls}" points="{pts}"/>')
        for i, v in enumerate(c["values"]):
            tip = f"{c['name']} · {LEVELS[i]}: {v:.2f} per packet"
            if c["cls"]:
                out.append(f'<circle class="pt {cls}" cx="{x(i):.1f}" cy="{y(v):.1f}" r="4"/>')
            out.append(f'<circle class="hit" cx="{x(i):.1f}" cy="{y(v):.1f}" r="9" data-tip="{esc(tip)}"/>')
    # Direct labels at the right end, dodged so they never overlap.
    labs = sorted([c for c in curves if c["cls"]], key=lambda c: -c["values"][-1])
    ys, prev = [], None
    for c in labs:
        ly = y(c["values"][-1]) + 4
        if prev is not None and ly < prev + 14:
            ly = prev + 14
        ys.append(ly)
        prev = ly
    for c, ly in zip(labs, ys):
        out.append(f'<text class="dlabel {c["cls"]}-text" x="{ml + pw + 10}" y="{ly:.1f}">'
                   f'{esc(c["name"])} {c["values"][-1]:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------ page
CSS = """
/* Layout: one reading column; charts get their own scroll boxes when wide. */
:root {
  --bg: #f6f7f4; --surface: #ffffff; --fg: #191c19; --muted: #5b615b; --line: #dde2dc;
  --band: #eaf1fb; --zero: #c9cfc8; --cell: #edf0ec;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --gray: #b9beb8;
  --h1: #9ec5f4; --h2: #5598e7; --h3: #256abf; --h4: #0d366b;
  --display: "Bricolage Grotesque", "Segoe UI", system-ui, sans-serif;
  --body: "Public Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141714; --surface: #1b1f1b; --fg: #eef1ed; --muted: #a3aaa2; --line: #2d332d;
  --band: #1b2a3d; --zero: #454c45; --cell: #232823;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --gray: #4d544c;
  --h1: #1c5cab; --h2: #3987e5; --h3: #86b6ef; --h4: #cde2fb; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #141714; --surface: #1b1f1b; --fg: #eef1ed; --muted: #a3aaa2; --line: #2d332d;
  --band: #1b2a3d; --zero: #454c45; --cell: #232823;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --gray: #4d544c;
  --h1: #1c5cab; --h2: #3987e5; --h3: #86b6ef; --h4: #cde2fb; color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body); }
.wrap { max-width: 980px; margin: 0 auto; padding-inline: 20px; padding-block: 36px 64px;
  display: grid; gap: 44px; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; line-height: 1.15; margin: 0; }
h1 { font-size: clamp(30px, 5vw, 44px); font-weight: 700; letter-spacing: -0.01em; }
h2 { font-size: 24px; font-weight: 650; }
h3 { font-size: 16px; font-weight: 650; }
p { margin: 0; max-width: 68ch; }
section { display: grid; gap: 14px; min-width: 0; }
.eyebrow { font: 600 12px/1 var(--mono); letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); }
.lede { font-size: 17px; }
.muted { color: var(--muted); }
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; min-width: 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th { text-align: left; font: 600 11px/1.3 var(--mono); letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--muted); padding: 6px 10px; border-bottom: 1px solid var(--line); }
td { padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: middle; }
td.r, th.r { text-align: right; }
tr.focus td { background: var(--band); }
tr.focus td:first-child { font-weight: 650; }
tr.group td { font: 600 12px/1 var(--mono); letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--muted); padding-top: 18px; border-bottom: none; background: transparent; }
.spark .bar { fill: var(--s1); }
.spark .zero { fill: var(--zero); }
.grid .h0 { fill: var(--cell); }
.grid .h1 { fill: var(--h1); } .grid .h2 { fill: var(--h2); }
.grid .h3 { fill: var(--h3); } .grid .h4 { fill: var(--h4); }
.tick { font: 10px var(--mono); fill: var(--muted); }
.axis-note { font: 11px var(--mono); fill: var(--muted); }
.gridline { stroke: var(--line); stroke-width: 1; }
.cool { width: 100%; min-width: 620px; height: auto; display: block; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.line.muted { stroke: var(--gray); stroke-width: 1.25; }
.line.s1, .pt.s1 { stroke: var(--s1); } .pt.s1 { fill: var(--s1); }
.line.s2, .pt.s2 { stroke: var(--s2); } .pt.s2 { fill: var(--s2); }
.line.s3, .pt.s3 { stroke: var(--s3); } .pt.s3 { fill: var(--s3); }
.line.s4, .pt.s4 { stroke: var(--s4); } .pt.s4 { fill: var(--s4); }
.line.s1 { stroke-width: 3; }
.pt { stroke: var(--surface); stroke-width: 2; }
.hit { fill: transparent; cursor: default; }
.dlabel { font: 600 12px var(--body); fill: var(--fg); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.key { display: inline-flex; gap: 4px; align-items: center; font: 11px var(--mono); color: var(--muted); }
.key i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.grids { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr)); gap: 22px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px;
  padding: 14px 16px; display: grid; gap: 8px; min-width: 0; }
.lit { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 290px), 1fr)); }
.lit .card p { font-size: 14px; }
.cite { font-size: 12.5px; color: var(--muted); }
.cite a { color: inherit; }
details summary { cursor: pointer; font-weight: 600; }
details summary:focus-visible, a:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; }
#tip { position: fixed; pointer-events: none; background: var(--fg); color: var(--bg);
  font: 12px/1.4 var(--body); padding: 6px 9px; border-radius: 6px; max-width: 300px;
  z-index: 10; }
"""

JS = """
const tip = document.getElementById('tip');
document.addEventListener('pointerover', e => {
  const t = e.target.closest('[data-tip]');
  if (!t) { tip.hidden = true; return; }
  tip.textContent = t.getAttribute('data-tip'); tip.hidden = false;
});
document.addEventListener('pointermove', e => {
  if (tip.hidden) return;
  const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + 'px'; tip.style.top = (e.clientY + 16) + 'px';
});
document.addEventListener('pointerout', e => { if (!e.relatedTarget?.closest?.('[data-tip]')) tip.hidden = true; });
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focus", default="chicago-open-2026")
    ap.add_argument("--series", default="Chicago Open")
    ap.add_argument("--season-start", default="2025-09-01")
    ap.add_argument("--season-end", default="2026-08-31")
    ap.add_argument("--min-packets", type=int, default=10)
    ap.add_argument("--output", default="monkey_visuals.html")
    args = ap.parse_args()

    q, m = load(args)
    st = set_stats(q, m.assign(round=m["round"]))
    pm = m[m["group"] == "primates"]
    big = st[st["packets"] >= args.min_packets]
    season = big[(big["date"] >= args.season_start) & (big["date"] <= args.season_end)]
    series = big[big["name"].str.contains(args.series) & ~big.index.isin(season.index)]
    rest = big[~big.index.isin(season.index) & ~big.index.isin(series.index)]
    prof = {s: profile(q[q["set"] == s]) for s in big.index}
    focus, fname = args.focus, st.loc[args.focus, "name"]

    vmax_m = max(max(p["ment"]) for p in prof.values())
    vmax_q = max(max(p["ques"]) for p in prof.values())

    def rows(df, title):
        out = [f'<tr class="group"><td colspan="5">{esc(title)}</td></tr>']
        for s in df.sort_values("per_packet", ascending=False).index:
            p, r = prof[s], st.loc[s]
            cls = ' class="focus"' if s == focus else ""
            out.append(
                f'<tr{cls}><td>{esc(r["name"])}</td>'
                f'<td>{sparkline(p["ment"], vmax_m, p["rounds"], "mentions", p["ques"], "questions")}</td>'
                f'<td class="r num">{r["per_packet"]:.2f}</td>'
                f'<td>{sparkline(p["ques"], vmax_q, p["rounds"], "monkey questions")}</td>'
                f'<td class="r num">{r["q_per_packet"]:.2f}</td></tr>')
        return out

    spark_table = ['<div class="scroll"><table><thead><tr><th>Set</th>'
                   '<th>Mentions, packet by packet</th><th class="r">Per packet</th>'
                   '<th>Monkey questions, packet by packet</th><th class="r">Per packet</th>'
                   '</tr></thead><tbody>']
    spark_table += rows(season, f"{args.season_start[:4]}–{args.season_end[2:4]} season")
    spark_table += rows(series, f"Earlier {args.series}s")
    spark_table.append("</tbody></table></div>")
    rest_table = ['<div class="scroll"><table><tbody>'] + rows(rest, "All other sets") + \
        ["</tbody></table></div>"]

    # Season top four by mentions per packet get the heat grids and the colors.
    top = list(season.sort_values("per_packet", ascending=False).index[:4])
    if focus not in top:
        top = [focus] + top[:3]
    top = [focus] + [s for s in top if s != focus]
    colors = {s: f"s{i + 1}" for i, s in enumerate(top)}

    grids = []
    for s in top:
        r = st.loc[s]
        grids.append(f'<div class="card"><h3>{esc(r["name"])}</h3>'
                     f'<p class="muted num" style="font-size:13px">{r["mentions"]} mentions in '
                     f'{r["questions"]} questions across {r["packets"]} packets</p>'
                     f'<div class="scroll">{heat_grid(q[q["set"] == s], pm[pm["set"] == s])}</div></div>')

    curves = [{"name": st.loc[s, "name"], "values": prof[s]["levels"],
               "cls": colors.get(s)} for s in season.index]
    legend = "".join(f'<span><i class="sw" style="background:var(--{colors[s]})"></i>'
                     f'{esc(st.loc[s, "name"])}</span>' for s in top)
    legend += '<span><i class="sw" style="background:var(--gray)"></i>Other season sets</span>'

    # Rank table: focus rank in season, and among its series, at each level.
    def rank_at(i, pool) -> tuple[int, bool]:
        """Competition rank of the focus set at level i, and whether it is tied."""
        v = prof[focus]["levels"][i]
        tied = any(abs(prof[s]["levels"][i] - v) < 1e-12 for s in pool if s != focus)
        return 1 + sum(1 for s in pool if prof[s]["levels"][i] > v + 1e-12), tied

    def fmt_rank(rt, n=None) -> str:
        r, tied = rt
        return f"{'=' if tied else ''}{r}" + (f" of {n}" if n else "")
    series_pool = [s for s in big.index if args.series in st.loc[s, "name"]]
    rank_rows = []
    for i, lab in enumerate(LEVELS):
        cells = "".join(f'<td class="r num">{prof[s]["levels"][i]:.2f}</td>' for s in top)
        rank_rows.append(f'<tr><td>{lab}</td>{cells}'
                         f'<td class="r num">{fmt_rank(rank_at(i, season.index), len(season))}</td>'
                         f'<td class="r num">{fmt_rank(rank_at(i, series_pool), len(series_pool))}</td></tr>')
    rank_head = "".join(f'<th class="r">{esc(st.loc[s, "name"])}</th>' for s in top)

    # Headline numbers for the lede.
    lv = {s: prof[s]["levels"] for s in season.index}
    def leader(i) -> str:
        top_v = max(v[i] for v in lv.values())
        names = [st.loc[s, "name"] for s in lv if abs(lv[s][i] - top_v) < 1e-12]
        return " and ".join(names) + (" (tied)" if len(names) > 1 else "")
    leaders = [leader(i) for i in range(len(LEVELS))]
    summary = f"On every mention, {esc(leaders[0])} leads the season."
    if len(set(leaders[1:])) == 1:
        summary += (f" Once each question counts once, {esc(leaders[1])} leads, and it stays "
                    "ahead at every cooldown.")
    else:
        summary += " The leader changes with the cooldown: " + ", ".join(
            f"{LEVELS[i].lower()}: {esc(leaders[i])}" for i in range(1, len(LEVELS))) + "."
    series_ranks = [rank_at(i, series_pool) for i in range(len(LEVELS))]
    if all(r == 1 and not t for r, t in series_ranks):
        summary += f" {esc(fname)} is first among {esc(args.series)}s at every level."
    elif all(r == 1 for r, _ in series_ranks):
        tied_at = [LEVELS[i].lower() for i, (_, t) in enumerate(series_ranks) if t]
        summary += (f" {esc(fname)} is first among {esc(args.series)}s at every level, "
                    f"tied at {', '.join(tied_at)}.")
    else:
        summary += (f" Among {esc(args.series)}s, {esc(fname)} ranks "
                    + ", ".join(fmt_rank(rt) for rt in series_ranks) + " across the levels.")

    page = f"""<meta charset="utf-8">
<title>Monkey Density</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=IBM+Plex+Mono:wght@400;600&family=Public+Sans:wght@400;600&display=swap">
<style>{CSS}</style>
<div id="tip" hidden></div>
<main class="wrap">
<header style="display:grid;gap:14px">
  <span class="eyebrow">Quizbowl · primates · buzzpoints.buzz</span>
  <h1>How monkeyful does {esc(fname)} feel?</h1>
  <p class="lede">Raw counts treat a six-monkey singerie bonus as six monkeys. Players probably
  remember it as one monkey question. This page shows where the monkeys sit inside each set
  and re-scores the {args.season_start[:4]}–{args.season_end[2:4]} season with clustered
  mentions collapsed.</p>
  <p class="muted">{summary}</p>
</header>

<section>
  <span class="eyebrow">1 · Distribution</span>
  <h2>Packet by packet</h2>
  <p>Each bar is one packet, in round order. The left column counts every monkey word; the right
  counts questions with at least one. Bars share one scale per column (tallest mentions bar = {vmax_m},
  tallest questions bar = {vmax_q}). A flat stub is a packet with none. Sets with
  &#8805; {args.min_packets} packets only.</p>
  {''.join(spark_table)}
  <details><summary>All other sets ({len(rest)})</summary>{''.join(rest_table)}</details>
</section>

<section>
  <span class="eyebrow">2 · Clustering</span>
  <h2>Where in the packet</h2>
  <p>Rows are packets, columns are questions in read order (tossup 1, bonus 1, tossup 2, …).
  Darker cells hold more monkey words. A single dark cell is a monkey-themed question; scattered
  light cells are monkeys spread through the set.</p>
  <div class="legend"><span class="key"><i style="background:var(--h1)"></i>1</span>
  <span class="key"><i style="background:var(--h2)"></i>2</span>
  <span class="key"><i style="background:var(--h3)"></i>3–4</span>
  <span class="key"><i style="background:var(--h4)"></i>5+ mentions</span></div>
  <div class="grids">{''.join(grids)}</div>
</section>

<section>
  <span class="eyebrow">3 · Perceived density</span>
  <h2>Hits per packet under a cooldown</h2>
  <p>Moving right, more clustered monkeys merge into one hit. <b>Monkey questions</b> counts each
  question once. <b>k between</b> counts a monkey question only if at least k other questions
  were read since the last counted one in that packet. <b>Any in packet</b> asks only whether a
  packet had a monkey at all. Hintzman (1969) found judged frequency stops growing with spacing
  after about 3–4 intervening items, so the 3–4 range is the most defensible guess at a cooldown.</p>
  <div class="legend">{legend}</div>
  <div class="scroll">{cooldown_chart(curves)}</div>
  <div class="scroll"><table><thead><tr><th>Level</th>{rank_head}
  <th class="r">{esc(fname)} rank, season</th><th class="r">Rank among {esc(args.series)}s</th>
  </tr></thead><tbody>{''.join(rank_rows)}</tbody></table></div>
</section>

<section>
  <span class="eyebrow">4 · What the research says</span>
  <h2>Why clustering should matter</h2>
  <p>No study looks at quizbowl, so these are analogies from frequency-judgment research. They
  support collapsing clustered mentions, but none of them sets the exact cooldown.</p>
  <div class="lit">
    <div class="card"><h3>Spaced repeats feel more frequent</h3>
      <p>Judged frequency of a repeated item rises with the spacing between its repetitions and
      levels off after about three or four intervening items. Massed repeats are judged as rarer
      than the same number of spaced ones.</p>
      <p class="cite">Hintzman (1969), <i>Apparent frequency as a function of frequency and the
      spacing of repetitions</i>, J. Exp. Psych. 80, 139–145.
      <a href="https://www.researchgate.net/publication/232552137_Memory_for_the_spacing_of_repetitions">Related review</a></p>
      <p><b>Use:</b> collapse mentions inside one question, and probably adjacent ones. This is
      the cooldown curve's basis.</p></div>
    <div class="card"><h3>Frequency is judged by ease of recall</h3>
      <p>People estimate how often something happened by how easily examples come to mind
      (the availability heuristic). Vivid, distinctive examples are overweighted.</p>
      <p class="cite">Tversky &amp; Kahneman (1973), <i>Availability: A heuristic for judging
      frequency and probability</i>, Cognitive Psychology 5.</p>
      <p><b>Use:</b> a memorable whole-question monkey (singerie, baboons) may count for more
      than its share of hits, which cuts against pure collapsing.</p></div>
    <div class="card"><h3>Memory stores events, not words</h3>
      <p>People segment experience into discrete events and remember those units. A question is a
      natural event boundary in a quizbowl game.</p>
      <p class="cite">Zacks, Speer, Swallow, Braver &amp; Reynolds (2007), <i>Event perception: a
      mind–brain perspective</i>, Psychological Bulletin 133.</p>
      <p><b>Use:</b> supports "monkey questions" as the base unit over raw word counts.</p></div>
    <div class="card"><h3>Noticing primes noticing</h3>
      <p>After you start noticing a thing, it seems to turn up everywhere (the frequency
      illusion). Selective attention, not real frequency, drives the impression.</p>
      <p class="cite">Zwicky (2005), coined "frequency illusion", Language Log.</p>
      <p><b>Use:</b> a reason to trust these counts over the impression that started the
      question.</p></div>
  </div>
</section>

<section>
  <span class="eyebrow">Method</span>
  <h2>How this was counted</h2>
  <p>Monkey/ape mentions are the <span class="num">primates</span> word list from the
  animal-mentions analysis (monkey, ape, simian, gorilla, baboon, macaque, singerie, Hanuman, …);
  names and titles count. Packets are distinct rounds with question text on buzzpoints.buzz.
  Read order assumes tossup n is followed by bonus n. Cooldowns reset at each packet, since
  packets are separate games. Counts are exact for these sets as written. They don't account
  for tossups that ended on an early buzz before a late monkey clue was read.</p>
</section>
</main>
<script>{JS}</script>
"""
    open(args.output, "w", encoding="utf-8").write(page)
    print(f"Wrote {args.output}")
    for s in top:
        print(f"  {st.loc[s, 'name']:28}", " ".join(f"{v:.2f}" for v in prof[s]["levels"]))


if __name__ == "__main__":
    main()
