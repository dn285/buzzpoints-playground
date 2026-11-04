"""
Turn mentions.csv + questions.csv (from count.py) into report.md.

Metrics, per set and animal group:
- rate      = mentions per 10,000 words (question text + primary answers)
- q_share   = % of questions with at least one mention of the group

Usage:
    python report.py [--focus chicago-open-2026] [--output report.md]
"""

import argparse

import numpy as np
import pandas as pd

from lexicon import GROUPS

ALL = "all animals"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focus", default="chicago-open-2026")
    ap.add_argument("--output", default="report.md")
    args = ap.parse_args()

    q = pd.read_csv("questions.csv")
    m = pd.read_csv("mentions.csv")
    q["key"] = q["set"] + "|" + q["kind"] + "|" + q["id"].astype(str)
    m["key"] = m["set"] + "|" + m["kind"] + "|" + m["id"].astype(str)

    # Question x group mention counts.
    groups = list(GROUPS)
    wide = m.pivot_table(index="key", columns="group", values="lemma",
                         aggfunc="size", fill_value=0)
    q = q.join(wide.reindex(columns=groups, fill_value=0), on="key").fillna(0)
    q[ALL] = q[groups].sum(axis=1)
    cols = [ALL] + groups
    names = q.drop_duplicates("set").set_index("set")["set_name"]
    focus_name = names[args.focus]

    # Per-set rates.
    rows = []
    for s, g in q.groupby("set"):
        w = g["words"].to_numpy(float)
        row = {"set": s, "name": names[s], "questions": len(g), "words": int(w.sum())}
        for c in cols:
            mc = g[c].to_numpy(float)
            row[c] = mc.sum() / w.sum() * 1e4
            row[c + " n"] = int(mc.sum())
            row[c + " q%"] = (mc > 0).mean() * 100
        rows.append(row)
    per_set = pd.DataFrame(rows).set_index("set")
    per_set.to_csv("per_set.csv")

    # Focus set vs. everyone else, per group.
    f, o = q[q["set"] == args.focus], q[q["set"] != args.focus]
    fw, ow = f["words"].to_numpy(float), o["words"].to_numpy(float)
    comp = []
    for c in cols:
        fm, om = f[c].to_numpy(float), o[c].to_numpy(float)
        fr, orate = fm.sum() / fw.sum() * 1e4, om.sum() / ow.sum() * 1e4
        rank = int((per_set[c] > per_set.loc[args.focus, c]).sum()) + 1
        comp.append({"group": c, "focus n": int(fm.sum()), "focus rate": fr,
                     "others rate": orate, "ratio": fr / orate if orate else np.nan,
                     "rank": rank})
    comp = pd.DataFrame(comp)

    # ---------------------------------------------------------------- report
    n_sets = len(per_set)
    L = []
    L.append(f"# Animal mentions: is {focus_name} unusually zoological?\n")
    L.append(f"{n_sets} public college/open sets from buzzpoints.buzz, "
             f"{len(q):,} questions (tossups + bonuses, repeats removed), "
             f"{int(q['words'].sum()):,} words, {int(q[ALL].sum()):,} animal mentions. "
             "Rates are mentions per 10,000 words of question text + primary answers. "
             "Counts are exact: they describe these sets as written, not a lasting tendency of their writers. "
             "Names, titles and idioms count (Stephen Crane, Swan Lake, Jim Crow); see README "
             "for what is excluded and why.\n")

    def headline(c, label):
        r = comp.set_index("group").loc[c]
        return (f"- **{label}:** {r['focus rate']:.2f} per 10k words "
                f"({int(r['focus n'])} mentions) vs {r['others rate']:.2f} in the other sets "
                f"→ **{r['ratio']:.2f}×**, "
                f"rank **{int(r['rank'])} of {n_sets}**.")
    L.append("## Headline\n")
    L.append(headline("primates", "Monkeys & apes"))
    L.append(headline(ALL, "All animals"))
    L.append("")

    L.append(f"## {focus_name} vs. the other {n_sets - 1} sets, by group\n")
    L.append("Ratio > 1 means more mentions per word than the pooled other sets.\n")
    L.append("| Group | Mentions | Rate | Others | Ratio | Rank |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for _, r in comp.sort_values("ratio", ascending=False).iterrows():
        ratio = f"{r['ratio']:.2f}" if r["others rate"] else "–"
        L.append(f"| {r['group']} | {int(r['focus n'])} | {r['focus rate']:.2f} | "
                 f"{r['others rate']:.2f} | {ratio} | {int(r['rank'])} |")
    L.append("")

    for c, title in (("primates", "Monkey & ape leaderboard"), (ALL, "All-animal leaderboard")):
        L.append(f"## {title}\n")
        L.append("| # | Set | Mentions | Per 10k words | % of questions |")
        L.append("|---:|---|---:|---:|---:|")
        for i, (s, r) in enumerate(per_set.sort_values(c, ascending=False).iterrows(), 1):
            name = f"**{r['name']}**" if s == args.focus else r["name"]
            L.append(f"| {i} | {name} | {r[c + ' n']} | {r[c]:.2f} | "
                     f"{r[c + ' q%']:.1f}% |")
        L.append("")

    co = per_set[per_set["name"].str.contains("Chicago Open")].sort_values("name")
    L.append("## Chicago Open by year\n")
    L.append("| Set | Words | Monkeys & apes /10k | All animals /10k |")
    L.append("|---|---:|---:|---:|")
    for s, r in co.iterrows():
        L.append(f"| {r['name']} | {r['words']:,} | {r['primates']:.2f} | {r[ALL]:.2f} |")
    L.append("")

    fm = m[(m["set"] == args.focus) & (m["group"] == "primates")]
    L.append(f"## Every monkey/ape mention in {focus_name}\n")
    L.append("| Question | Word | Context |")
    L.append("|---|---|---|")
    for _, r in fm.sort_values(["kind", "id"]).iterrows():
        ctx = str(r["context"]).replace("|", "/")
        L.append(f"| {r['kind']} {r['id']} | {r['match']} | …{ctx}… |")
    L.append("")

    myth = m.groupby("set")["mythical"].mean() * 100
    L.append("## Mythical creatures\n")
    L.append(f"{int(m['mythical'].sum())} of {len(m):,} mentions ({m['mythical'].mean() * 100:.1f}%) "
             f"are mythical creatures (counted toward their base animal); "
             f"{myth[args.focus]:.1f}% in {focus_name}.\n")

    open(args.output, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:12]))
    print(f"\nWrote {args.output} and per_set.csv")


if __name__ == "__main__":
    main()
