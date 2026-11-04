"""
Follow-up report on two specific monkey claims about the focus set
(default Chicago Open 2026), from mentions.csv + questions.csv (count.py):

1. It has more monkey/ape mentions than any other Chicago Open.
2. Among sets from its season with at least --min-packets packets, it has
   the most monkey/ape mentions.

Sets are ranked by mentions per packet: the absolute number of hits a player
hears in a typical packet (packets are a fixed ~20 tossups + ~20 bonuses), which
still puts short and long sets on equal footing. Robustness checks: raw mentions,
questions with a mention per packet, and per packet after dropping the set's
single most monkey-heavy question.
Every underlying mention is listed so the numbers can be checked by hand.

Usage:
    python monkey_claims.py [--focus chicago-open-2026]
        [--season-start 2025-09-01] [--season-end 2026-08-31] [--min-packets 10]
"""

import argparse

import pandas as pd

import numpy as np

GROUP = "primates"


def with_mentions(q: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    per_q = m[m["group"] == GROUP].groupby("key").size()
    return q.assign(mon=q["key"].map(per_q).fillna(0))


def packet_counts(g: pd.DataFrame) -> np.ndarray:
    """Monkey mentions in each packet of a set (packets with none count as 0)."""
    return g.groupby("round")["mon"].sum().to_numpy(float)



def set_stats(q: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """One row per set: packets, mentions, questions, per-packet rates."""
    q = with_mentions(q, m)
    rows = []
    for s, g in q.groupby("set"):
        mc, pk = g["mon"].to_numpy(float), packet_counts(g)
        n = len(pk)
        rows.append({
            "set": s, "name": g["set_name"].iat[0], "date": g["date"].iat[0],
            "packets": n, "mentions": int(mc.sum()), "questions": int((mc > 0).sum()),
            "per_packet": mc.sum() / n,
            "q_per_packet": (mc > 0).sum() / n,
            "wo_top": (mc.sum() - mc.max()) / n,
        })
    return pd.DataFrame(rows).set_index("set")


def ranks(df: pd.DataFrame, focus: str) -> dict[str, int]:
    """Competition rank (ties share the better rank) of focus on each measure."""
    return {c: int((df[c] > df.loc[focus, c]).sum()) + 1
            for c in ("per_packet", "mentions", "q_per_packet", "wo_top")}


def table(df: pd.DataFrame, focus: str, with_date: bool = False) -> list[str]:
    head = "| # | Set |" + (" Date |" if with_date else "") + \
        " Packets | Mentions | **Per packet** | Questions (per packet) |" \
        " Per packet without top question |"
    L = [head, "|---:|---|" + ("---|" if with_date else "") + "---:|---:|---:|---:|---:|"]
    order = df.sort_values(["per_packet", "mentions"], ascending=False)
    for i, (s, r) in enumerate(order.iterrows(), 1):
        name = f"**{r['name']}**" if s == focus else r["name"]
        date = f" {r['date'] or '–'} |" if with_date else ""
        L.append(f"| {i} | {name} |{date} {r['packets']} | {r['mentions']} | "
                 f"**{r['per_packet']:.2f}** | "
                 f"{r['questions']} ({r['q_per_packet']:.2f}) | {r['wo_top']:.2f} |")
    return L


def rank_line(df: pd.DataFrame, focus: str) -> str:
    rk, n = ranks(df, focus), len(df)
    return (f"{df.loc[focus, 'name']} ranks **{rk['per_packet']} of {n}** on mentions per "
            f"packet. Robustness: **{rk['mentions']}** on raw mentions, "
            f"**{rk['q_per_packet']}** on questions with a mention per packet, "
            f"**{rk['wo_top']}** per packet without its top question.")


def mention_list(m: pd.DataFrame, s: str) -> list[str]:
    pm = m[(m["set"] == s) & (m["group"] == GROUP)]
    if pm.empty:
        return ["_No monkey/ape mentions._", ""]
    L = ["| Question | Word | Context |", "|---|---|---|"]
    for _, r in pm.sort_values(["kind", "round", "id"]).iterrows():
        ctx = str(r["context"]).replace("|", "/")
        L.append(f"| {r['kind']} {r['id']} | {r['match']} | …{ctx}… |")
    return L + [""]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focus", default="chicago-open-2026")
    ap.add_argument("--series", default="Chicago Open",
                    help="substring identifying the focus set's tournament series")
    ap.add_argument("--season-start", default="2025-09-01")
    ap.add_argument("--season-end", default="2026-08-31")
    ap.add_argument("--min-packets", type=int, default=10)
    ap.add_argument("--output", default="monkey_claims.md")
    args = ap.parse_args()

    q = pd.read_csv("questions.csv", keep_default_na=False)
    m = pd.read_csv("mentions.csv")
    for d in (q, m):
        d["key"] = d["set"] + "|" + d["kind"] + "|" + d["id"].astype(str)
    m["round"] = m["id"].astype(str).str.split("-").str[0].astype(int)

    st = set_stats(q, m)
    focus = args.focus
    fname = st.loc[focus, "name"]
    L = [f"# Monkey claims check: {fname}\n",
         "Monkey/ape mentions = the `primates` group (monkey, ape, simian, chimp, gorilla, "
         "orangutan, gibbon, baboon, macaque, lemur, singerie, Hanuman, Sun Wukong, …). "
         "Names and titles count (e.g. *Gibbons v. Ogden*, *Dream on Monkey Mountain*). "
         "**Without top question** drops each set's single most monkey-heavy question, so "
         "one themed bonus can't carry a set. Sets are ranked by **mentions per packet**. "
         "Counts are exact: they describe these sets as written, not a lasting tendency of their writers. "
         "Methodology: README.md.\n"]

    # ---- Claim 1: series by year
    series = st[st["name"].str.contains(args.series)].sort_values("date")
    others = series.drop(focus)
    L.append(f"## Claim 1: more monkeys than any other {args.series}\n")
    L += table(series, focus, with_date=True)
    L.append("")
    L.append(rank_line(series, focus))
    qm = with_mentions(q, m)
    fp = packet_counts(qm[qm["set"] == focus])
    op = np.concatenate([packet_counts(qm[qm["set"] == s]) for s in others.index])
    ratio = fp.mean() / op.mean()
    L.append("")
    L.append(f"Against the earlier {args.series}s pooled ({op.mean():.2f} per packet), "
             f"{fname} has **{ratio:.2f}×** the monkeys per packet.")
    top_all = all(v == 1 for v in ranks(series, focus).values())
    if top_all:
        verdict = (f"**Verdict: holds.** {fname} is first on every measure. "
                   "It is the most monkey-heavy edition so far.")
    else:
        verdict = "**Verdict: does not hold on every measure** (see ranks above)."
    L.append("")
    L.append(verdict)
    L.append("")
    for s, r in series.iterrows():
        L.append(f"### {r['name']} ({r['mentions']} mentions in {r['questions']} questions)\n")
        L += mention_list(m, s)

    # ---- Claim 2: season with packet minimum
    in_season = st[(st["date"] >= args.season_start) & (st["date"] <= args.season_end)]
    no_date = st[st["date"] == ""]
    eligible = in_season[in_season["packets"] >= args.min_packets]
    excluded = pd.concat([in_season[in_season["packets"] < args.min_packets], no_date])
    L.append(f"## Claim 2: most monkeys of the {args.season_start[:4]}–"
             f"{args.season_end[:4]} season (≥ {args.min_packets} packets)\n")
    L.append(f"Season = tournament date from {args.season_start} to {args.season_end} "
             f"(fall through the summer Chicago Open; {args.series} "
             f"{int(args.season_start[:4])} falls in the previous season). Packets = distinct "
             f"rounds with question text on buzzpoints.\n")
    L.append("Not eligible:\n")
    for s, r in excluded.iterrows():
        why = []
        if r["packets"] < args.min_packets:
            why.append(f"{r['packets']} packets")
        if not r["date"]:
            why.append("no tournament date on buzzpoints")
        L.append(f"- {r['name']}: {', '.join(why)} ({r['mentions']} mentions, "
                 f"{r['per_packet']:.2f} per packet)")
    below_all = st[(st["packets"] < args.min_packets) & ~st.index.isin(excluded.index)]
    for s, r in below_all.iterrows():
        L.append(f"- (outside the season, but also under {args.min_packets} packets: "
                 f"{r['name']}, {r['packets']} packets)")
    L.append("")
    L += table(eligible, focus, with_date=True)
    L.append("")
    L.append(rank_line(eligible, focus))
    L.append("")
    rk = ranks(eligible, focus)
    if rk["per_packet"] == 1:
        weak = [k for k, v in rk.items() if v != 1]
        L.append(f"**Verdict: holds.** {fname} leads the eligible sets on mentions per packet"
                 + (" and every robustness check." if not weak else
                    ", though not on every robustness check (see ranks above)."))
    else:
        ahead = eligible[eligible["per_packet"] > eligible.loc[focus, "per_packet"]]
        names = ", ".join(f"{r['name']} ({r['per_packet']:.2f})" for _, r in
                          ahead.sort_values("per_packet", ascending=False).iterrows())
        L.append(f"**Verdict: does not hold.** Eligible sets ahead of {fname} on mentions per "
                 f"packet: {names}.")
        # Would nudging the season window change the answer?
        edge = pd.Timedelta(days=60)
        inner = all(pd.Timestamp(args.season_start) + edge <= pd.Timestamp(d)
                    <= pd.Timestamp(args.season_end) - edge for d in ahead["date"])
        if inner:
            L.append(f"They were all played at least two months inside the season, so moving "
                     f"the season boundaries doesn't change this.")
        if not any(v == 1 for v in rk.values()):
            L.append(f"No measure puts {fname} first.")
        L.append("")
        for s, r in ahead.sort_values("per_packet", ascending=False).iterrows():
            L.append(f"### {r['name']} ({r['mentions']} mentions in {r['questions']} questions)\n")
            L += mention_list(m, s)

    # ---- All sets, same packet minimum (context)
    big = st[st["packets"] >= args.min_packets]
    L.append(f"## Context: all {len(big)} sets with ≥ {args.min_packets} packets\n")
    L += table(big, focus, with_date=True)
    L.append("")
    L.append(rank_line(big, focus))
    L.append("")

    open(args.output, "w", encoding="utf-8").write("\n".join(L))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
