"""
Tossup quality analysis — empirical shape metrics per tossup, plus a coarse editorial
classification. Strictly downstream of the survival-model fit (reuses its `d_t`).

Usage:
    python model.py --output-dir results-1
    python model.py --raw ../buzzpoints_data.csv \
                    --surv-dir ../survival-model/results-1 \
                    --output-dir results-1 \
                    --power-mark 75

See HANDOFF.md for the design rationale.
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats


CATEGORY_MAP = {
    "Fine Arts": "Fine Arts",
    "Science": "Science",
    "History": "History",
    "Literature": "Literature",
    "Religion": "Beliefs",
    "Mythology": "Beliefs",
    "Philosophy": "Thought",
    "Social Science": "Thought",
    "Geography": "Other",
    "Current Events": "Other",
    "Other Academic": "Other",
    "Trash": "Other",
}


def aggregate_category(raw: str) -> str:
    raw = raw.strip()
    for prefix, top in CATEGORY_MAP.items():
        if raw.startswith(prefix):
            return top
    return "Other"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["won"] = df["won"].map({"True": True, "False": False})
    for c in ("buzzpoint", "buzz_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["category_agg"] = df["category"].apply(aggregate_category)
    return df


def compute_tossup_lengths(df: pd.DataFrame) -> pd.Series:
    """Length proxy = max observed buzzpoint per tossup (across games)."""
    return df.groupby("tossup_slug")["buzzpoint"].max().rename("length")


def per_tossup_metrics(df: pd.DataFrame, surv_diff: pd.DataFrame,
                       power_mark: int = 75, n_heard_min: int = 3,
                       n_bins: int = 10, pl_skill: pd.Series | None = None) -> pd.DataFrame:
    """One row per tossup with all metrics + an editorial label."""
    lengths = compute_tossup_lengths(df)
    correct = df[(df["buzz_value"] == 10) & df["buzzpoint"].notna()].copy()
    diff_map = surv_diff.set_index("tossup_slug")["difficulty"]
    cat_map = df.groupby("tossup_slug")["category_agg"].first()

    games_heard = df.groupby("tossup_slug")["game_id"].nunique().rename("n_heard")
    games_dead = (
        df[df["won"] == True].groupby("tossup_slug")["game_id"].nunique().rename("n_won")
    )
    dead_games = (games_heard - games_dead).fillna(games_heard)
    dead_rate = (dead_games / games_heard).rename("dead_rate")

    rows = []
    skipped = 0
    for slug, c_grp in correct.groupby("tossup_slug"):
        n_h = int(games_heard.get(slug, 0))
        if n_h < n_heard_min:
            skipped += 1
            continue
        bps = c_grp["buzzpoint"].values
        if len(bps) == 0:
            continue

        length = float(lengths.get(slug, np.nan))
        if pd.isna(length) or length <= 0:
            continue

        mean_buzz = float(np.mean(bps))
        std_buzz = float(np.std(bps, ddof=1)) if len(bps) > 1 else 0.0
        iqr_buzz = float(np.percentile(bps, 75) - np.percentile(bps, 25)) if len(bps) > 1 else 0.0
        power_rate = float(np.mean(bps <= power_mark))

        # cliff_score: 1 - iqr/length (high = trigger)
        cliff_score = 1.0 - (iqr_buzz / length) if length > 0 else 0.0

        # Pyramidality / slope: bin the buzzes and look at the per-bin density vs bin index.
        # Use empirical CDF rank correlation between (position, cumulative-fraction).
        # By construction CDF is monotone in position; the metric is whether the
        # *rate* increases (front loaded -> slope>0 vs back loaded -> slope<0).
        # We use Spearman rho of (bp, rank) against the *uniform-spaced* expected
        # positions; positive = front-loaded (rate decays), negative = back-loaded.
        # Easier formulation: standardised slope of relative-position regressed on
        # uniform rank.
        rel_pos = bps / length
        rel_pos_sorted = np.sort(rel_pos)
        ranks = np.arange(1, len(rel_pos_sorted) + 1) / (len(rel_pos_sorted) + 1)
        # Slope of sorted rel_pos vs uniform ranks: 1 if uniform, <1 if front-loaded,
        # >1 if back-loaded. We invert so + = front-loaded.
        slope = float(np.polyfit(ranks, rel_pos_sorted, 1)[0]) if len(rel_pos_sorted) >= 2 else 1.0
        slope_score = 1.0 - slope  # standardise around 0; >0 front-loaded

        # Pyramidality: fraction of buzzes in each successive bin should be
        # non-decreasing. Use Spearman rho between bin index and bin density.
        if length > 0 and len(bps) >= 3:
            edges = np.linspace(0, length, n_bins + 1)
            density = np.histogram(bps, bins=edges)[0]
            # Trim trailing zero bins (after last buzz) to avoid biasing rho toward 0
            last_nz = len(density) - 1
            while last_nz > 0 and density[last_nz] == 0:
                last_nz -= 1
            d_trim = density[: last_nz + 1]
            if len(d_trim) >= 2:
                rho, _ = scipy.stats.spearmanr(np.arange(len(d_trim)), d_trim)
                pyramidality = float(max(0.0, rho))
            else:
                pyramidality = 0.0
        else:
            pyramidality = 0.0

        # Discrimination: corr between "did this player buzz correctly on this tossup"
        # and the player's PL skill (across players who heard it).
        discrimination = np.nan
        if pl_skill is not None:
            heard_grp = df[df["tossup_slug"] == slug]
            buzzed = (heard_grp["buzz_value"] == 10).astype(int)
            slugs = heard_grp["player_slug"].values
            sk = pl_skill.reindex(slugs).values
            mask = ~np.isnan(sk)
            if mask.sum() >= 5 and len(set(buzzed[mask])) > 1:
                r, _ = scipy.stats.pearsonr(buzzed[mask], sk[mask])
                discrimination = float(r) if not np.isnan(r) else np.nan

        rows.append({
            "tossup_slug": slug,
            "category": cat_map.get(slug, ""),
            "length": length,
            "n_heard": n_h,
            "n_correct_buzzes": len(bps),
            "dead_rate": float(dead_rate.get(slug, 0.0)),
            "difficulty_d": float(diff_map.get(slug, np.nan)),
            "mean_buzz": mean_buzz,
            "std_buzz": std_buzz,
            "iqr_buzz": iqr_buzz,
            "power_rate": power_rate,
            "pyramidality": pyramidality,
            "slope_score": slope_score,
            "cliff_score": cliff_score,
            "discrimination": discrimination,
        })

    out = pd.DataFrame(rows)
    print(f"Tossups with metrics: {len(out)} (skipped {skipped} for n_heard<{n_heard_min})")
    return out


def classify_tossups(t: pd.DataFrame) -> pd.Series:
    """Editorial label per tossup.

    Priority order (first match wins):
        1. dud           dead_rate > 0.5
        2. gimme         rel_mean_buzz < 0.4 AND dead_rate < 0.05
        3. cliff         cliff_score > 0.85 (very tight buzz cluster)
        4. back-loaded   rel_mean_buzz > 0.85 (only converted by giveaway)
        5. pyramidal     pyramidality > 0.5 AND dead_rate < 0.3
        6. front-loaded  rel_mean_buzz < 0.55 (significant early conversion)
        7. noisy         everything else
    """
    rel_mean = t["mean_buzz"] / t["length"]
    label = pd.Series(["noisy"] * len(t), index=t.index)

    # Step through priorities; only relabel if currently "noisy"
    # 1. dud
    label[t["dead_rate"] > 0.5] = "dud"
    # 2. gimme
    mask = (rel_mean < 0.4) & (t["dead_rate"] < 0.05) & (label == "noisy")
    label[mask] = "gimme"
    # 3. cliff
    mask = (t["cliff_score"] > 0.85) & (label == "noisy")
    label[mask] = "cliff"
    # 4. back-loaded
    mask = (rel_mean > 0.85) & (label == "noisy")
    label[mask] = "back-loaded"
    # 5. pyramidal
    mask = (t["pyramidality"] > 0.5) & (t["dead_rate"] < 0.3) & (label == "noisy")
    label[mask] = "pyramidal"
    # 6. front-loaded
    mask = (rel_mean < 0.55) & (label == "noisy")
    label[mask] = "front-loaded"

    return label


def plot_examples(df: pd.DataFrame, t: pd.DataFrame, label: str, out_path: str,
                  n_examples: int = 6):
    """Grid of empirical buzz CDFs for tossups with the given label."""
    sub = t[t["label"] == label].sort_values("n_heard", ascending=False).head(n_examples)
    if len(sub) == 0:
        return
    correct = df[(df["buzz_value"] == 10) & df["buzzpoint"].notna()]
    n = len(sub)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.0), squeeze=False)
    for ax, (_, row) in zip(axes.flat, sub.iterrows()):
        slug = row["tossup_slug"]
        bps = correct[correct["tossup_slug"] == slug]["buzzpoint"].values
        if len(bps) == 0:
            ax.set_visible(False)
            continue
        sorted_bp = np.sort(bps)
        cdf = np.arange(1, len(sorted_bp) + 1) / len(sorted_bp)
        ax.step(sorted_bp, cdf, where="post", color="steelblue")
        ax.axvline(75, color="gray", ls="--", lw=0.7, alpha=0.7)
        ax.axvline(row["length"], color="red", ls=":", lw=0.7, alpha=0.7)
        ax.set_xlim(0, max(135, row["length"]))
        ax.set_ylim(0, 1)
        ax.set_title(f"{slug}\nn={int(row['n_correct_buzzes'])} dead={row['dead_rate']:.0%}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(sub):]:
        ax.set_visible(False)
    fig.suptitle(f"Empirical buzz CDFs — label: {label}", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_scatter(t: pd.DataFrame, x: str, y: str, hue: str, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {"gimme": "tab:green", "dud": "tab:gray", "cliff": "tab:red",
               "front-loaded": "tab:orange", "back-loaded": "tab:purple",
               "pyramidal": "tab:blue", "noisy": "lightgray"}
    for lab, sub in t.groupby(hue):
        ax.scatter(sub[x], sub[y], s=18, alpha=0.6, label=lab,
                   color=palette.get(lab, "k"))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}, by label")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def df_to_md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Render a DataFrame as a github-flavoured markdown table without tabulate."""
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("—")
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_report(t: pd.DataFrame, out_dir: str):
    lines = ["# Tossup Quality Analysis Report", ""]
    lines.append("Empirical shape metrics per tossup at 2026 ACF Nationals + a coarse "
                 "editorial label. See HANDOFF.md for definitions.")
    lines.append("")
    lines.append(f"Tossups analysed: **{len(t)}**")
    lines.append("")

    # Label histogram
    counts = t["label"].value_counts()
    lines.append("## Editorial label distribution")
    lines.append("")
    lines.append("| label | count | share |")
    lines.append("|---|---:|---:|")
    for lab, n in counts.items():
        lines.append(f"| {lab} | {n} | {n/len(t):.0%} |")
    lines.append("")

    # Per category
    lines.append("## Per-category mean metrics")
    lines.append("")
    g = t.groupby("category").agg(
        n=("tossup_slug", "count"),
        mean_pyr=("pyramidality", "mean"),
        mean_dead=("dead_rate", "mean"),
        mean_d=("difficulty_d", "mean"),
        mean_power=("power_rate", "mean"),
        mean_cliff=("cliff_score", "mean"),
    ).round(3)
    lines.append(df_to_md(g.reset_index()))
    lines.append("")

    # Hardest 25
    lines.append("## Top 25 hardest tossups (by `difficulty_d`)")
    lines.append("")
    hardest = t.sort_values("difficulty_d", ascending=False).head(25)
    cols_show = ["tossup_slug", "category", "difficulty_d", "dead_rate", "n_heard",
                 "mean_buzz", "label"]
    lines.append(hardest[cols_show].pipe(df_to_md))
    lines.append("")

    # Most pyramidal 20
    lines.append("## Top 20 most pyramidal tossups")
    lines.append("")
    pyr = t[t["dead_rate"] < 0.3].sort_values("pyramidality", ascending=False).head(20)
    lines.append(pyr[["tossup_slug", "category", "pyramidality", "slope_score",
                      "n_heard", "label"]].pipe(df_to_md))
    lines.append("")

    # Most cliff-like 20
    lines.append("## Top 20 most cliff-like tossups (single trigger word)")
    lines.append("")
    cl = t.sort_values("cliff_score", ascending=False).head(20)
    lines.append(cl[["tossup_slug", "category", "cliff_score", "iqr_buzz", "n_heard",
                     "label"]].pipe(df_to_md))
    lines.append("")

    out = os.path.join(out_dir, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../buzzpoints_data.csv")
    ap.add_argument("--surv-dir", default="../survival-model/results-1")
    ap.add_argument("--pl-dir", default="../plackett-luce/results-2",
                    help="Used to compute discrimination metric.")
    ap.add_argument("--output-dir", default="results-1")
    ap.add_argument("--power-mark", type=int, default=75)
    ap.add_argument("--n-heard-min", type=int, default=3)
    ap.add_argument("--n-bins", type=int, default=10,
                    help="Number of bins for pyramidality density")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_raw(args.raw)
    print(f"Loaded {len(df)} raw rows, {df['tossup_slug'].nunique()} tossups")

    surv_diff = pd.read_csv(os.path.join(args.surv_dir, "tossup_difficulty.csv"))

    pl_skill = None
    pl_path = os.path.join(args.pl_dir, "overall_rankings.csv")
    if os.path.exists(pl_path):
        pl = pd.read_csv(pl_path)
        pl_skill = pl.set_index("player_slug")["skill_estimate"]

    t = per_tossup_metrics(df, surv_diff, power_mark=args.power_mark,
                           n_heard_min=args.n_heard_min, n_bins=args.n_bins,
                           pl_skill=pl_skill)
    t["label"] = classify_tossups(t)

    out_path = os.path.join(args.output_dir, "tossup_quality.csv")
    t.sort_values("difficulty_d", ascending=False).to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    counts = t["label"].value_counts().rename_axis("label").reset_index(name="n")
    counts.to_csv(os.path.join(args.output_dir, "classification_counts.csv"), index=False)

    per_cat = t.groupby("category").agg(
        n=("tossup_slug", "count"),
        mean_pyramidality=("pyramidality", "mean"),
        mean_dead=("dead_rate", "mean"),
        mean_difficulty=("difficulty_d", "mean"),
        mean_power=("power_rate", "mean"),
        mean_cliff=("cliff_score", "mean"),
    ).round(3)
    per_cat.to_csv(os.path.join(args.output_dir, "per_category_quality.csv"))
    print(f"Saved per-category summary")

    # Plots: scatter, then example grids per label
    plot_scatter(t, "difficulty_d", "pyramidality", "label",
                 os.path.join(args.output_dir, "pyramidality_vs_difficulty.png"))
    plot_scatter(t, "mean_buzz", "std_buzz", "label",
                 os.path.join(args.output_dir, "std_vs_mean.png"))

    examples_dir = args.output_dir
    for lab in sorted(t["label"].unique()):
        out = os.path.join(examples_dir, f"shape_examples_{lab}.png")
        plot_examples(df, t, lab, out)

    write_report(t, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
