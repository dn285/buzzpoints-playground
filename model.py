"""
Plackett-Luce model on buzzpoints data.

Usage:
    python model.py --input buzzpoints_data.csv
    python model.py --input buzzpoints_data.csv --lambda 0.1 --min-buzzes 10 --top-category Science
    python model.py --input buzzpoints_data.csv --aggregate-categories
"""

import argparse
import warnings
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Category aggregation
# ---------------------------------------------------------------------------

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

# Minimum correct buzzes required for a player to appear in each category's model.
MIN_BUZZES: dict[str, int] = {
    "Fine Arts":   10,
    "History":     10,
    "Literature":  10,
    "Science":     10,
    "Beliefs":     5,
    "Thought":     8,
    "Other":       8,
}


def aggregate_category(raw: str) -> str:
    """Map 'Science - Biology' -> 'Science', 'Geography' -> 'Other', etc."""
    raw = raw.strip()
    for prefix, top in CATEGORY_MAP.items():
        if raw.startswith(prefix):
            return top
    return "Other"


# ---------------------------------------------------------------------------
# Plackett-Luce fitting
# ---------------------------------------------------------------------------

def fit_plackett_luce(
    tossups: list[dict],
    players: list[str],
    lam: float = 0.1,
) -> np.ndarray:
    """
    Fit PL model.

    tossups: list of {"winner_idx": int, "eligible": [int, ...]}
             indices into `players` list.
    players: ordered list of player IDs (for index mapping).
    lam: L2 regularization strength.

    Returns: np.ndarray of shape (len(players),), skill parameters.
    """
    n = len(players)
    if n == 0:
        return np.array([])

    def neg_log_likelihood(s):
        ll = 0.0
        for t in tossups:
            wi = t["winner_idx"]
            elig = t["eligible"]
            if len(elig) == 0:
                continue
            s_elig = s[elig]
            # log-sum-exp trick
            max_s = np.max(s_elig)
            lse = max_s + np.log(np.sum(np.exp(s_elig - max_s)))
            ll += s[wi] - lse
        # L2 penalty
        ll -= lam * np.dot(s, s)
        return -ll

    def grad(s):
        g = np.zeros(n)
        for t in tossups:
            wi = t["winner_idx"]
            elig = t["eligible"]
            if len(elig) == 0:
                continue
            s_elig = s[elig]
            max_s = np.max(s_elig)
            exp_s = np.exp(s_elig - max_s)
            probs = exp_s / exp_s.sum()
            g[wi] += 1.0
            for j, pidx in enumerate(elig):
                g[pidx] -= probs[j]
        # L2
        g -= 2.0 * lam * s
        return -g

    s0 = np.zeros(n)
    result = scipy.optimize.minimize(
        neg_log_likelihood,
        s0,
        jac=grad,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    s = result.x
    # Center so mean = 0
    s -= s.mean()
    return s


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_plackett_luce(
    records: list[dict],
    players: list[str],
    n_bootstrap: int = 200,
    lam: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Resample game-tossup records with replacement and refit the PL model each time.

    Players absent from a bootstrap sample (none of their eligible sets were drawn)
    are assigned skill = 0, consistent with the L2 prior that shrinks toward zero.
    This lets us rank every player in every sample and compute rank probabilities.

    Returns a DataFrame indexed by player_slug with columns:
        ci_low, ci_high   — 2.5th / 97.5th percentile of skill across samples
        p_top5, p_top10, p_top20  — fraction of samples where rank <= threshold
    """
    rng = np.random.default_rng(seed)
    n_records = len(records)
    n_players = len(players)
    pidx = {p: i for i, p in enumerate(players)}

    skill_samples = np.zeros((n_bootstrap, n_players))  # absent → 0 (prior mean)
    rank_samples  = np.zeros((n_bootstrap, n_players), dtype=int)

    print(f"  Running {n_bootstrap} bootstrap iterations...", flush=True)
    for b in range(n_bootstrap):
        if (b + 1) % 50 == 0:
            print(f"    {b + 1}/{n_bootstrap}", flush=True)

        # Resample records with replacement
        indices = rng.integers(0, n_records, size=n_records)
        sample = [records[i] for i in indices]

        # Identify players present in this sample (appear in at least one eligible set)
        present = sorted({p for rec in sample for p in rec["eligible"]})
        if len(present) < 2:
            skill_samples[b] = 0.0
            rank_samples[b] = n_players  # everyone gets last rank; sample is unusable
            continue

        # Remap eligible sets to local indices
        local_pidx = {p: i for i, p in enumerate(present)}
        local_records = []
        for rec in sample:
            winner_slug = players[rec["winner_idx"]]
            if winner_slug not in local_pidx:
                continue
            local_elig = [local_pidx[players[gi]] for gi in rec["eligible"]
                          if players[gi] in local_pidx]
            local_winner = local_pidx[winner_slug]
            if local_winner not in local_elig or len(local_elig) < 2:
                continue
            local_records.append({"winner_idx": local_winner, "eligible": local_elig})

        if not local_records:
            skill_samples[b] = 0.0
            rank_samples[b] = n_players
            continue

        local_skills = fit_plackett_luce(local_records, present, lam=lam)

        # Write back into full-player array; absent players keep 0
        full_skills = np.zeros(n_players)
        for local_i, p in enumerate(present):
            full_skills[pidx[p]] = local_skills[local_i]
        skill_samples[b] = full_skills

        # Rank all players in this sample (1 = best)
        rank_samples[b] = scipy.stats.rankdata(-full_skills, method="min")

    ci_low  = np.percentile(skill_samples, 2.5,  axis=0)
    ci_high = np.percentile(skill_samples, 97.5, axis=0)
    p_top5  = (rank_samples <= 5).mean(axis=0)
    p_top10 = (rank_samples <= 10).mean(axis=0)
    p_top20 = (rank_samples <= 20).mean(axis=0)

    return pd.DataFrame({
        "ci_low":   ci_low,
        "ci_high":  ci_high,
        "p_top5":   p_top5,
        "p_top10":  p_top10,
        "p_top20":  p_top20,
    }, index=players)


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_data(path: str, aggregate: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    bool_cols = {"won"}
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({"True": True, "False": False, True: True, False: False})

    for col in ("buzzpoint", "buzz_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if aggregate and "category" in df.columns:
        df["category"] = df["category"].apply(aggregate_category)

    print(f"Loaded {len(df)} rows, {df['tossup_slug'].nunique()} tossups, "
          f"{df['player_slug'].nunique()} players, {df['category'].nunique()} categories")
    return df


def build_tossup_records(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """
    Returns (tossups, players) where:
      tossups: list of {tossup_slug, category, winner_slug, winner_idx, eligible: [idx]}
      players: ordered list of player slugs (index = player_idx)
    """
    players = sorted(df["player_slug"].unique())
    pidx = {p: i for i, p in enumerate(players)}

    records = []
    skipped_no_winner = 0
    skipped_trivial = 0

    group_cols = ["tossup_slug", "game_id"] if "game_id" in df.columns else ["tossup_slug"]
    for _, grp in df.groupby(group_cols):
        category = grp["category"].iloc[0]

        # Winner
        winners = grp[grp["won"] == True]
        if len(winners) == 0:
            skipped_no_winner += 1
            continue
        winner_slug = winners.iloc[0]["player_slug"]
        winner_idx = pidx[winner_slug]

        # Discard this game if any neg occurred — rebound wins have reduced competition.
        # Because we group by game_id, this only drops the specific game where the neg
        # happened; other teams playing the same tossup are in separate groups and unaffected.
        if (grp["buzz_value"] == -5).any():
            skipped_trivial += 1
            continue

        eligible_slugs = list(grp["player_slug"].unique())
        eligible_idxs = [pidx[p] for p in eligible_slugs]

        if len(eligible_idxs) <= 1:
            skipped_trivial += 1
            continue

        records.append({
            "tossup_slug": grp["tossup_slug"].iloc[0],
            "category": category,
            "winner_slug": winner_slug,
            "winner_idx": winner_idx,
            "eligible": eligible_idxs,
        })

    print(f"Usable tossups: {len(records)}  "
          f"(skipped {skipped_no_winner} no-winner, {skipped_trivial} trivial)")
    return records, players


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def player_stats(df: pd.DataFrame, players: list[str]) -> pd.DataFrame:
    gets = df[df["won"] == True].groupby("player_slug").size().rename("total_gets")
    heard = df.groupby("player_slug").size().rename("total_tossups_heard")
    team = df.groupby("player_slug")["team_slug"].first()
    stats = pd.DataFrame({"player_slug": players}).set_index("player_slug")
    stats = stats.join(gets).join(heard).join(team)
    stats["total_gets"] = stats["total_gets"].fillna(0).astype(int)
    stats["total_tossups_heard"] = stats["total_tossups_heard"].fillna(0).astype(int)
    return stats.reset_index()


# ---------------------------------------------------------------------------
# Overall model
# ---------------------------------------------------------------------------

def run_overall_model(
    df: pd.DataFrame,
    lam: float = 0.1,
    n_bootstrap: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (result, boot) where result is the point-estimate ranking and
    boot is the bootstrap summary (ci_low, ci_high, p_top5, p_top10, p_top20)."""
    print("\n=== Overall Plackett-Luce Model ===")
    records, players = build_tossup_records(df)
    skills = fit_plackett_luce(records, players, lam=lam)

    stats = player_stats(df, players)
    result = stats.copy()
    result["skill_estimate"] = skills
    result = result.sort_values("skill_estimate", ascending=False).reset_index(drop=True)
    result.index += 1  # 1-based rank

    print(f"\nTop 20 players (overall):")
    print(result[["player_slug", "team_slug", "skill_estimate", "total_gets", "total_tossups_heard"]].head(20).to_string())

    print(f"\nBootstrapping overall model ({n_bootstrap} iterations)...")
    boot = bootstrap_plackett_luce(records, players, n_bootstrap=n_bootstrap, lam=lam)
    return result, boot


# ---------------------------------------------------------------------------
# Per-category models
# ---------------------------------------------------------------------------

def run_category_models(
    df: pd.DataFrame,
    lam: float = 0.1,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by player_slug with one column per category.
    Only cells where the player meets the MIN_BUZZES threshold for that category
    are filled; others NaN.
    """
    print(f"\n=== Per-Category Plackett-Luce Models ===")
    categories = sorted(df["category"].unique())
    print(f"Categories: {categories}")

    # Count correct buzzes per (player, category)
    gets_by_cat = (
        df[df["won"] == True]
        .groupby(["player_slug", "category"])
        .size()
        .rename("gets")
    )

    all_players = sorted(df["player_slug"].unique())

    # Build as a dict of dicts to avoid pandas CoW issues with .loc in a loop
    data: dict[str, dict[str, float]] = {cat: {} for cat in categories}

    for cat in categories:
        threshold = MIN_BUZZES.get(cat, 5)
        cat_df = df[df["category"] == cat]
        records, cat_players = build_tossup_records(cat_df)

        if len(records) < 5:
            print(f"  {cat}: too few tossups ({len(records)}), skipping")
            continue

        skills = fit_plackett_luce(records, cat_players, lam=lam)

        for ps, sk in zip(cat_players, skills):
            n_gets = gets_by_cat.get((ps, cat), 0)
            if n_gets >= threshold:
                data[cat][ps] = float(sk)

        print(f"  {cat} (min_buzzes={threshold}): {len(records)} tossups, {len(data[cat])} players qualify")

    skill_matrix = pd.DataFrame(data, index=all_players, dtype=float)
    return skill_matrix


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analysis_specialization(skill_matrix: pd.DataFrame, output_dir: str = "."):
    """Deviation of each category skill from player's mean category skill."""
    print("\n=== Specialization Profiles ===")
    # Only use rows with at least 2 non-NaN categories
    valid = skill_matrix[skill_matrix.notna().sum(axis=1) >= 2].copy()
    if valid.empty:
        print("No players with 2+ category qualifications.")
        return

    row_means = valid.mean(axis=1)
    deviation = valid.subtract(row_means, axis=0)

    print("\nTop specialists (largest max absolute deviation):")
    max_dev = deviation.abs().max(axis=1).sort_values(ascending=False)
    print(max_dev.head(20).to_string())

    # Heatmap: subset to players with at least 3 categories, then take top 50 by deviation
    heat_data = deviation[deviation.notna().sum(axis=1) >= 3]
    if len(heat_data) > 50:
        heat_max_dev = heat_data.abs().max(axis=1).sort_values(ascending=False)
        heat_data = heat_data.loc[heat_max_dev.head(50).index]

    if not heat_data.empty:
        n_players = len(heat_data)
        n_cats = len(heat_data.columns)
        # Players vertical, categories horizontal; thin category columns
        fig, ax = plt.subplots(figsize=(max(5, n_cats * 0.5), max(8, n_players * 0.25)))
        sns.heatmap(
            heat_data,
            cmap="RdBu_r",
            center=0,
            ax=ax,
            xticklabels=True,
            yticklabels=True,
        )
        ax.set_title("Category skill deviation from player mean")
        ax.set_xlabel("Category")
        ax.set_ylabel("Player")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out = f"{output_dir}/specialization_heatmap.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")


def analysis_cross_category_correlation(skill_matrix: pd.DataFrame, output_dir: str = "."):
    """Pairwise Pearson correlations between category skill vectors."""
    print("\n=== Cross-Category Correlation Matrix ===")
    cats = skill_matrix.columns.tolist()
    corr = pd.DataFrame(index=cats, columns=cats, dtype=float)

    for c1, c2 in combinations(cats, 2):
        joint = skill_matrix[[c1, c2]].dropna()
        if len(joint) < 5:
            continue
        r = joint[c1].corr(joint[c2])
        corr.loc[c1, c2] = r
        corr.loc[c2, c1] = r
    for c in cats:
        corr.loc[c, c] = 1.0

    print(corr.round(2).to_string())

    fig, ax = plt.subplots(figsize=(max(8, len(cats)), max(6, len(cats))))
    mask = corr.isna()
    sns.heatmap(
        corr.astype(float),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        mask=mask,
        ax=ax,
        vmin=-1, vmax=1,
    )
    ax.set_title("Cross-category skill correlation")
    plt.tight_layout()
    out = f"{output_dir}/category_correlation.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
    return corr



def analysis_residual_variance(skill_matrix: pd.DataFrame, overall_result: pd.DataFrame, output_dir: str = "."):
    """Variance of (category skill - overall skill) per category."""
    print("\n=== Residual Variance by Category ===")
    overall = overall_result.set_index("player_slug")["skill_estimate"]

    variances = {}
    for cat in skill_matrix.columns:
        col = skill_matrix[cat].dropna()
        if len(col) < 5:
            continue
        diff = col - overall.reindex(col.index).fillna(0)
        variances[cat] = diff.var()

    var_series = pd.Series(variances, dtype=float).sort_values(ascending=False)
    print(var_series.round(4).to_string())

    if var_series.empty:
        print("No categories with enough qualifying players — skipping plot.")
        return var_series

    fig, ax = plt.subplots(figsize=(10, 5))
    var_series.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Residual variance by category\n(Var of category skill − overall skill)")
    ax.set_ylabel("Variance")
    ax.set_xlabel("Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = f"{output_dir}/residual_variance.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
    return var_series


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

REPORT_TOP_N = {
    "History": 15,
    "Literature": 15,
    "Science": 15,
}
REPORT_TOP_N_DEFAULT = 10
REPORT_TOP_OVERALL = 30


def generate_report(overall_result: pd.DataFrame, skill_matrix: pd.DataFrame, output_dir: str = "."):
    """Write results/report.md with overall and per-category rankings."""
    lines = ["# Plackett-Luce Rankings Report\n"]

    # Overall
    lines.append(f"## Overall (Top {REPORT_TOP_OVERALL})\n")
    lines.append("| Rank | Player | Team | Skill | Gets | Heard |")
    lines.append("|------|--------|------|------:|-----:|------:|")
    for rank, row in overall_result.head(REPORT_TOP_OVERALL).iterrows():
        lines.append(
            f"| {rank} | {row['player_slug']} | {row['team_slug']} "
            f"| {row['skill_estimate']:.3f} | {row['total_gets']} | {row['total_tossups_heard']} |"
        )

    # Per-category — always use the aggregated skill matrix
    overall_skill = overall_result.set_index("player_slug")["skill_estimate"]
    overall_team  = overall_result.set_index("player_slug")["team_slug"]

    for cat in sorted(skill_matrix.columns):
        n = REPORT_TOP_N.get(cat, REPORT_TOP_N_DEFAULT)
        col = skill_matrix[cat].dropna().sort_values(ascending=False)
        if col.empty:
            continue
        lines.append(f"\n## {cat} (Top {n})\n")
        lines.append("| Rank | Player | Team | Category Skill | Overall Skill |")
        lines.append("|------|--------|------|---------------:|--------------:|")
        for rank, (player, skill) in enumerate(col.head(n).items(), 1):
            team = overall_team.get(player, "")
            overall = overall_skill.get(player, float("nan"))
            lines.append(
                f"| {rank} | {player} | {team} "
                f"| {skill:.3f} | {overall:.3f} |"
            )

    report_path = f"{output_dir}/report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fit Plackett-Luce model on buzzpoints data")
    parser.add_argument("--input", default="buzzpoints_data.csv", help="CSV from scraper.py")
    parser.add_argument("--lam", type=float, default=0.1, help="L2 regularization lambda")
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    parser.add_argument("--aggregate-categories", action="store_true",
                        help="Aggregate subcategories (e.g. Science - Biology -> Science)")
    parser.add_argument("--no-aggregate", dest="aggregate_categories", action="store_false")
    parser.set_defaults(aggregate_categories=True)
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    df = load_data(args.input, aggregate=args.aggregate_categories)

    # Overall model
    overall_result = run_overall_model(df, lam=args.lam)
    overall_out = f"{args.output_dir}/overall_rankings.csv"
    overall_result.to_csv(overall_out)
    print(f"\nSaved overall rankings to {overall_out}")

    # Category models
    skill_matrix = run_category_models(df, lam=args.lam)
    matrix_out = f"{args.output_dir}/category_skills.csv"
    skill_matrix.to_csv(matrix_out)
    print(f"Saved category skill matrix to {matrix_out}")

    # Analysis
    analysis_specialization(skill_matrix, output_dir=args.output_dir)
    analysis_cross_category_correlation(skill_matrix, output_dir=args.output_dir)
    analysis_residual_variance(skill_matrix, overall_result, output_dir=args.output_dir)

    # Markdown report — always uses aggregated categories
    if args.aggregate_categories:
        report_overall, report_matrix = overall_result, skill_matrix
    else:
        print("\nBuilding aggregated data for report...")
        df_agg = load_data(args.input, aggregate=True)
        report_overall = run_overall_model(df_agg, lam=args.lam)
        report_matrix = run_category_models(df_agg, lam=args.lam)
    generate_report(report_overall, report_matrix, output_dir=args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
