"""
Plackett-Luce model on buzzpoints data.

Usage:
    python model.py --input buzzpoints_data.csv
    python model.py --input buzzpoints_data.csv --lambda 0.1 --min-buzzes 10 --top-category Science
    python model.py --input buzzpoints_data.csv --aggregate-categories
"""

import argparse
import time
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
    rank_thresholds: tuple[int, ...] = (5, 10, 20),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Resample game-tossup records with replacement and refit the PL model each time.

    Players absent from a bootstrap sample are assigned skill = 0 (L2 prior mean),
    allowing full-field ranking in every sample.

    Returns a DataFrame indexed by player_slug with columns:
        ci_low, ci_high       — 2.5th / 97.5th percentile of skill across samples
        p_top{k} for each k in rank_thresholds
    """
    rng = np.random.default_rng(seed)
    n_records = len(records)
    n_players = len(players)

    skill_samples = np.zeros((n_bootstrap, n_players))
    rank_samples  = np.zeros((n_bootstrap, n_players), dtype=int)

    if verbose:
        print(f"  Running {n_bootstrap} bootstrap iterations...", flush=True)
    for b in range(n_bootstrap):
        if (b + 1) % 50 == 0:
            if verbose:
                print(f"    {b + 1}/{n_bootstrap}", flush=True)

        # Resample records with replacement
        indices = rng.integers(0, n_records, size=n_records)
        sample = [records[i] for i in indices]

        # Present = global player indices that appear in at least one eligible set
        present_global = sorted({gi for rec in sample for gi in rec["eligible"]})
        if len(present_global) < 2:
            skill_samples[b] = 0.0
            rank_samples[b] = n_players
            continue

        # Map global index -> local index for this sample
        global_to_local = {gi: li for li, gi in enumerate(present_global)}

        local_records = []
        for rec in sample:
            w_global = rec["winner_idx"]
            if w_global not in global_to_local:
                continue
            local_elig = [global_to_local[gi] for gi in rec["eligible"]
                          if gi in global_to_local]
            local_winner = global_to_local[w_global]
            if local_winner not in local_elig or len(local_elig) < 2:
                continue
            local_records.append({"winner_idx": local_winner, "eligible": local_elig})

        if not local_records:
            skill_samples[b] = 0.0
            rank_samples[b] = n_players
            continue

        present_players = [players[gi] for gi in present_global]
        local_skills = fit_plackett_luce(local_records, present_players, lam=lam)

        # Write back into full-player array; absent players keep 0 (prior mean)
        full_skills = np.zeros(n_players)
        for li, gi in enumerate(present_global):
            full_skills[gi] = local_skills[li]
        skill_samples[b] = full_skills

        # Rank all players in this sample (1 = best)
        rank_samples[b] = scipy.stats.rankdata(-full_skills, method="min")

    ci_low  = np.percentile(skill_samples, 2.5,  axis=0)
    ci_high = np.percentile(skill_samples, 97.5, axis=0)

    result = {"ci_low": ci_low, "ci_high": ci_high}
    for k in rank_thresholds:
        result[f"p_top{k}"] = (rank_samples <= k).mean(axis=0)

    return pd.DataFrame(result, index=players)


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

    if n_bootstrap > 0:
        print(f"\nBootstrapping overall model ({n_bootstrap} iterations)...")
        t0 = time.perf_counter()
        boot = bootstrap_plackett_luce(records, players, n_bootstrap=n_bootstrap, lam=lam)
        print(f"  Overall bootstrap done in {time.perf_counter() - t0:.1f}s")
    else:
        boot = None
    return result, boot


# ---------------------------------------------------------------------------
# Per-category models
# ---------------------------------------------------------------------------

def run_category_models(
    df: pd.DataFrame,
    lam: float = 0.1,
    n_bootstrap: int = 200,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Returns (skill_matrix, cat_boots) where:
      skill_matrix  — players x categories, NaN below min_buzzes threshold
      cat_boots     — {category: bootstrap DataFrame with ci_low/ci_high/p_top3/p_top5}
                      only rows for players who qualified in the point estimate
    """
    print(f"\n=== Per-Category Plackett-Luce Models ===")
    categories = sorted(df["category"].unique())
    print(f"Categories: {categories}")

    gets_by_cat = (
        df[df["won"] == True]
        .groupby(["player_slug", "category"])
        .size()
        .rename("gets")
    )

    all_players = sorted(df["player_slug"].unique())
    data: dict[str, dict[str, float]] = {cat: {} for cat in categories}
    cat_boots: dict[str, pd.DataFrame] = {}

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

        qualified = list(data[cat].keys())
        print(f"  {cat} (min_buzzes={threshold}): {len(records)} tossups, {len(qualified)} players qualify")

        if n_bootstrap > 0 and len(records) >= 5:
            t0 = time.perf_counter()
            boot = bootstrap_plackett_luce(
                records, cat_players, n_bootstrap=n_bootstrap, lam=lam,
                rank_thresholds=(3, 5), verbose=False,
            )
            elapsed = time.perf_counter() - t0
            cat_boots[cat] = boot.loc[[p for p in qualified if p in boot.index]]
            print(f"    bootstrap done in {elapsed:.1f}s")

    skill_matrix = pd.DataFrame(data, index=all_players, dtype=float)
    return skill_matrix, cat_boots


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


def generate_report(
    overall_result: pd.DataFrame,
    skill_matrix: pd.DataFrame,
    boot: pd.DataFrame | None = None,
    cat_boots: dict[str, pd.DataFrame] | None = None,
    output_dir: str = ".",
):
    """Write results/report.md with overall and per-category rankings."""
    lines = ["# Plackett-Luce Rankings Report\n"]

    # Overall
    lines.append(f"## Overall (Top {REPORT_TOP_OVERALL})\n")
    if boot is not None:
        lines.append("| Rank | Player | Team | Skill | 95% CI | P(Top 5) | P(Top 10) | P(Top 20) | Gets | Heard |")
        lines.append("|------|--------|------|------:|-------:|---------:|----------:|----------:|-----:|------:|")
    else:
        lines.append("| Rank | Player | Team | Skill | Gets | Heard |")
        lines.append("|------|--------|------|------:|-----:|------:|")

    for rank, row in overall_result.head(REPORT_TOP_OVERALL).iterrows():
        p = row["player_slug"]
        base = (f"| {rank} | {p} | {row['team_slug']} "
                f"| {row['skill_estimate']:.3f}")
        if boot is not None and p in boot.index:
            b = boot.loc[p]
            base += (f" | [{b['ci_low']:.3f}, {b['ci_high']:.3f}]"
                     f" | {b['p_top5']:.0%} | {b['p_top10']:.0%} | {b['p_top20']:.0%}")
        base += f" | {row['total_gets']} | {row['total_tossups_heard']} |"
        lines.append(base)

    # Per-category — always use the aggregated skill matrix
    overall_skill = overall_result.set_index("player_slug")["skill_estimate"]
    overall_team  = overall_result.set_index("player_slug")["team_slug"]

    for cat in sorted(skill_matrix.columns):
        n = REPORT_TOP_N.get(cat, REPORT_TOP_N_DEFAULT)
        col = skill_matrix[cat].dropna().sort_values(ascending=False)
        if col.empty:
            continue
        cb = (cat_boots or {}).get(cat)
        lines.append(f"\n## {cat} (Top {n})\n")
        if cb is not None:
            lines.append("| Rank | Player | Team | Category Skill | 95% CI | P(Top 3) | P(Top 5) | Overall Skill |")
            lines.append("|------|--------|------|---------------:|-------:|---------:|---------:|--------------:|")
        else:
            lines.append("| Rank | Player | Team | Category Skill | Overall Skill |")
            lines.append("|------|--------|------|---------------:|--------------:|")
        for rank, (player, skill) in enumerate(col.head(n).items(), 1):
            team = overall_team.get(player, "")
            ov = overall_skill.get(player, float("nan"))
            row_str = f"| {rank} | {player} | {team} | {skill:.3f}"
            if cb is not None and player in cb.index:
                b = cb.loc[player]
                row_str += (f" | [{b['ci_low']:.3f}, {b['ci_high']:.3f}]"
                            f" | {b['p_top3']:.0%} | {b['p_top5']:.0%}")
            elif cb is not None:
                row_str += " | — | — | —"
            row_str += f" | {ov:.3f} |"
            lines.append(row_str)

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
    parser.add_argument("--bootstrap", type=int, default=200, help="Number of bootstrap iterations (0 to skip)")
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

    # Overall model + bootstrap
    overall_result, boot = run_overall_model(df, lam=args.lam, n_bootstrap=args.bootstrap)
    overall_out = f"{args.output_dir}/overall_rankings.csv"
    overall_result.to_csv(overall_out)
    print(f"\nSaved overall rankings to {overall_out}")
    if boot is not None:
        boot.to_csv(f"{args.output_dir}/bootstrap_overall.csv")
        print(f"Saved bootstrap summary to {args.output_dir}/bootstrap_overall.csv")

    # Category models + bootstrap
    skill_matrix, cat_boots = run_category_models(df, lam=args.lam, n_bootstrap=args.bootstrap)
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
        report_boot, report_cat_boots = boot, cat_boots
    else:
        print("\nBuilding aggregated data for report...")
        df_agg = load_data(args.input, aggregate=True)
        report_overall, report_boot = run_overall_model(df_agg, lam=args.lam, n_bootstrap=args.bootstrap)
        report_matrix, report_cat_boots = run_category_models(df_agg, lam=args.lam, n_bootstrap=args.bootstrap)
    generate_report(report_overall, report_matrix, boot=report_boot,
                    cat_boots=report_cat_boots, output_dir=args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
