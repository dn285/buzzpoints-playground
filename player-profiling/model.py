"""
Player profiling — multi-dimensional player profiles + archetype clustering.

Strictly downstream of the Plackett-Luce and survival-model fits. Joins their
saved outputs with the raw buzz CSV, derives 9 profile dimensions per player,
clusters into archetypes, and emits CSVs / charts / a markdown report.

Usage:
    python model.py --output-dir results-1
    python model.py --pl-dir ../plackett-luce/results-2 \
                    --surv-dir ../survival-model/results-1 \
                    --output-dir results-1 --k 5

See HANDOFF.md for the design rationale.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


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


PROFILE_DIMS = [
    "skill_pl", "skill_surv", "speed", "accuracy",
    "neg_rate", "power_rate", "breadth", "specialization", "hard_skill",
]
RADAR_DIMS = ["skill_surv", "speed", "accuracy", "power_rate", "specialization", "hard_skill"]


def load_inputs(raw_path: str, pl_dir: str, surv_dir: str) -> dict:
    """Load every upstream artefact into one dict of DataFrames."""
    raw = pd.read_csv(raw_path, dtype=str)
    raw["won"] = raw["won"].map({"True": True, "False": False})
    for c in ("buzzpoint", "buzz_value"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["category_agg"] = raw["category"].apply(aggregate_category)

    pl_overall = pd.read_csv(os.path.join(pl_dir, "overall_rankings.csv"))
    pl_overall = pl_overall.rename(columns={pl_overall.columns[0]: "rank"})
    pl_cats = pd.read_csv(os.path.join(pl_dir, "category_skills.csv"), index_col=0)

    surv_overall = pd.read_csv(os.path.join(surv_dir, "overall_rankings.csv"))
    surv_overall = surv_overall.rename(columns={surv_overall.columns[0]: "rank"})
    surv_speed = pd.read_csv(os.path.join(surv_dir, "player_speed_profile.csv"))
    surv_diff = pd.read_csv(os.path.join(surv_dir, "tossup_difficulty.csv"))

    print(f"Raw rows: {len(raw)} | PL players: {len(pl_overall)} | "
          f"Surv players: {len(surv_overall)} | Surv speed players: {len(surv_speed)} | "
          f"Tossups w/ difficulty: {len(surv_diff)}")
    return dict(raw=raw, pl_overall=pl_overall, pl_cats=pl_cats,
                surv_overall=surv_overall, surv_speed=surv_speed, surv_diff=surv_diff)


def derive_dimensions(d: dict, n_correct_min: int = 10,
                      hard_pctile: float = 75.0) -> pd.DataFrame:
    """Compute the 9 profile dimensions per player. Filter to n_correct >= min."""
    raw = d["raw"]

    # Per-player buzz totals
    buzzed = raw[raw["buzz_value"].notna()]
    total_buzzes = buzzed.groupby("player_slug").size().rename("n_buzzes")
    n_correct = (
        buzzed[buzzed["buzz_value"] == 10]
        .groupby("player_slug").size().rename("n_correct")
    )
    n_neg = (
        buzzed[buzzed["buzz_value"] == -5]
        .groupby("player_slug").size().rename("n_neg")
    )
    n_wrong_nonneg = (
        buzzed[buzzed["buzz_value"] == 0]
        .groupby("player_slug").size().rename("n_wrong_nonneg")
    )
    n_heard = raw.groupby("player_slug").size().rename("n_heard")

    stats = pd.concat([n_heard, total_buzzes, n_correct, n_neg, n_wrong_nonneg], axis=1).fillna(0)
    stats[["n_buzzes", "n_correct", "n_neg", "n_wrong_nonneg"]] = stats[
        ["n_buzzes", "n_correct", "n_neg", "n_wrong_nonneg"]
    ].astype(int)
    stats["accuracy"] = stats["n_correct"] / stats["n_buzzes"].replace(0, np.nan)
    stats["neg_rate"] = stats["n_neg"] / stats["n_heard"].replace(0, np.nan)

    # Power rate: of correct buzzes, what fraction were at bp <= 75
    correct = buzzed[buzzed["buzz_value"] == 10].copy()
    correct["is_power"] = correct["buzzpoint"] <= 75
    pr = correct.groupby("player_slug")["is_power"].mean().rename("power_rate")
    stats = stats.join(pr)

    # PL skill (overall + category breakdth/specialization)
    pl_o = d["pl_overall"].set_index("player_slug")["skill_estimate"].rename("skill_pl")
    stats = stats.join(pl_o)

    pl_c = d["pl_cats"]
    breadth = pl_c.notna().sum(axis=1).rename("breadth")
    cat_max = pl_c.max(axis=1, skipna=True)
    cat_mean = pl_c.mean(axis=1, skipna=True)
    spec = (cat_max - cat_mean).rename("specialization")
    stats = stats.join(breadth).join(spec)

    # Survival skill + speed (sign-flipped residual buzzpoint so + = faster)
    surv_o = d["surv_overall"].set_index("player_slug")["skill_estimate"].rename("skill_surv")
    surv_s = d["surv_speed"].set_index("player_slug")
    stats = stats.join(surv_o).join(-surv_s["mean_residual_bp"].rename("speed"))

    # Hard skill: residual buzz on hard tossups (top 25% of d_t), sign-flipped.
    diff = d["surv_diff"].set_index("tossup_slug")
    cutoff = np.percentile(diff["difficulty"], hard_pctile)
    hard_slugs = set(diff[diff["difficulty"] >= cutoff].index)
    correct_hard = correct[correct["tossup_slug"].isin(hard_slugs)].copy()
    if len(correct_hard) > 0:
        # Per-tossup mean correct buzz on hard tossups
        per_tossup_mean = correct_hard.groupby("tossup_slug")["buzzpoint"].mean()
        correct_hard["tossup_mean"] = correct_hard["tossup_slug"].map(per_tossup_mean)
        correct_hard["resid"] = correct_hard["buzzpoint"] - correct_hard["tossup_mean"]
        # Aggregated per player; flip sign so higher = better-on-hard
        hard = (-correct_hard.groupby("player_slug")["resid"].mean()).rename("hard_skill")
        # Add a count and require at least 5 correct on hard tossups
        hard_n = correct_hard.groupby("player_slug").size().rename("n_correct_hard")
        stats = stats.join(hard).join(hard_n)
        stats.loc[stats["n_correct_hard"].fillna(0) < 5, "hard_skill"] = np.nan
    else:
        stats["hard_skill"] = np.nan

    # Team slug (first observed)
    team = raw.groupby("player_slug")["team_slug"].first().rename("team_slug")
    stats = stats.join(team)

    # Filter
    qualified = stats[stats["n_correct"] >= n_correct_min].copy()
    qualified.index.name = "player_slug"

    print(f"Players with n_correct >= {n_correct_min}: {len(qualified)}")
    return qualified


def cluster_archetypes(profiles: pd.DataFrame, k: int, dims: list[str],
                       seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """k-means in the standardised dimension space; impute NaN with 0.

    Returns (assignments_df, centroids_df, diagnostics).
    """
    X_raw = profiles[dims].copy()
    # Impute missing with column mean (after standardisation, that's 0)
    X_raw_fill = X_raw.copy()
    for c in dims:
        col = X_raw_fill[c]
        X_raw_fill[c] = col.fillna(col.mean())
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw_fill.values)
    # Winsorise at +/- 3
    X = np.clip(X, -3, 3)

    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_  # in standardised space

    # Silhouette across k = 3..8 (diagnostic)
    diagnostics = {"silhouette": {}, "inertia": {}}
    for kk in range(3, 9):
        try:
            km_k = KMeans(n_clusters=kk, random_state=seed, n_init=10)
            lk = km_k.fit_predict(X)
            diagnostics["silhouette"][kk] = float(silhouette_score(X, lk))
            diagnostics["inertia"][kk] = float(km_k.inertia_)
        except Exception:
            diagnostics["silhouette"][kk] = float("nan")
            diagnostics["inertia"][kk] = float("nan")

    assignments = profiles.copy()
    assignments["cluster"] = labels
    centroids_df = pd.DataFrame(centroids, columns=dims)
    centroids_df.index.name = "cluster"

    diagnostics["chosen_k"] = k
    diagnostics["scaler_means"] = dict(zip(dims, scaler.mean_.tolist()))
    diagnostics["scaler_scales"] = dict(zip(dims, scaler.scale_.tolist()))
    diagnostics["X_standardised"] = X
    return assignments, centroids_df, diagnostics


def label_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """Assign a human label per cluster from its centroid coords. Heuristic only."""
    labels = {}
    for cid, row in centroids.iterrows():
        speed = row.get("speed", 0)
        spec = row.get("specialization", 0)
        skill = row.get("skill_surv", 0)
        acc = row.get("accuracy", 0)
        power = row.get("power_rate", 0)
        # Crude rule-based labelling. The reader is expected to override.
        if spec > 0.7 and skill > 0:
            labels[cid] = "deep specialist"
        elif speed > 0.7 and skill > 0:
            labels[cid] = "fast generalist"
        elif power > 0.7 and acc < -0.2:
            labels[cid] = "trigger-happy"
        elif speed < -0.5 and acc > 0:
            labels[cid] = "late convertor"
        elif abs(speed) < 0.4 and abs(spec) < 0.4 and abs(skill) < 0.4:
            labels[cid] = "balanced"
        elif skill < -0.5:
            labels[cid] = "developing"
        else:
            # Fallback: name by strongest absolute coord
            top_dim = row.abs().idxmax()
            sign = "high" if row[top_dim] > 0 else "low"
            labels[cid] = f"{sign}-{top_dim}"
    return labels


def plot_radar(player_row: pd.Series, dims: list[str], scaler_means: dict,
               scaler_scales: dict, out_path: str):
    """Single-player radar chart in standardised space."""
    vals = []
    for c in dims:
        v = player_row.get(c)
        if pd.isna(v):
            v = 0.0
        z = (v - scaler_means[c]) / scaler_scales[c]
        vals.append(float(np.clip(z, -3, 3)))
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    vals = vals + vals[:1]
    angles = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, marker="o", lw=2, color="steelblue")
    ax.fill(angles, vals, alpha=0.25, color="steelblue")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=9)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_ylim(-3, 3)
    ax.set_title(player_row.name, pad=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_pca_scatter(profiles: pd.DataFrame, X: np.ndarray, labels: np.ndarray,
                     cluster_labels: dict, out_path: str):
    pca = PCA(n_components=2, random_state=0)
    Y = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for cid in sorted(set(labels)):
        mask = labels == cid
        ax.scatter(Y[mask, 0], Y[mask, 1], s=30, alpha=0.7,
                   color=cmap(cid % 10),
                   label=f"{cid}: {cluster_labels.get(cid, '')}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title("Player profile PCA, colored by archetype cluster")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_scatter(profiles: pd.DataFrame, x: str, y: str, hue: str | None,
                 out_path: str, top_n_label: int = 15):
    fig, ax = plt.subplots(figsize=(8, 6))
    if hue is not None and hue in profiles.columns:
        cmap = plt.get_cmap("tab10")
        for cid in sorted(profiles[hue].dropna().unique()):
            sub = profiles[profiles[hue] == cid]
            ax.scatter(sub[x], sub[y], s=25, alpha=0.7, label=f"cluster {int(cid)}",
                       color=cmap(int(cid) % 10))
        ax.legend(fontsize=8)
    else:
        ax.scatter(profiles[x], profiles[y], s=25, alpha=0.7, color="steelblue")
    # Label top-N by skill_surv if available, else by y
    score = profiles.get("skill_surv", profiles[y])
    top = score.dropna().sort_values(ascending=False).head(top_n_label).index
    for p in top:
        if p in profiles.index and not pd.isna(profiles.at[p, x]) and not pd.isna(profiles.at[p, y]):
            ax.annotate(p, (profiles.at[p, x], profiles.at[p, y]), fontsize=7, alpha=0.85)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def write_report(profiles: pd.DataFrame, centroids: pd.DataFrame,
                 cluster_labels: dict, diagnostics: dict, out_dir: str):
    lines = ["# Player Profiling Report", ""]
    lines.append("Multi-dimensional profile of every player with at least 10 correct buzzes "
                 "at 2026 ACF Nationals. Dimensions are derived from the Plackett-Luce and "
                 "survival-model fits. See HANDOFF.md for the rationale.")
    lines.append("")
    lines.append(f"Players profiled: **{len(profiles)}**")
    lines.append("")
    lines.append("## Archetype clusters")
    lines.append("")
    lines.append(f"k-means with k={diagnostics['chosen_k']}. Silhouette score by k:")
    lines.append("")
    lines.append("| k | silhouette | inertia |")
    lines.append("|---|-----------:|--------:|")
    for kk in sorted(diagnostics["silhouette"]):
        lines.append(f"| {kk} | {diagnostics['silhouette'][kk]:.3f} "
                     f"| {diagnostics['inertia'][kk]:.1f} |")
    lines.append("")
    lines.append("Cluster centroids (standardised):")
    lines.append("")
    cols = list(centroids.columns)
    lines.append("| cluster | label | " + " | ".join(cols) + " |")
    lines.append("|---|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for cid, row in centroids.iterrows():
        cells = " | ".join(f"{row[c]:+.2f}" for c in cols)
        lines.append(f"| {cid} | {cluster_labels.get(cid, '')} | {cells} |")
    lines.append("")
    lines.append("Cluster sizes:")
    sz = profiles["cluster"].value_counts().sort_index()
    lines.append("")
    lines.append("| cluster | label | n |")
    lines.append("|---|---|---:|")
    for cid in sz.index:
        lines.append(f"| {cid} | {cluster_labels.get(cid, '')} | {int(sz[cid])} |")
    lines.append("")

    lines.append("## Top 30 players by survival skill")
    lines.append("")
    cols_show = ["team_slug", "cluster"] + PROFILE_DIMS
    sub = profiles.sort_values("skill_surv", ascending=False).head(30)
    head_cells = "| player | " + " | ".join(cols_show) + " |"
    sep = "|---|" + "|".join(["---"] * len(cols_show)) + "|"
    lines.append(head_cells)
    lines.append(sep)
    for p, row in sub.iterrows():
        cluster_str = f"{int(row['cluster'])}: {cluster_labels.get(int(row['cluster']), '')}"
        cells = []
        for c in cols_show:
            v = row.get(c)
            if c == "team_slug":
                cells.append(str(v))
            elif c == "cluster":
                cells.append(cluster_str)
            elif pd.isna(v):
                cells.append("—")
            elif c == "breadth":
                cells.append(str(int(v)))
            else:
                cells.append(f"{v:+.2f}")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    lines.append("")

    # Per cluster: 5 exemplars (closest to centroid in standardised space)
    lines.append("## Cluster exemplars")
    lines.append("")
    X = diagnostics["X_standardised"]
    for cid, row in centroids.iterrows():
        idx_in_cluster = np.where(profiles["cluster"].values == cid)[0]
        if len(idx_in_cluster) == 0:
            continue
        center = row.values  # standardised
        dists = np.linalg.norm(X[idx_in_cluster] - center, axis=1)
        order = np.argsort(dists)
        top_global_idx = idx_in_cluster[order[:5]]
        top_players = profiles.index[top_global_idx]
        lines.append(f"### Cluster {cid}: {cluster_labels.get(cid, '')}")
        lines.append("")
        lines.append(f"Centroid: " + ", ".join(
            f"{c}={row[c]:+.2f}" for c in centroids.columns
        ))
        lines.append("")
        lines.append("Closest 5 players:")
        for p in top_players:
            lines.append(f"- `{p}` ({profiles.loc[p, 'team_slug']})")
        lines.append("")

    out = os.path.join(out_dir, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../buzzpoints_data.csv")
    ap.add_argument("--pl-dir", default="../plackett-luce/results-2")
    ap.add_argument("--surv-dir", default="../survival-model/results-1")
    ap.add_argument("--output-dir", default="results-1")
    ap.add_argument("--k", type=int, default=5, help="Number of archetype clusters")
    ap.add_argument("--n-correct-min", type=int, default=10)
    ap.add_argument("--n-radar", type=int, default=30, help="Top-N players to render radar charts for")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    d = load_inputs(args.raw, args.pl_dir, args.surv_dir)
    profiles = derive_dimensions(d, n_correct_min=args.n_correct_min)

    # Save raw profile table (before clustering)
    profile_out = os.path.join(args.output_dir, "player_profiles.csv")
    profiles.to_csv(profile_out)
    print(f"Saved {profile_out}")

    assignments, centroids, diag = cluster_archetypes(profiles, args.k, PROFILE_DIMS)
    labels_map = label_clusters(centroids)

    # Save artefacts
    centroids_out = os.path.join(args.output_dir, "archetypes.csv")
    centroids_df = centroids.copy()
    centroids_df["label"] = [labels_map[cid] for cid in centroids_df.index]
    centroids_df.to_csv(centroids_out)
    print(f"Saved {centroids_out}")

    assign_out = os.path.join(args.output_dir, "archetype_assignments.csv")
    assignments_save = assignments[["team_slug", "cluster"] + PROFILE_DIMS].copy()
    assignments_save["cluster_label"] = assignments_save["cluster"].map(labels_map)
    assignments_save.to_csv(assign_out)
    print(f"Saved {assign_out}")

    # Plots
    plot_pca_scatter(assignments, diag["X_standardised"],
                     assignments["cluster"].values, labels_map,
                     os.path.join(args.output_dir, "pca_scatter.png"))
    plot_scatter(assignments, "speed", "skill_surv", "cluster",
                 os.path.join(args.output_dir, "speed_vs_skill.png"))
    plot_scatter(assignments, "breadth", "specialization", "cluster",
                 os.path.join(args.output_dir, "specialization_vs_breadth.png"))

    # Per-player radar charts (top N by survival skill)
    radar_dir = os.path.join(args.output_dir, "radars")
    os.makedirs(radar_dir, exist_ok=True)
    top = assignments.sort_values("skill_surv", ascending=False).head(args.n_radar)
    for p, row in top.iterrows():
        safe = p.replace("/", "_")
        out = os.path.join(radar_dir, f"radar_{safe}.png")
        plot_radar(row, RADAR_DIMS, diag["scaler_means"], diag["scaler_scales"], out)
    print(f"Saved {len(top)} radar charts to {radar_dir}")

    write_report(assignments, centroids, labels_map, diag, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
