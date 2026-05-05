"""
Plackett-Luce model on buzzpoints data.

Usage:
    python model.py --input buzzpoints_data.csv
    python model.py --input buzzpoints_data.csv --lambda 0.1 --min-buzzes 10 --top-category Science
    python model.py --input buzzpoints_data.csv --aggregate-categories
"""

import argparse
import multiprocessing
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
    Fit PL model (no difficulty parameters — legacy / fallback).

    tossups: list of {"winner_idx": int, "eligible": [int, ...]}
             indices into `players` list.  Dead-tossup records (dead=True) are skipped.
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
            if t.get("dead", False):
                continue
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
            if t.get("dead", False):
                continue
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


def fit_plackett_luce_with_difficulty(
    tossups: list[dict],
    players: list[str],
    lam: float = 0.1,
    lam_d: float = 0.1,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Fit PL model jointly estimating player skill and per-tossup difficulty.

    Each tossup acts as a virtual competitor: the difficulty parameter d_t represents
    the log-odds that the question goes dead relative to a player with skill 0.

    For won events:   LL += s[winner] - log(exp(d_t) + sum(exp(s[k]) for k in eligible))
    For dead events:  LL += d_t       - log(exp(d_t) + sum(exp(s[k]) for k in eligible))

    tossups: records with keys tossup_slug, eligible, dead (bool), winner_idx (if not dead).
    players: ordered list of player slugs.
    lam:   L2 lambda for skill params.
    lam_d: L2 lambda for difficulty params.

    Returns (skills, tossup_slugs, difficulties):
      skills:        shape (n_players,), mean-centered post-fit
      tossup_slugs:  sorted list of unique tossup slugs in this dataset
      difficulties:  shape (n_tossups,), same scale as skills after centering
    """
    n = len(players)
    if n == 0:
        return np.array([]), [], np.array([])

    tossup_slugs = sorted({r["tossup_slug"] for r in tossups})
    m = len(tossup_slugs)
    tidx = {sl: i for i, sl in enumerate(tossup_slugs)}

    def neg_log_likelihood(theta):
        s = theta[:n]
        d = theta[n:]
        ll = 0.0
        for t in tossups:
            ti = tidx[t["tossup_slug"]]
            elig = t["eligible"]
            dead = t.get("dead", False)
            vals = np.empty(1 + len(elig))
            vals[0] = d[ti]
            if len(elig):
                vals[1:] = s[np.array(elig)]
            max_v = np.max(vals)
            lse = max_v + np.log(np.sum(np.exp(vals - max_v)))
            ll += (d[ti] if dead else s[t["winner_idx"]]) - lse
        ll -= lam * np.dot(s, s)
        ll -= lam_d * np.dot(d, d)
        return -ll

    def grad_func(theta):
        s = theta[:n]
        d = theta[n:]
        g = np.zeros(n + m)
        for t in tossups:
            ti = tidx[t["tossup_slug"]]
            elig = t["eligible"]
            dead = t.get("dead", False)
            vals = np.empty(1 + len(elig))
            vals[0] = d[ti]
            if len(elig):
                vals[1:] = s[np.array(elig)]
            max_v = np.max(vals)
            exp_v = np.exp(vals - max_v)
            probs = exp_v / exp_v.sum()
            # Credit the winner (+1)
            if dead:
                g[n + ti] += 1.0
            else:
                g[t["winner_idx"]] += 1.0
            # Subtract expected probabilities from all competitors
            g[n + ti] -= probs[0]
            for j, pidx in enumerate(elig):
                g[pidx] -= probs[1 + j]
        g[:n] -= 2.0 * lam * s
        g[n:] -= 2.0 * lam_d * d
        return -g

    theta0 = np.zeros(n + m)
    result = scipy.optimize.minimize(
        neg_log_likelihood,
        theta0,
        jac=grad_func,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    s = result.x[:n]
    d = result.x[n:]
    # Center skills at zero; difficulties stay on the same relative scale
    s -= s.mean()
    return s, tossup_slugs, d


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _bootstrap_chunk(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Worker: runs a chunk of bootstrap iterations (no-difficulty model). Top-level for pickling.

    Resamples whole games (cluster bootstrap) so within-game dependence is respected.
    games is a list of lists; each inner list is all records from one game.
    Dead-tossup records in the sample are ignored (winner_idx=None skips the filter).
    """
    games, players, n_iter, lam, seed = args
    rng = np.random.default_rng(seed)
    n_games = len(games)
    n_players = len(players)

    skill_samples = np.zeros((n_iter, n_players))
    rank_samples  = np.zeros((n_iter, n_players), dtype=int)

    for b in range(n_iter):
        game_indices = rng.integers(0, n_games, size=n_games)
        sample = [rec for gi in game_indices for rec in games[gi]]

        present_global = sorted({gi for rec in sample for gi in rec["eligible"]})
        if len(present_global) < 2:
            rank_samples[b] = n_players
            continue

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
            rank_samples[b] = n_players
            continue

        present_players = [players[gi] for gi in present_global]
        local_skills = fit_plackett_luce(local_records, present_players, lam=lam)

        full_skills = np.zeros(n_players)
        for li, gi in enumerate(present_global):
            full_skills[gi] = local_skills[li]
        skill_samples[b] = full_skills
        rank_samples[b] = scipy.stats.rankdata(-full_skills, method="min")

    return skill_samples, rank_samples


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
    Game-level cluster bootstrap: resample whole games (all tossups from a matchup)
    with replacement, then refit PL. This respects within-game dependence (shared players,
    moderator pace, etc.) and produces wider, more honest CIs than event-level resampling.

    Parallelises across CPU cores — each core gets an independent chunk of iterations.
    Players absent from a bootstrap sample are assigned skill = 0 (L2 prior mean).

    Returns a DataFrame indexed by player_slug with columns:
        ci_low, ci_high       — 2.5th / 97.5th percentile of skill across samples
        p_top{k} for each k in rank_thresholds
    """
    # Group records by game_id for cluster resampling
    game_buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        game_buckets[rec["game_id"]].append(rec)
    games = list(game_buckets.values())

    n_workers = multiprocessing.cpu_count() - 2
    # Split iterations as evenly as possible across workers
    base, remainder = divmod(n_bootstrap, n_workers)
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(n_workers)]
    # Each worker gets a well-separated seed so their RNG streams don't overlap
    worker_args = [
        (games, players, chunk_sizes[i], lam, seed + i * 100_000)
        for i in range(n_workers)
    ]

    if verbose:
        print(f"  Cluster bootstrap: {len(games)} games, {n_bootstrap} iterations across {n_workers} cores...", flush=True)

    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(_bootstrap_chunk, worker_args)

    skill_samples = np.vstack([r[0] for r in results])
    rank_samples  = np.vstack([r[1] for r in results])

    ci_low  = np.percentile(skill_samples, 2.5,  axis=0)
    ci_high = np.percentile(skill_samples, 97.5, axis=0)

    result = {"ci_low": ci_low, "ci_high": ci_high}
    for k in rank_thresholds:
        result[f"p_top{k}"] = (rank_samples <= k).mean(axis=0)

    return pd.DataFrame(result, index=players)


def _bootstrap_chunk_with_difficulty(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Worker: runs a chunk of bootstrap iterations for the difficulty model. Top-level for pickling."""
    games, players, n_iter, lam, lam_d, seed = args
    rng = np.random.default_rng(seed)
    n_games = len(games)
    n_players = len(players)

    skill_samples = np.zeros((n_iter, n_players))
    rank_samples  = np.zeros((n_iter, n_players), dtype=int)

    for b in range(n_iter):
        game_indices = rng.integers(0, n_games, size=n_games)
        sample = [rec for gi in game_indices for rec in games[gi]]

        present_global = sorted({pi for rec in sample for pi in rec["eligible"]})
        if len(present_global) < 2:
            rank_samples[b] = n_players
            continue

        global_to_local = {gi: li for li, gi in enumerate(present_global)}
        local_records = []
        for rec in sample:
            dead = rec.get("dead", False)
            local_elig = [global_to_local[gi] for gi in rec["eligible"]
                          if gi in global_to_local]
            if not local_elig:
                continue
            if dead:
                local_records.append({
                    "tossup_slug": rec["tossup_slug"],
                    "eligible": local_elig,
                    "dead": True,
                })
            else:
                w_global = rec["winner_idx"]
                if w_global not in global_to_local:
                    continue
                local_winner = global_to_local[w_global]
                if local_winner not in local_elig:
                    continue
                local_records.append({
                    "tossup_slug": rec["tossup_slug"],
                    "eligible": local_elig,
                    "winner_idx": local_winner,
                    "dead": False,
                })

        if not local_records:
            rank_samples[b] = n_players
            continue

        present_players = [players[gi] for gi in present_global]
        local_skills, _, _ = fit_plackett_luce_with_difficulty(
            local_records, present_players, lam=lam, lam_d=lam_d
        )

        full_skills = np.zeros(n_players)
        for li, gi in enumerate(present_global):
            full_skills[gi] = local_skills[li]
        skill_samples[b] = full_skills
        rank_samples[b] = scipy.stats.rankdata(-full_skills, method="min")

    return skill_samples, rank_samples


def bootstrap_plackett_luce_with_difficulty(
    records: list[dict],
    players: list[str],
    n_bootstrap: int = 200,
    lam: float = 0.1,
    lam_d: float = 0.1,
    seed: int = 42,
    rank_thresholds: tuple[int, ...] = (5, 10, 20),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Game-level cluster bootstrap for the difficulty model.
    Resamples whole games, refits both skill and difficulty parameters, reports skill CIs only.
    """
    game_buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        game_buckets[rec["game_id"]].append(rec)
    games = list(game_buckets.values())

    n_workers = multiprocessing.cpu_count()
    base, remainder = divmod(n_bootstrap, n_workers)
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(n_workers)]
    worker_args = [
        (games, players, chunk_sizes[i], lam, lam_d, seed + i * 100_000)
        for i in range(n_workers)
    ]

    if verbose:
        print(f"  Cluster bootstrap (difficulty model): {len(games)} games, "
              f"{n_bootstrap} iterations across {n_workers} cores...", flush=True)

    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(_bootstrap_chunk_with_difficulty, worker_args)

    skill_samples = np.vstack([r[0] for r in results])
    rank_samples  = np.vstack([r[1] for r in results])

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


def build_tossup_records(
    df: pd.DataFrame,
    include_dead: bool = True,
) -> tuple[list[dict], list[str]]:
    """
    Returns (records, players) where:
      records: list of {tossup_slug, game_id, category, winner_slug, winner_idx,
                        eligible: [idx], dead: bool}
      players: ordered list of player slugs (index = player_idx)

    Neg handling: players who negged are removed from the eligible set rather than
    discarding the event. The winner faced the reduced field — PL handles variable set sizes.

    Dead tossups (no correct buzz): included when include_dead=True.  These are valid
    observations for the difficulty model — the tossup "won" the race.  A neg-then-dead
    event is also included: the negger is excluded from eligible, and winner_idx is None.
    """
    players = sorted(df["player_slug"].unique())
    pidx = {p: i for i, p in enumerate(players)}

    records = []
    skipped_no_winner = 0
    skipped_trivial = 0

    group_cols = ["tossup_slug", "game_id"] if "game_id" in df.columns else ["tossup_slug"]
    for _, grp in df.groupby(group_cols):
        category = grp["category"].iloc[0]
        game_id = grp["game_id"].iloc[0] if "game_id" in grp.columns else grp["tossup_slug"].iloc[0]
        tossup_slug = grp["tossup_slug"].iloc[0]

        # Remove negging players from the eligible set — they can't win after a neg.
        # The winner faced the remaining field; PL handles variable eligible-set sizes natively.
        neg_slugs = set(grp.loc[grp["buzz_value"] == -5, "player_slug"])
        eligible_slugs = [p for p in grp["player_slug"].unique() if p not in neg_slugs]
        eligible_idxs = [pidx[p] for p in eligible_slugs]

        winners = grp[grp["won"] == True]
        if len(winners) == 0:
            # Dead tossup: no correct buzz in this game
            if not include_dead:
                skipped_no_winner += 1
                continue
            if len(eligible_idxs) == 0:
                # Everyone negated — no meaningful observation
                skipped_trivial += 1
                continue
            records.append({
                "tossup_slug": tossup_slug,
                "game_id": game_id,
                "category": category,
                "winner_slug": None,
                "winner_idx": None,
                "eligible": eligible_idxs,
                "dead": True,
            })
        else:
            winner_slug = winners.iloc[0]["player_slug"]
            winner_idx = pidx[winner_slug]

            if len(eligible_idxs) <= 1:
                skipped_trivial += 1
                continue

            records.append({
                "tossup_slug": tossup_slug,
                "game_id": game_id,
                "category": category,
                "winner_slug": winner_slug,
                "winner_idx": winner_idx,
                "eligible": eligible_idxs,
                "dead": False,
            })

    n_won  = sum(not r["dead"] for r in records)
    n_dead = sum(r["dead"] for r in records)
    print(f"Usable tossup events: {len(records)} ({n_won} won, {n_dead} dead)  "
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
# Cross-validation for lam_d selection
# ---------------------------------------------------------------------------

LAM_D_GRID = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0)


def cross_validate_lam_d(
    records: list[dict],
    players: list[str],
    lam: float = 0.1,
    lam_d_grid: tuple[float, ...] = LAM_D_GRID,
    k: int = 5,
    seed: int = 42,
) -> float:
    """
    K-fold cross-validation (game-level splits) to select lam_d.

    Trains on k-1 folds of games, evaluates held-out per-event log-likelihood on the
    remaining fold.  Tossups unseen during training are assigned difficulty 0 (L2 prior
    mean); players unseen during training are assigned skill 0.

    Returns the lam_d value with the highest mean held-out log-likelihood.
    """
    game_buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        game_buckets[rec["game_id"]].append(rec)
    game_ids = np.array(list(game_buckets.keys()))

    rng = np.random.default_rng(seed)
    rng.shuffle(game_ids)
    folds = np.array_split(game_ids, k)

    scores: dict[float, list[float]] = {ld: [] for ld in lam_d_grid}

    for fold_idx in range(k):
        val_set   = set(folds[fold_idx])
        train_set = set(game_ids) - val_set

        train_records = [r for r in records if r["game_id"] in train_set]
        val_records   = [r for r in records if r["game_id"] in val_set]
        if not train_records or not val_records:
            continue

        for lam_d in lam_d_grid:
            skills, tossup_slugs, difficulties = fit_plackett_luce_with_difficulty(
                train_records, players, lam=lam, lam_d=lam_d
            )
            diff_map = dict(zip(tossup_slugs, difficulties))

            ll = 0.0
            n_eval = 0
            for rec in val_records:
                elig = rec["eligible"]
                if not elig:
                    continue
                d_t = diff_map.get(rec["tossup_slug"], 0.0)
                vals = np.empty(1 + len(elig))
                vals[0] = d_t
                vals[1:] = skills[np.array(elig)]
                max_v = np.max(vals)
                lse = max_v + np.log(np.sum(np.exp(vals - max_v)))
                ll += (d_t if rec.get("dead", False) else skills[rec["winner_idx"]]) - lse
                n_eval += 1

            if n_eval:
                scores[lam_d].append(ll / n_eval)

    mean_scores = {ld: float(np.mean(v)) for ld, v in scores.items() if v}
    best = max(mean_scores, key=mean_scores.__getitem__)

    print("  CV held-out log-likelihood per event:")
    for ld in lam_d_grid:
        mark = " <-- best" if ld == best else ""
        print(f"    lam_d={ld:<6}: {mean_scores.get(ld, float('nan')):.4f}{mark}")

    return best


# ---------------------------------------------------------------------------
# Difficulty table helpers
# ---------------------------------------------------------------------------

def build_difficulty_table(
    records: list[dict],
    tossup_slugs: list[str],
    difficulties: np.ndarray,
) -> pd.DataFrame:
    """Build a per-tossup difficulty summary from a fitted difficulty model."""
    slug_to_idx = {sl: i for i, sl in enumerate(tossup_slugs)}
    times_heard: dict[str, int] = defaultdict(int)
    dead_count:  dict[str, int] = defaultdict(int)
    category:    dict[str, str] = {}

    for rec in records:
        sl = rec["tossup_slug"]
        times_heard[sl] += 1
        if rec.get("dead", False):
            dead_count[sl] += 1
        if sl not in category:
            category[sl] = rec.get("category", "")

    rows = []
    for sl in tossup_slugs:
        th = times_heard[sl]
        rows.append({
            "tossup_slug": sl,
            "category": category.get(sl, ""),
            "difficulty": difficulties[slug_to_idx[sl]],
            "times_heard": th,
            "dead_rate": dead_count[sl] / th if th > 0 else 0.0,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("difficulty", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Overall model
# ---------------------------------------------------------------------------

def run_overall_model(
    df: pd.DataFrame,
    lam: float = 0.1,
    lam_d: float = 0.1,
    n_bootstrap: int = 200,
    use_difficulty: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Returns (result, boot, difficulty_table).

    result            — point-estimate player ranking (1-indexed)
    boot              — bootstrap summary (ci_low, ci_high, p_top5/10/20), or None
    difficulty_table  — per-tossup difficulty estimates, or None when use_difficulty=False
    """
    model_label = "difficulty" if use_difficulty else "no-difficulty (fallback)"
    print(f"\n=== Overall Plackett-Luce Model [{model_label}] ===")
    records, players = build_tossup_records(df, include_dead=use_difficulty)

    if use_difficulty:
        skills, tossup_slugs, difficulties = fit_plackett_luce_with_difficulty(
            records, players, lam=lam, lam_d=lam_d
        )
        diff_table = build_difficulty_table(records, tossup_slugs, difficulties)
    else:
        skills = fit_plackett_luce(records, players, lam=lam)
        diff_table = None

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
        if use_difficulty:
            boot = bootstrap_plackett_luce_with_difficulty(
                records, players, n_bootstrap=n_bootstrap, lam=lam, lam_d=lam_d
            )
        else:
            boot = bootstrap_plackett_luce(records, players, n_bootstrap=n_bootstrap, lam=lam)
        print(f"  Overall bootstrap done in {time.perf_counter() - t0:.1f}s")
    else:
        boot = None

    return result, boot, diff_table


# ---------------------------------------------------------------------------
# Per-category models
# ---------------------------------------------------------------------------

def run_category_models(
    df: pd.DataFrame,
    lam: float = 0.1,
    lam_d: float = 0.1,
    n_bootstrap: int = 200,
    use_difficulty: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Returns (skill_matrix, cat_boots) where:
      skill_matrix  — players x categories, NaN below min_buzzes threshold
      cat_boots     — {category: bootstrap DataFrame with ci_low/ci_high/p_top3/p_top5}
                      only rows for players who qualified in the point estimate
    """
    model_label = "difficulty" if use_difficulty else "no-difficulty (fallback)"
    print(f"\n=== Per-Category Plackett-Luce Models [{model_label}] ===")
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
        records, cat_players = build_tossup_records(cat_df, include_dead=use_difficulty)

        if len(records) < 5:
            print(f"  {cat}: too few tossups ({len(records)}), skipping")
            continue

        if use_difficulty:
            skills_arr, _, _ = fit_plackett_luce_with_difficulty(
                records, cat_players, lam=lam, lam_d=lam_d
            )
        else:
            skills_arr = fit_plackett_luce(records, cat_players, lam=lam)

        for ps, sk in zip(cat_players, skills_arr):
            n_gets = gets_by_cat.get((ps, cat), 0)
            if n_gets >= threshold:
                data[cat][ps] = float(sk)

        qualified = list(data[cat].keys())
        print(f"  {cat} (min_buzzes={threshold}): {len(records)} tossups, {len(qualified)} players qualify")

        if n_bootstrap > 0 and len(records) >= 5:
            t0 = time.perf_counter()
            if use_difficulty:
                boot = bootstrap_plackett_luce_with_difficulty(
                    records, cat_players, n_bootstrap=n_bootstrap, lam=lam, lam_d=lam_d,
                    rank_thresholds=(3, 5), verbose=False,
                )
            else:
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


REPORT_TOP_DIFFICULTY = 40


def generate_report(
    overall_result: pd.DataFrame,
    skill_matrix: pd.DataFrame,
    boot: pd.DataFrame | None = None,
    cat_boots: dict[str, pd.DataFrame] | None = None,
    output_dir: str = ".",
    difficulty_table: pd.DataFrame | None = None,
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

    # Tossup difficulty table
    if difficulty_table is not None and not difficulty_table.empty:
        n_d = min(REPORT_TOP_DIFFICULTY, len(difficulty_table))
        lines.append(f"\n## Tossup Difficulty Estimates (Top {n_d} Hardest)\n")
        lines.append("| Rank | Tossup | Category | Difficulty | Times Heard | Dead Rate |")
        lines.append("|------|--------|----------|----------:|------------:|----------:|")
        for i, row in difficulty_table.head(n_d).iterrows():
            lines.append(
                f"| {i + 1} | {row['tossup_slug']} | {row['category']}"
                f" | {row['difficulty']:.3f} | {row['times_heard']}"
                f" | {row['dead_rate']:.1%} |"
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
    parser.add_argument("--lam", type=float, default=0.1, help="L2 regularization lambda for player skill")
    parser.add_argument("--lam-d", type=float, default=None,
                        help="L2 lambda for tossup difficulty (default: chosen by 5-fold CV)")
    parser.add_argument("--bootstrap", type=int, default=200, help="Number of bootstrap iterations (0 to skip)")
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    parser.add_argument("--aggregate-categories", action="store_true",
                        help="Aggregate subcategories (e.g. Science - Biology -> Science)")
    parser.add_argument("--no-aggregate", dest="aggregate_categories", action="store_false")
    parser.set_defaults(aggregate_categories=True)
    parser.add_argument("--no-difficulty", action="store_true",
                        help="Fall back to the original model without tossup difficulty parameters")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    use_difficulty = not args.no_difficulty

    # Load
    df = load_data(args.input, aggregate=args.aggregate_categories)

    # Select lam_d via CV (skipped when --no-difficulty or --lam-d is explicit)
    lam_d = args.lam_d
    if use_difficulty and lam_d is None:
        print("\n=== Selecting lam_d via 5-fold cross-validation ===")
        records_for_cv, players_for_cv = build_tossup_records(df, include_dead=True)
        t0 = time.perf_counter()
        lam_d = cross_validate_lam_d(records_for_cv, players_for_cv, lam=args.lam)
        print(f"  Selected lam_d={lam_d} in {time.perf_counter() - t0:.1f}s")
    elif not use_difficulty:
        lam_d = 0.1  # unused but keep a valid float for the call signatures

    # Overall model + bootstrap
    overall_result, boot, diff_table = run_overall_model(
        df, lam=args.lam, lam_d=lam_d, n_bootstrap=args.bootstrap,
        use_difficulty=use_difficulty,
    )
    overall_out = f"{args.output_dir}/overall_rankings.csv"
    overall_result.to_csv(overall_out)
    print(f"\nSaved overall rankings to {overall_out}")
    if boot is not None:
        boot.to_csv(f"{args.output_dir}/bootstrap_overall.csv")
        print(f"Saved bootstrap summary to {args.output_dir}/bootstrap_overall.csv")
    if diff_table is not None:
        diff_out = f"{args.output_dir}/tossup_difficulty.csv"
        diff_table.to_csv(diff_out, index=False)
        print(f"Saved tossup difficulty table to {diff_out}")

    # Category models + bootstrap
    skill_matrix, cat_boots = run_category_models(
        df, lam=args.lam, lam_d=lam_d, n_bootstrap=args.bootstrap,
        use_difficulty=use_difficulty,
    )
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
        report_diff = diff_table
    else:
        print("\nBuilding aggregated data for report...")
        df_agg = load_data(args.input, aggregate=True)
        report_overall, report_boot, report_diff = run_overall_model(
            df_agg, lam=args.lam, lam_d=lam_d, n_bootstrap=args.bootstrap,
            use_difficulty=use_difficulty,
        )
        report_matrix, report_cat_boots = run_category_models(
            df_agg, lam=args.lam, lam_d=lam_d, n_bootstrap=args.bootstrap,
            use_difficulty=use_difficulty,
        )
    generate_report(report_overall, report_matrix, boot=report_boot,
                    cat_boots=report_cat_boots, output_dir=args.output_dir,
                    difficulty_table=report_diff)

    print("\nDone.")


if __name__ == "__main__":
    main()
