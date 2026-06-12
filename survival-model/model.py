"""
Buzz-position-aware competing-risks survival model for quizbowl ranking.

Discrete-time grouped-logistic hazard formulation. For each (tossup, game),
multiple players race to buzz; the first correct buzz wins. Negs and other
wrong buzzes act as right-censoring for the negging player. The tossup
itself acts as a virtual competitor whose "buzz" event is the tossup going
dead at the end of the question (length-out censoring with a difficulty
offset).

Model:
    For player i at bin b on tossup t (given still in race):
        log h_i(b, t) = a_b + s_i - d_t              (player hazard)
    Tossup-end "hazard" only fires at the final bin if no one buzzed:
        the dead event has prob equivalent to a tossup difficulty parameter
        d_t entering as a competitor with constant hazard exp(-d_t).

Concretely we use a per-bin grouped multinomial-logit:
    At each bin b that the race reaches, exactly one of the following:
      (a) some still-in-race player j buzzes correctly  -> they win
      (b) some still-in-race player j negs              -> they exit
      (c) no event in this bin                          -> race continues
      (d) end-of-question reached & nobody won          -> dead (only at b=B_t)

We model (a)+(b) jointly: every "buzz" by player j at bin b (correct OR neg)
contributes the *same* hazard signal s_j (the player's propensity to commit at
this point). The correct-vs-neg split is conditioned on the buzz happening
and treated separately (a per-player accuracy, NOT modelled in v1: the HANDOFF
defers neg-rate-vs-skill to a nice-to-have). For (a)+(b) combined the hazard
structure above gives the standard discrete-time competing-risks likelihood.

Tossup difficulty enters by treating "tossup goes dead at length B_t" as a
competing event whose log-hazard at every bin is -d_t (so larger d_t = harder
question = more dead-prone). This mirrors the Plackett-Luce difficulty model,
extended to the per-bin grouped-logit setting.

Usage:
    python model.py --input ../buzzpoints_data.csv
    python model.py --input ../buzzpoints_data.csv --bin-width 5 --bootstrap 100
"""

import argparse
import gc
import multiprocessing
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)


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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str, aggregate: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["won"] = df["won"].map({"True": True, "False": False, True: True, False: False})
    for col in ("buzzpoint", "buzz_value"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if aggregate:
        df["category"] = df["category"].apply(aggregate_category)
    print(f"Loaded {len(df)} rows, {df['tossup_slug'].nunique()} tossups, "
          f"{df['player_slug'].nunique()} players, {df['category'].nunique()} categories")
    return df


def build_event_records(df: pd.DataFrame, bin_width: int = 5) -> tuple[list[dict], list[str], list[str], int]:
    """
    Produce one record per (tossup, game) with the discrete-time event sequence.

    Each record:
        tossup_slug, game_id, category,
        length_bin: int (how many bins the race reaches; max bin index = length_bin - 1),
        events: list of (bin_index, player_idx, kind) with kind in {"win","exit"}
                Players still in the race after their last listed bin remain at risk
                until length_bin (then they are censored by the "dead" outcome).
        winner_idx: int or None
        dead: bool   (True iff no win occurred -> tossup-end event at bin (length_bin-1))
        eligible: list[int] -- all players who heard the question

    bin_width: width in raw buzzpoint units. position p falls in bin floor(p / bin_width).
    """
    players = sorted(df["player_slug"].unique())
    pidx = {p: i for i, p in enumerate(players)}

    # tossup length proxy: max observed buzzpoint across ALL games of that tossup,
    # rounded UP to next bin. Used when a (tossup,game) has no buzzes (true dead) and
    # we need a length to censor at.
    bp_per_tossup = df.groupby("tossup_slug")["buzzpoint"].max()
    tossup_length_bin: dict[str, int] = {}
    for sl, mx in bp_per_tossup.items():
        if pd.isna(mx):
            mx = 0
        tossup_length_bin[sl] = int(mx) // bin_width + 1  # at least 1 bin

    tossup_slugs = sorted(df["tossup_slug"].unique())
    max_length_bin = max(tossup_length_bin.values())

    records = []
    skipped_trivial = 0
    for (sl, gid), grp in df.groupby(["tossup_slug", "game_id"]):
        eligible_slugs = list(grp["player_slug"].unique())
        eligible = [pidx[p] for p in eligible_slugs]
        if len(eligible) < 2:
            skipped_trivial += 1
            continue

        # Per-player events
        events = []  # (bin_idx, pidx, kind)
        winner_idx = None
        for _, r in grp.iterrows():
            bv = r["buzz_value"]
            bp = r["buzzpoint"]
            if pd.isna(bv) or pd.isna(bp):
                continue  # heard but never buzzed -> stays in race to end
            b = int(bp) // bin_width
            if bv == 10:  # correct
                kind = "win"
                winner_idx = pidx[r["player_slug"]]
            else:  # -5 (neg) or 0 (wrong, no neg) -> exit / right-censor
                kind = "exit"
            events.append((b, pidx[r["player_slug"]], kind))
        # Sort by bin
        events.sort(key=lambda e: (e[0], 0 if e[2] == "exit" else 1))
        # If there's a winner, truncate any events after the win (they shouldn't exist
        # in clean data but be safe)
        if winner_idx is not None:
            win_bin = next(b for b, p, k in events if p == winner_idx and k == "win")
            events = [e for e in events if e[0] <= win_bin]

        # length_bin: where the race "ends" if no win occurred. Use the max observed bin
        # in this game if any, else fall back to the per-tossup global max.
        if events:
            last_bin = max(e[0] for e in events) + 1  # next bin after last event
            length_bin = max(last_bin, tossup_length_bin[sl])
        else:
            length_bin = tossup_length_bin[sl]
        length_bin = min(length_bin, max_length_bin)

        records.append({
            "tossup_slug": sl,
            "game_id": gid,
            "category": grp["category"].iloc[0],
            "eligible": eligible,
            "events": events,
            "winner_idx": winner_idx,
            "dead": winner_idx is None,
            "length_bin": length_bin,
        })

    n_won = sum(not r["dead"] for r in records)
    n_dead = sum(r["dead"] for r in records)
    print(f"Built {len(records)} (tossup,game) events ({n_won} won, {n_dead} dead, "
          f"skipped {skipped_trivial} trivial). "
          f"Bin width={bin_width}, max bin index={max_length_bin}.")
    return records, players, tossup_slugs, max_length_bin


# ---------------------------------------------------------------------------
# Likelihood for the discrete-time competing-risks hazard model
# ---------------------------------------------------------------------------
#
# Parameters:
#   s     in R^N  (player log-hazard offsets)
#   d     in R^M  (tossup log-hazard offsets, larger d = harder)
#   a     in R^B  (per-bin baseline log-hazards)
#
# At bin b for player j still in race on tossup t: hazard offset = s_j + a_b
# Tossup-dead competing risk: hazard offset = -d_t + a_b at bin (B_t - 1) only.
#
# Per-bin likelihood (grouped multinomial-logit at the bin):
#   denominator at bin b on (t, game):
#       Z = 1 + sum_{j in risk(b)} exp(s_j + a_b)
#                  + (b == B_t-1 ? exp(-d_t + a_b) : 0)
#   The "1" is the no-event slot. (Equivalently, the shared a_b is the log-odds
#   of an event vs no event with all hazards normalized.)
#   At the bin where an event occurs, that event's exp(...) sits in the numerator;
#   all other bins (where the race is still alive and nothing happened) contribute
#   the no-event slot 1/Z.
#
# This matches a standard discrete-time competing-risks likelihood.

def _flatten_records(records, n_players, n_tossups, n_bins,
                     tossup_to_idx):
    """
    Pre-compute, for each record, the per-bin risk sets and event labels.

    Returns list of dicts:
        bins:        np.ndarray shape (T,)   bin indices visited (0..length_bin-1)
        risk:        list of np.ndarray      players at risk at each bin
        event_pidx:  np.ndarray (T,) int     player index of event at this bin, or -1
        event_type:  np.ndarray (T,) int     0=no event, 1=win, 2=exit, 3=dead
        tossup_idx:  int
        length_bin:  int
    """
    flat = []
    for rec in records:
        ti = tossup_to_idx[rec["tossup_slug"]]
        L = rec["length_bin"]
        # state: at_risk set evolves bin by bin
        # initialize: all eligible players at risk
        at_risk = set(rec["eligible"])
        # events grouped by bin
        ev_by_bin: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for b, p, k in rec["events"]:
            ev_by_bin[b].append((p, k))
        bins = np.arange(L, dtype=np.int64)
        risk_sets = []
        event_pidx = np.full(L, -1, dtype=np.int64)
        event_type = np.zeros(L, dtype=np.int64)
        for b in range(L):
            risk_sets.append(np.array(sorted(at_risk), dtype=np.int64))
            evs = ev_by_bin.get(b, [])
            # Process at most one event per bin; if multiple, the WIN takes precedence
            # (others would be exits at the same bin which is rare; treat them as occurring
            # immediately after — we'll merge them onto same bin but only one becomes the
            # "labelled" event for the multinomial slot. To stay statistically clean, we
            # log the win and drop concurrent exits as no-event. v1 simplification.)
            chosen = None
            for p, k in evs:
                if k == "win":
                    chosen = (p, k); break
            if chosen is None and evs:
                chosen = evs[0]
            if chosen is not None:
                p, k = chosen
                if k == "win":
                    event_pidx[b] = p
                    event_type[b] = 1
                    # race ends here
                    at_risk = set()
                else:
                    event_pidx[b] = p
                    event_type[b] = 2
                    at_risk.discard(p)
                # remove all other concurrent exits silently from at_risk
                for p2, k2 in evs:
                    if (p2, k2) != chosen and k2 == "exit":
                        at_risk.discard(p2)
                if k == "win":
                    # truncate: don't extend beyond
                    risk_sets = risk_sets[:b+1]
                    bins = bins[:b+1]
                    event_pidx = event_pidx[:b+1]
                    event_type = event_type[:b+1]
                    L = b + 1
                    break
            # else no event in this bin
        # If race reached final bin without a winner, that final bin's outcome is "dead"
        if rec["dead"] and L > 0 and event_type[L - 1] == 0:
            event_type[L - 1] = 3  # dead
        flat.append({
            "tossup_idx": ti,
            "bins": bins,
            "risk": risk_sets,
            "event_pidx": event_pidx,
            "event_type": event_type,
            "length_bin": L,
        })
    return flat


def neg_log_lik_and_grad(theta, flat, n_players, n_tossups, n_bins,
                         lam_s=0.1, lam_d=0.1, lam_a=0.01):
    """
    theta = [s (N), d (M), a (B)]
    """
    s = theta[:n_players]
    d = theta[n_players:n_players + n_tossups]
    a = theta[n_players + n_tossups:]

    nll = 0.0
    g_s = np.zeros_like(s)
    g_d = np.zeros_like(d)
    g_a = np.zeros_like(a)

    for rec in flat:
        ti = rec["tossup_idx"]
        d_t = d[ti]
        L = rec["length_bin"]
        if L == 0:
            continue
        for b in range(L):
            risk = rec["risk"][b]
            if len(risk) == 0:
                continue
            ab = a[b]
            # hazard contributions
            h_players = np.exp(s[risk] + ab)  # (R,)
            # tossup-dead competing risk only fires on final bin AND only if dead label there
            dead_active = (rec["event_type"][b] == 3) or (b == L - 1 and rec["event_type"][b] == 0)
            # Actually the dead competing-risk "is at risk" on the final bin always when
            # no winner yet. The label tells us whether the dead event fires. But for the
            # multinomial denominator we include it as a competitor whenever b == L-1
            # AND no win has occurred (which is true since we're still iterating).
            include_dead_in_Z = (b == L - 1)
            if include_dead_in_Z:
                h_dead = np.exp(-d_t + ab)
            else:
                h_dead = 0.0
            Z = 1.0 + h_players.sum() + h_dead
            et = rec["event_type"][b]
            if et == 0:
                # no event slot
                nll += np.log(Z)  # -log(1/Z) = log Z
                # gradient: d/dtheta log Z
                weights = h_players / Z
                g_s[risk] += weights
                g_a[b] += (h_players.sum() + h_dead) / Z
                if include_dead_in_Z:
                    g_d[ti] += -h_dead / Z
            elif et == 1:
                ep = rec["event_pidx"][b]
                # numerator: exp(s[ep] + a_b)
                nll += np.log(Z) - (s[ep] + ab)
                weights = h_players / Z
                g_s[risk] += weights
                g_s[ep] -= 1.0
                g_a[b] += (h_players.sum() + h_dead) / Z - 1.0
                if include_dead_in_Z:
                    g_d[ti] += -h_dead / Z
            elif et == 2:
                ep = rec["event_pidx"][b]
                nll += np.log(Z) - (s[ep] + ab)
                weights = h_players / Z
                g_s[risk] += weights
                g_s[ep] -= 1.0
                g_a[b] += (h_players.sum() + h_dead) / Z - 1.0
                if include_dead_in_Z:
                    g_d[ti] += -h_dead / Z
            elif et == 3:
                # dead event
                nll += np.log(Z) - (-d_t + ab)
                weights = h_players / Z
                g_s[risk] += weights
                g_a[b] += (h_players.sum() + h_dead) / Z - 1.0
                # dead numerator gradient w.r.t d_t: -1
                g_d[ti] += -h_dead / Z - (-1.0)  # = -h_dead/Z + 1

    # L2 regularization
    nll += lam_s * np.dot(s, s) + lam_d * np.dot(d, d) + lam_a * np.dot(a, a)
    g_s += 2.0 * lam_s * s
    g_d += 2.0 * lam_d * d
    g_a += 2.0 * lam_a * a

    return nll, np.concatenate([g_s, g_d, g_a])


def fit_survival_model(records, players, tossup_slugs, n_bins,
                       lam_s=0.1, lam_d=0.1, lam_a=0.01, verbose=True):
    n = len(players)
    m = len(tossup_slugs)
    tossup_to_idx = {sl: i for i, sl in enumerate(tossup_slugs)}
    flat = _flatten_records(records, n, m, n_bins, tossup_to_idx)
    theta0 = np.zeros(n + m + n_bins)
    if verbose:
        print(f"  Fitting: N={n} players, M={m} tossups, B={n_bins} bins, "
              f"{sum(r['length_bin'] for r in flat)} bin-observations")
    t0 = time.perf_counter()
    res = scipy.optimize.minimize(
        lambda th: neg_log_lik_and_grad(th, flat, n, m, n_bins, lam_s, lam_d, lam_a),
        theta0, jac=True, method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-8, "gtol": 1e-6},
    )
    s = res.x[:n]
    d = res.x[n:n + m]
    a = res.x[n + m:]
    s_centered = s - s.mean()
    if verbose:
        print(f"  Fit done in {time.perf_counter()-t0:.1f}s, nll={res.fun:.2f}, "
              f"converged={res.success}")
    return s_centered, d, a, flat


# ---------------------------------------------------------------------------
# Cluster bootstrap (game-level)
# ---------------------------------------------------------------------------

def _bootstrap_chunk(args):
    (games, players, tossup_slugs, n_bins, n_iter, lam_s, lam_d, lam_a, seed,
     worker_id) = args
    rng = np.random.default_rng(seed)
    n_games = len(games)
    n_players = len(players)
    skill_samples = np.zeros((n_iter, n_players))
    rank_samples = np.zeros((n_iter, n_players), dtype=int)
    chunk_t0 = time.perf_counter()
    for b in range(n_iter):
        iter_t0 = time.perf_counter()
        gi = rng.integers(0, n_games, size=n_games)
        sample = [rec for k in gi for rec in games[k]]
        present = sorted({p for rec in sample for p in rec["eligible"]})
        if len(present) < 2:
            rank_samples[b] = n_players
            continue
        g2l = {g: l for l, g in enumerate(present)}
        local_records = []
        for rec in sample:
            local_elig = [g2l[g] for g in rec["eligible"] if g in g2l]
            if len(local_elig) < 2:
                continue
            local_events = []
            for bin_, p, k in rec["events"]:
                if p in g2l:
                    local_events.append((bin_, g2l[p], k))
            local_winner = g2l.get(rec["winner_idx"]) if rec["winner_idx"] is not None and rec["winner_idx"] in g2l else None
            local_dead = local_winner is None
            local_records.append({
                "tossup_slug": rec["tossup_slug"],
                "game_id": rec["game_id"],
                "category": rec["category"],
                "eligible": local_elig,
                "events": local_events,
                "winner_idx": local_winner,
                "dead": local_dead,
                "length_bin": rec["length_bin"],
            })
        if not local_records:
            rank_samples[b] = n_players
            continue
        local_players = [players[g] for g in present]
        local_tossups = sorted({r["tossup_slug"] for r in local_records})
        try:
            s_local, _, _, flat_local = fit_survival_model(
                local_records, local_players, local_tossups, n_bins,
                lam_s=lam_s, lam_d=lam_d, lam_a=lam_a, verbose=False,
            )
        except Exception:
            rank_samples[b] = n_players
            del local_records
            gc.collect()
            continue
        full = np.zeros(n_players)
        for l, g in enumerate(present):
            full[g] = s_local[l]
        skill_samples[b] = full
        rank_samples[b] = scipy.stats.rankdata(-full, method="min")
        # Free the per-bin risk arrays/event labels — these are the bulk of memory.
        del local_records, flat_local, s_local
        gc.collect()
        # Per-iter progress line. flush=True so it shows up in real time.
        elapsed = time.perf_counter() - iter_t0
        total = time.perf_counter() - chunk_t0
        rate = (b + 1) / total if total > 0 else 0
        eta = (n_iter - (b + 1)) / rate if rate > 0 else float("inf")
        print(f"  [worker {worker_id}] iter {b+1}/{n_iter} done in {elapsed:.0f}s "
              f"(total {total:.0f}s, ETA {eta:.0f}s)", flush=True)
    return skill_samples, rank_samples


def bootstrap_survival(records, players, tossup_slugs, n_bins,
                       n_bootstrap=100, lam_s=0.1, lam_d=0.1, lam_a=0.01,
                       seed=42, rank_thresholds=(5, 10, 20), verbose=True,
                       n_workers=None):
    by_game: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_game[r["game_id"]].append(r)
    games = list(by_game.values())
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 2)
    # Don't spin up more workers than there are iterations.
    n_workers = max(1, min(n_workers, n_bootstrap))
    base, rem = divmod(n_bootstrap, n_workers)
    chunks = [base + (1 if i < rem else 0) for i in range(n_workers)]
    args = [(games, players, tossup_slugs, n_bins, chunks[i], lam_s, lam_d, lam_a,
             seed + i * 100_000, i) for i in range(n_workers)]
    if verbose:
        print(f"  Cluster bootstrap: {len(games)} games, {n_bootstrap} iters across {n_workers} workers")
        print(f"  Iters per worker: {chunks}", flush=True)
    t0 = time.perf_counter()
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(_bootstrap_chunk, args)
    skill_samples = np.vstack([r[0] for r in results])
    rank_samples = np.vstack([r[1] for r in results])
    if verbose:
        print(f"  Bootstrap done in {time.perf_counter()-t0:.1f}s")
    ci_low = np.percentile(skill_samples, 2.5, axis=0)
    ci_high = np.percentile(skill_samples, 97.5, axis=0)
    out = {"ci_low": ci_low, "ci_high": ci_high}
    for k in rank_thresholds:
        out[f"p_top{k}"] = (rank_samples <= k).mean(axis=0)
    return pd.DataFrame(out, index=players)


# ---------------------------------------------------------------------------
# Cross-validation for lambda selection (held-out per-event log-likelihood)
# ---------------------------------------------------------------------------

def cross_validate_lambdas(records, players, tossup_slugs, n_bins,
                           lam_s=0.1, lam_d_grid=(0.05, 0.1, 0.5, 1.0),
                           k=5, seed=42, lam_a=0.01, verbose=True):
    by_game: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_game[r["game_id"]].append(r)
    game_ids = np.array(list(by_game.keys()))
    rng = np.random.default_rng(seed)
    rng.shuffle(game_ids)
    folds = np.array_split(game_ids, k)
    scores = {ld: [] for ld in lam_d_grid}
    for f in range(k):
        val = set(folds[f])
        train = [r for r in records if r["game_id"] not in val]
        test = [r for r in records if r["game_id"] in val]
        if not train or not test:
            continue
        train_tossups = sorted({r["tossup_slug"] for r in train})
        train_t2i = {sl: i for i, sl in enumerate(train_tossups)}
        for ld in lam_d_grid:
            s, d, a, _ = fit_survival_model(
                train, players, train_tossups, n_bins,
                lam_s=lam_s, lam_d=ld, lam_a=lam_a, verbose=False,
            )
            # held-out per-bin log-likelihood
            ll = 0.0
            n_bins_eval = 0
            test_flat = _flatten_records(
                test, len(players), len(train_tossups), n_bins,
                {**train_t2i, **{sl: -1 for sl in {r["tossup_slug"] for r in test} - set(train_tossups)}},
            )
            for rec in test_flat:
                ti = rec["tossup_idx"]
                d_t = d[ti] if ti >= 0 else 0.0
                L = rec["length_bin"]
                for b in range(L):
                    risk = rec["risk"][b]
                    if len(risk) == 0:
                        continue
                    ab = a[b]
                    h_p = np.exp(s[risk] + ab)
                    inc_dead = (b == L - 1)
                    h_d = np.exp(-d_t + ab) if inc_dead else 0.0
                    Z = 1.0 + h_p.sum() + h_d
                    et = rec["event_type"][b]
                    if et == 0:
                        ll += -np.log(Z)
                    elif et in (1, 2):
                        ep = rec["event_pidx"][b]
                        ll += (s[ep] + ab) - np.log(Z)
                    elif et == 3:
                        ll += (-d_t + ab) - np.log(Z)
                    n_bins_eval += 1
            if n_bins_eval:
                scores[ld].append(ll / n_bins_eval)
    means = {ld: float(np.mean(v)) for ld, v in scores.items() if v}
    best = max(means, key=means.__getitem__)
    if verbose:
        print("  CV held-out log-likelihood per bin:")
        for ld in lam_d_grid:
            mark = " <-- best" if ld == best else ""
            print(f"    lam_d={ld:<6}: {means.get(ld, float('nan')):.4f}{mark}")
    return best


# ---------------------------------------------------------------------------
# Player profiling: residual buzz position (speed) controlling for difficulty
# ---------------------------------------------------------------------------

def player_speed_profile(df: pd.DataFrame, records, tossup_slugs,
                         difficulties, players) -> pd.DataFrame:
    """
    For each player, compute mean buzzpoint on correct buzzes minus the per-tossup
    average buzzpoint. Negative => buzzes earlier than average; positive => later.
    Reported alongside count.
    """
    # mean correct buzz per tossup (across games)
    bp = df.copy()
    bp["bp"] = pd.to_numeric(bp["buzzpoint"], errors="coerce")
    correct = bp[(bp["buzz_value"] == 10) & bp["bp"].notna()].copy()
    tossup_mean_bp = correct.groupby("tossup_slug")["bp"].mean()
    correct["tossup_mean"] = correct["tossup_slug"].map(tossup_mean_bp)
    correct["residual"] = correct["bp"] - correct["tossup_mean"]
    rows = []
    for p, sub in correct.groupby("player_slug"):
        rows.append({
            "player_slug": p,
            "team_slug": sub["team_slug"].iloc[0],
            "n_correct": len(sub),
            "mean_buzzpoint": sub["bp"].mean(),
            "mean_residual_bp": sub["residual"].mean(),  # negative = earlier
        })
    return pd.DataFrame(rows).sort_values("mean_residual_bp")


# ---------------------------------------------------------------------------
# Difficulty + baseline hazard tables
# ---------------------------------------------------------------------------

def build_difficulty_table(records, tossup_slugs, difficulties) -> pd.DataFrame:
    times_heard = defaultdict(int)
    dead_count = defaultdict(int)
    cat = {}
    for rec in records:
        sl = rec["tossup_slug"]
        times_heard[sl] += 1
        if rec["dead"]:
            dead_count[sl] += 1
        cat.setdefault(sl, rec.get("category", ""))
    rows = []
    for i, sl in enumerate(tossup_slugs):
        th = times_heard[sl]
        rows.append({
            "tossup_slug": sl,
            "category": cat.get(sl, ""),
            "difficulty": float(difficulties[i]),
            "times_heard": th,
            "dead_rate": dead_count[sl] / th if th else 0.0,
        })
    return (pd.DataFrame(rows).sort_values("difficulty", ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_baseline_hazard(a, bin_width, out_path):
    bins = np.arange(len(a)) * bin_width + bin_width / 2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bins, a, marker="o", lw=1)
    ax.set_xlabel("Buzzpoint (word index, bin midpoint)")
    ax.set_ylabel("Baseline log-hazard a_b")
    ax.set_title("Shared baseline log-hazard by position")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_buzz_distribution(df, out_path, bin_width):
    bp = pd.to_numeric(df["buzzpoint"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(bp, bins=range(0, int(bp.max()) + bin_width, bin_width), color="steelblue", edgecolor="white")
    ax.set_xlabel("Buzzpoint (word index)")
    ax.set_ylabel("Count")
    ax.set_title("Empirical buzzpoint distribution (all buzzes)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(overall, boot, diff_table, speed_profile, output_dir):
    lines = ["# Survival Model Rankings Report\n",
             "Discrete-time competing-risks hazard model on buzzpoint data.\n"]
    lines.append("## Overall (Top 30)\n")
    if boot is not None:
        lines.append("| Rank | Player | Team | Skill | 95% CI | P(Top 5) | P(Top 10) | P(Top 20) | Gets | Heard |")
        lines.append("|------|--------|------|------:|-------:|---------:|----------:|----------:|-----:|------:|")
    else:
        lines.append("| Rank | Player | Team | Skill | Gets | Heard |")
        lines.append("|------|--------|------|------:|-----:|------:|")
    for rank, row in overall.head(30).iterrows():
        p = row["player_slug"]
        s = f"| {rank} | {p} | {row['team_slug']} | {row['skill_estimate']:.3f}"
        if boot is not None and p in boot.index:
            b = boot.loc[p]
            s += f" | [{b['ci_low']:.3f}, {b['ci_high']:.3f}] | {b['p_top5']:.0%} | {b['p_top10']:.0%} | {b['p_top20']:.0%}"
        s += f" | {int(row['total_gets'])} | {int(row['total_tossups_heard'])} |"
        lines.append(s)

    lines.append("\n## Player Speed Profile (Top 20 fastest, Bottom 20 slowest)\n")
    lines.append("Residual = mean buzzpoint of player correct buzzes minus per-tossup mean. Negative = faster.\n")
    sp = speed_profile[speed_profile["n_correct"] >= 10]
    lines.append("### Fastest 20\n")
    lines.append("| Player | Team | N Correct | Mean Buzzpoint | Mean Residual |")
    lines.append("|--------|------|----------:|---------------:|--------------:|")
    for _, r in sp.head(20).iterrows():
        lines.append(f"| {r['player_slug']} | {r['team_slug']} | {int(r['n_correct'])} | {r['mean_buzzpoint']:.1f} | {r['mean_residual_bp']:+.2f} |")
    lines.append("\n### Slowest 20\n")
    lines.append("| Player | Team | N Correct | Mean Buzzpoint | Mean Residual |")
    lines.append("|--------|------|----------:|---------------:|--------------:|")
    for _, r in sp.tail(20).iloc[::-1].iterrows():
        lines.append(f"| {r['player_slug']} | {r['team_slug']} | {int(r['n_correct'])} | {r['mean_buzzpoint']:.1f} | {r['mean_residual_bp']:+.2f} |")

    if diff_table is not None and not diff_table.empty:
        lines.append("\n## Tossup Difficulty (Top 40 hardest)\n")
        lines.append("| Rank | Tossup | Category | Difficulty | Times Heard | Dead Rate |")
        lines.append("|------|--------|----------|-----------:|------------:|----------:|")
        for i, r in diff_table.head(40).iterrows():
            lines.append(f"| {i+1} | {r['tossup_slug']} | {r['category']} | {r['difficulty']:.3f} | {r['times_heard']} | {r['dead_rate']:.1%} |")

    out = f"{output_dir}/report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {out}")


def player_basic_stats(df, players):
    gets = df[df["won"] == True].groupby("player_slug").size().rename("total_gets")
    heard = df.groupby("player_slug").size().rename("total_tossups_heard")
    team = df.groupby("player_slug")["team_slug"].first()
    s = pd.DataFrame({"player_slug": players}).set_index("player_slug")
    s = s.join(gets).join(heard).join(team)
    s["total_gets"] = s["total_gets"].fillna(0).astype(int)
    s["total_tossups_heard"] = s["total_tossups_heard"].fillna(0).astype(int)
    return s.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../buzzpoints_data.csv")
    ap.add_argument("--bin-width", type=int, default=5, help="Discrete-time bin width in words")
    ap.add_argument("--lam-s", type=float, default=0.1)
    ap.add_argument("--lam-d", type=float, default=None, help="If unset, chosen by 5-fold CV")
    ap.add_argument("--lam-a", type=float, default=0.01)
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--workers", type=int, default=0,
                    help="Bootstrap pool workers. 0 = auto (cpu_count - 2). Each worker peaks at "
                         "~2-4 GB on full data; lower if OOMing.")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--skip-fit", action="store_true",
                    help="Skip the point-estimate refit and reuse overall_rankings.csv + tossup_difficulty.csv from --output-dir; "
                         "still runs bootstrap and report generation.")
    ap.add_argument("--no-aggregate", dest="aggregate", action="store_false")
    ap.set_defaults(aggregate=True)
    args = ap.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.input, aggregate=args.aggregate)
    plot_buzz_distribution(df, f"{args.output_dir}/buzz_distribution.png", args.bin_width)

    records, players, tossup_slugs, n_bins = build_event_records(df, bin_width=args.bin_width)

    lam_d = args.lam_d
    if lam_d is None:
        print("\n=== Selecting lam_d via 5-fold CV ===")
        lam_d = cross_validate_lambdas(records, players, tossup_slugs, n_bins,
                                       lam_s=args.lam_s, lam_a=args.lam_a, k=args.cv_folds)
        print(f"  Selected lam_d={lam_d}")

    if args.skip_fit:
        print("\n=== Reusing prior fit from --output-dir ===")
        prev = pd.read_csv(f"{args.output_dir}/overall_rankings.csv")
        # Re-derive s in the original `players` order (note: players is sorted slug list
        # used everywhere; overall_rankings is sorted by skill so we re-index).
        s_map = dict(zip(prev["player_slug"], prev["skill_estimate"]))
        s = np.array([s_map.get(p, 0.0) for p in players])
        prev_diff = pd.read_csv(f"{args.output_dir}/tossup_difficulty.csv")
        d_map = dict(zip(prev_diff["tossup_slug"], prev_diff["difficulty"]))
        d = np.array([d_map.get(sl, 0.0) for sl in tossup_slugs])
        # baseline hazard isn't persisted; skip plotting + use placeholder
        diff_table = prev_diff.sort_values("difficulty", ascending=False).reset_index(drop=True)
        overall = prev.set_index(prev.columns[0]) if "Unnamed: 0" in prev.columns else prev
        overall.index = overall.index.astype(int)
    else:
        print("\n=== Fitting survival model ===")
        s, d, a, _ = fit_survival_model(records, players, tossup_slugs, n_bins,
                                        lam_s=args.lam_s, lam_d=lam_d, lam_a=args.lam_a)
        plot_baseline_hazard(a, args.bin_width, f"{args.output_dir}/baseline_hazard.png")

        stats = player_basic_stats(df, players)
        overall = stats.copy()
        overall["skill_estimate"] = s
        overall = overall.sort_values("skill_estimate", ascending=False).reset_index(drop=True)
        overall.index += 1
        overall.to_csv(f"{args.output_dir}/overall_rankings.csv")
        print(f"Saved {args.output_dir}/overall_rankings.csv")

        diff_table = build_difficulty_table(records, tossup_slugs, d)
        diff_table.to_csv(f"{args.output_dir}/tossup_difficulty.csv", index=False)
        print(f"Saved {args.output_dir}/tossup_difficulty.csv")

    speed_profile = player_speed_profile(df, records, tossup_slugs, d, players)
    speed_profile.to_csv(f"{args.output_dir}/player_speed_profile.csv", index=False)
    print(f"Saved {args.output_dir}/player_speed_profile.csv")

    boot = None
    if args.bootstrap > 0:
        n_workers = args.workers if args.workers > 0 else None
        print(f"\n=== Bootstrap ({args.bootstrap} iters) ===", flush=True)
        boot_t0 = time.perf_counter()
        boot = bootstrap_survival(records, players, tossup_slugs, n_bins,
                                  n_bootstrap=args.bootstrap,
                                  lam_s=args.lam_s, lam_d=lam_d, lam_a=args.lam_a,
                                  n_workers=n_workers)
        print(f"  Bootstrap wall time: {time.perf_counter()-boot_t0:.0f}s", flush=True)
        boot.to_csv(f"{args.output_dir}/bootstrap_overall.csv")
        print(f"Saved {args.output_dir}/bootstrap_overall.csv")

    generate_report(overall, boot, diff_table, speed_profile, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
