"""
Counterfactual & live prediction from buzzpoint data.

Loads survival-model parameters (s, d) from disk, re-derives the baseline hazard
a_b conditional on (s, d) on the same data, then exposes three predictive primitives:

    1. p_win_outcome(player_i, eligible, tossup)   marginalised P(player wins)
    2. p_event_at_bin(state)                       per-bin multinomial probs
    3. simulate_game(roster_a, roster_b, packet)   forward simulation

It also runs end-to-end:

    - per_event_predictions.csv: P(actual winner wins) per (tossup, game)
    - calibration_plot.png:      reliability diagram
    - roster_swap_examples.csv:  4 hand-picked swaps with predicted Δwins
    - live_prediction_demo.csv:  3 worked examples of primitive 2

CLI usage:
    python model.py                                   # run full batch
    python model.py predict --tossup SLUG --players a,b,c
    python model.py replay --game GAME_ID
    python model.py swap --game GAME_ID --remove a --add b

See HANDOFF.md.
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Data loading + record building (pared-down copy of survival-model logic)
# -----------------------------------------------------------------------------

CATEGORY_MAP = {
    "Fine Arts": "Fine Arts", "Science": "Science", "History": "History",
    "Literature": "Literature", "Religion": "Beliefs", "Mythology": "Beliefs",
    "Philosophy": "Thought", "Social Science": "Thought", "Geography": "Other",
    "Current Events": "Other", "Other Academic": "Other", "Trash": "Other",
}


def aggregate_category(raw: str) -> str:
    raw = raw.strip()
    for prefix, top in CATEGORY_MAP.items():
        if raw.startswith(prefix):
            return top
    return "Other"


def load_data(path: str, aggregate: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["won"] = df["won"].map({"True": True, "False": False})
    for c in ("buzzpoint", "buzz_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if aggregate:
        df["category"] = df["category"].apply(aggregate_category)
    return df


def build_event_records(df: pd.DataFrame, bin_width: int = 5):
    """Mirror survival-model.build_event_records — returns (records, players, tossup_slugs, n_bins).

    Each record is one (tossup, game) with a discrete-time event sequence.
    """
    players = sorted(df["player_slug"].unique())
    pidx = {p: i for i, p in enumerate(players)}

    bp_per_tossup = df.groupby("tossup_slug")["buzzpoint"].max()
    tossup_length_bin: dict[str, int] = {}
    for sl, mx in bp_per_tossup.items():
        if pd.isna(mx):
            mx = 0
        tossup_length_bin[sl] = int(mx) // bin_width + 1
    tossup_slugs = sorted(df["tossup_slug"].unique())
    max_length_bin = max(tossup_length_bin.values()) if tossup_length_bin else 1

    records = []
    for (sl, gid), grp in df.groupby(["tossup_slug", "game_id"]):
        eligible_slugs = list(grp["player_slug"].unique())
        eligible = [pidx[p] for p in eligible_slugs]
        if len(eligible) < 2:
            continue

        events = []
        winner_idx = None
        for _, r in grp.iterrows():
            bv = r["buzz_value"]
            bp = r["buzzpoint"]
            if pd.isna(bv) or pd.isna(bp):
                continue
            b = int(bp) // bin_width
            kind = "win" if bv == 10 else "exit"
            if bv == 10:
                winner_idx = pidx[r["player_slug"]]
            events.append((b, pidx[r["player_slug"]], kind))
        events.sort(key=lambda e: (e[0], 0 if e[2] == "exit" else 1))
        if winner_idx is not None:
            win_bin = next(b for b, p, k in events if p == winner_idx and k == "win")
            events = [e for e in events if e[0] <= win_bin]

        if events:
            last_bin = max(e[0] for e in events) + 1
            length_bin = max(last_bin, tossup_length_bin[sl])
        else:
            length_bin = tossup_length_bin[sl]
        length_bin = min(length_bin, max_length_bin)

        records.append({
            "tossup_slug": sl, "game_id": gid, "category": grp["category"].iloc[0],
            "eligible": eligible, "events": events,
            "winner_idx": winner_idx, "dead": winner_idx is None,
            "length_bin": length_bin,
        })
    return records, players, tossup_slugs, max_length_bin


# -----------------------------------------------------------------------------
# Pre-flatten records into per-bin risk sets + event labels
# -----------------------------------------------------------------------------

def flatten_records(records, tossup_to_idx):
    """Per-record list of (bin_idx, risk_set, event_pidx, event_type, length, tossup_idx).

    event_type: 0 = none, 1 = win, 2 = exit, 3 = dead.
    """
    flat = []
    for rec in records:
        ti = tossup_to_idx[rec["tossup_slug"]]
        L = rec["length_bin"]
        at_risk = set(rec["eligible"])
        ev_by_bin: dict[int, list] = defaultdict(list)
        for b, p, k in rec["events"]:
            ev_by_bin[b].append((p, k))

        risk_sets = []
        event_pidx = np.full(L, -1, dtype=np.int64)
        event_type = np.zeros(L, dtype=np.int64)
        for b in range(L):
            risk_sets.append(np.array(sorted(at_risk), dtype=np.int64))
            evs = ev_by_bin.get(b, [])
            chosen = next(((p, k) for p, k in evs if k == "win"), None)
            if chosen is None and evs:
                chosen = evs[0]
            if chosen is not None:
                p, k = chosen
                event_pidx[b] = p
                event_type[b] = 1 if k == "win" else 2
                if k == "win":
                    at_risk = set()
                    risk_sets = risk_sets[:b + 1]
                    event_pidx = event_pidx[:b + 1]
                    event_type = event_type[:b + 1]
                    L = b + 1
                    break
                else:
                    at_risk.discard(p)
                    for p2, k2 in evs:
                        if (p2, k2) != chosen and k2 == "exit":
                            at_risk.discard(p2)
        if rec["dead"] and L > 0 and event_type[L - 1] == 0:
            event_type[L - 1] = 3
        flat.append({
            "tossup_idx": ti,
            "risk": risk_sets,
            "event_pidx": event_pidx,
            "event_type": event_type,
            "length_bin": L,
            "tossup_slug": rec["tossup_slug"],
            "game_id": rec["game_id"],
            "winner_idx": rec["winner_idx"],
            "eligible": rec["eligible"],
        })
    return flat


# -----------------------------------------------------------------------------
# Re-derive baseline hazard a_b given fixed s, d
# -----------------------------------------------------------------------------

def fit_baseline_hazard(flat, s: np.ndarray, d: np.ndarray, n_bins: int,
                        lam_a: float = 0.01, verbose: bool = True) -> np.ndarray:
    """1-D fit of a (length n_bins) holding s and d fixed. L-BFGS-B with closed-form grad."""
    def neg_ll_and_grad(a):
        nll = 0.0
        g = np.zeros_like(a)
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
                h_p = np.exp(s[risk] + ab)
                include_dead = (b == L - 1)
                h_d = np.exp(-d_t + ab) if include_dead else 0.0
                Z = 1.0 + h_p.sum() + h_d
                et = rec["event_type"][b]
                # log denominator
                nll += np.log(Z)
                g[b] += (h_p.sum() + h_d) / Z
                if et == 1 or et == 2:
                    nll -= ab + s[rec["event_pidx"][b]]
                    g[b] -= 1.0
                elif et == 3:
                    nll -= ab + (-d_t)
                    g[b] -= 1.0
                # else et==0: no numerator subtraction
        nll += lam_a * np.dot(a, a)
        g += 2.0 * lam_a * a
        return nll, g

    a0 = np.zeros(n_bins)
    if verbose:
        print(f"  Refitting baseline hazard a_b (n_bins={n_bins})...")
    t0 = time.perf_counter()
    res = scipy.optimize.minimize(
        neg_ll_and_grad, a0, jac=True, method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-7},
    )
    if verbose:
        print(f"  Baseline fit: nll={res.fun:.2f} in {time.perf_counter()-t0:.1f}s, "
              f"converged={res.success}")
    return res.x


# -----------------------------------------------------------------------------
# Predictive primitives
# -----------------------------------------------------------------------------

class Predictor:
    def __init__(self, s: np.ndarray, d: np.ndarray, a: np.ndarray,
                 player_to_idx: dict[str, int],
                 tossup_to_idx: dict[str, int],
                 tossup_lengths: dict[str, int]):
        self.s = s
        self.d = d
        self.a = a
        self.player_to_idx = player_to_idx
        self.tossup_to_idx = tossup_to_idx
        self.tossup_lengths = tossup_lengths
        self.n_bins = len(a)

    def _tossup_idx(self, tossup_slug: str) -> int:
        return self.tossup_to_idx.get(tossup_slug, -1)

    def _player_idxs(self, players: list[str]) -> list[int]:
        return [self.player_to_idx[p] for p in players]

    # Primitive 1: marginalised win probability across the full tossup
    def p_win_outcome(self, player: str, eligible: list[str],
                      tossup_slug: str) -> dict:
        ti = self._tossup_idx(tossup_slug)
        d_t = self.d[ti] if ti >= 0 else 0.0
        L = self.tossup_lengths.get(tossup_slug, self.n_bins)
        L = min(L, self.n_bins)
        elig_idxs = self._player_idxs(eligible)
        if player not in self.player_to_idx:
            return {"player": player, "p_win": 0.0, "p_dead": 0.0,
                    "note": "unknown player"}
        player_idx = self.player_to_idx[player]
        if player_idx not in elig_idxs:
            return {"player": player, "p_win": 0.0, "p_dead": 0.0,
                    "note": "player not in eligible set"}
        # Survive bin-by-bin.
        p_alive = 1.0
        p_win = 0.0
        p_dead = 0.0
        # We assume nobody negs; this is an "outcome marginal" prediction conditional
        # on no exits (closer to a clean what-if than a forward simulation).
        elig_arr = np.array(elig_idxs)
        for b in range(L):
            ab = self.a[b]
            h_players = np.exp(self.s[elig_arr] + ab)
            include_dead = (b == L - 1)
            h_dead = np.exp(-d_t + ab) if include_dead else 0.0
            Z = 1.0 + h_players.sum() + h_dead
            this_player_h = np.exp(self.s[player_idx] + ab)
            p_win_this_bin = (this_player_h / Z) * p_alive
            p_dead_this_bin = (h_dead / Z) * p_alive
            p_win += p_win_this_bin
            p_dead += p_dead_this_bin
            # advance
            p_alive *= (1.0 - (h_players.sum() + h_dead) / Z)
        return {"player": player, "p_win": float(p_win),
                "p_dead": float(p_dead), "p_someone_else": float(1 - p_win - p_dead - p_alive),
                "p_alive_at_end": float(p_alive),
                "tossup_difficulty_d": float(d_t)}

    # Primitive 2: per-bin event probabilities for given state
    def p_event_at_bin(self, eligible: list[str], tossup_slug: str,
                       current_bin: int) -> dict:
        ti = self._tossup_idx(tossup_slug)
        d_t = self.d[ti] if ti >= 0 else 0.0
        L = self.tossup_lengths.get(tossup_slug, self.n_bins)
        L = min(L, self.n_bins)
        elig_idxs = np.array(self._player_idxs(eligible))
        if current_bin >= L:
            return {"note": "past end of tossup"}
        ab = self.a[current_bin]
        h_players = np.exp(self.s[elig_idxs] + ab)
        include_dead = (current_bin == L - 1)
        h_dead = np.exp(-d_t + ab) if include_dead else 0.0
        Z = 1.0 + h_players.sum() + h_dead
        per_player = {p: float(h_players[i] / Z) for i, p in enumerate(eligible)}
        return {
            "p_no_event": 1.0 / Z,
            "p_some_buzz_total": float(h_players.sum() / Z),
            "p_dead_now": float(h_dead / Z),
            "per_player_p_buzz": per_player,
        }

    # Primitive 3: forward simulation of a packet of tossups for two rosters
    def simulate_game(self, roster_a: list[str], roster_b: list[str],
                      tossup_slugs: list[str], n_sims: int = 1000,
                      seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        wins_a = np.zeros(n_sims, dtype=int)
        wins_b = np.zeros(n_sims, dtype=int)
        elig_all = roster_a + roster_b
        elig_idx_all = np.array(self._player_idxs(elig_all))
        a_set = set(roster_a)
        for _ in range(n_sims):
            for ts in tossup_slugs:
                ti = self._tossup_idx(ts)
                d_t = self.d[ti] if ti >= 0 else 0.0
                L = self.tossup_lengths.get(ts, self.n_bins)
                L = min(L, self.n_bins)
                in_race = elig_idx_all.copy()
                won = False
                for b in range(L):
                    ab = self.a[b]
                    h_p = np.exp(self.s[in_race] + ab)
                    include_dead = (b == L - 1)
                    h_d = np.exp(-d_t + ab) if include_dead else 0.0
                    Z = 1.0 + h_p.sum() + h_d
                    # Sample outcome category
                    p_no = 1.0 / Z
                    p_d = h_d / Z
                    cum = np.cumsum(np.concatenate([[p_no], h_p / Z, [p_d]]))
                    u = rng.random()
                    pick = np.searchsorted(cum, u)
                    if pick == 0:
                        continue
                    if pick == len(cum) - 1:
                        # dead
                        break
                    # buzz
                    buzzer = in_race[pick - 1]
                    # Treat all sampled buzzes as wins (simplification — neg sampling
                    # would need a per-player accuracy model)
                    p_slug = list(self.player_to_idx.keys())[
                        list(self.player_to_idx.values()).index(buzzer)
                    ]
                    if p_slug in a_set:
                        wins_a[_] += 1
                    else:
                        wins_b[_] += 1
                    won = True
                    break
        return {
            "mean_wins_a": float(wins_a.mean()),
            "mean_wins_b": float(wins_b.mean()),
            "std_margin": float((wins_a - wins_b).std(ddof=1)),
            "p_a_outscores_b": float((wins_a > wins_b).mean()),
        }


# -----------------------------------------------------------------------------
# Per-event predictions (batch evaluation)
# -----------------------------------------------------------------------------

def per_event_predictions(predictor: Predictor, flat: list[dict],
                          players: list[str]) -> pd.DataFrame:
    rows = []
    for rec in flat:
        elig_slugs = [players[i] for i in rec["eligible"]]
        ts = rec["tossup_slug"]
        for_each_player = {}
        # Compute marginalised P(p wins) for each eligible player, fixed field
        ti = rec["tossup_idx"]
        d_t = predictor.d[ti] if ti >= 0 else 0.0
        L = rec["length_bin"]
        elig_arr = np.array(rec["eligible"])
        # Bin-by-bin survival
        h_per_player_total = np.zeros(len(elig_arr))
        p_alive = 1.0
        p_dead = 0.0
        for b in range(L):
            ab = predictor.a[b]
            h_p = np.exp(predictor.s[elig_arr] + ab)
            include_dead = (b == L - 1)
            h_d = np.exp(-d_t + ab) if include_dead else 0.0
            Z = 1.0 + h_p.sum() + h_d
            h_per_player_total += (h_p / Z) * p_alive
            p_dead += (h_d / Z) * p_alive
            p_alive *= 1.0 - (h_p.sum() + h_d) / Z

        winner_idx = rec["winner_idx"]
        if winner_idx is None:
            actual_p = float(p_dead)
            actual_label = "dead"
        else:
            local = list(elig_arr).index(winner_idx) if winner_idx in elig_arr else -1
            actual_p = float(h_per_player_total[local]) if local >= 0 else float("nan")
            actual_label = players[winner_idx]
        rows.append({
            "tossup_slug": ts,
            "game_id": rec["game_id"],
            "n_eligible": len(elig_slugs),
            "actual_outcome": actual_label,
            "predicted_p_actual": actual_p,
            "p_dead": float(p_dead),
        })
    return pd.DataFrame(rows)


def calibration_plot(events: pd.DataFrame, out_path: str, n_bins: int = 10):
    df = events.dropna(subset=["predicted_p_actual"]).copy()
    df["was_dead"] = df["actual_outcome"] == "dead"
    # The "event" we predict is "actual outcome occurred" — by construction this is 1.
    # Better calibration: bin by predicted P(actual winner wins) and check if the
    # average P matches the observed rate (which is 1 — not informative).
    # Instead, evaluate calibration on P(dead) vs actual dead.
    df["p_dead_bin"] = pd.cut(df["p_dead"], bins=np.linspace(0, 1, n_bins + 1),
                              labels=False, include_lowest=True)
    g = df.groupby("p_dead_bin").agg(
        avg_pred=("p_dead", "mean"),
        avg_obs=("was_dead", "mean"),
        n=("p_dead", "count"),
    ).dropna()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
    ax.scatter(g["avg_pred"], g["avg_obs"], s=g["n"], alpha=0.7, color="steelblue")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted P(dead) in bucket")
    ax.set_ylabel("Observed dead rate in bucket")
    ax.set_title("Calibration: predicted P(dead) vs observed (size = #events)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


# -----------------------------------------------------------------------------
# Roster-swap examples
# -----------------------------------------------------------------------------

def roster_swap_examples(predictor: Predictor, df: pd.DataFrame, players: list[str],
                         flat: list[dict], n_examples: int = 4) -> pd.DataFrame:
    """For a few hand-picked games, swap one player and report Δ(expected wins)."""
    # Pick games with at least 6 players and at least one strong player.
    by_game: dict[str, list[dict]] = defaultdict(list)
    for rec in flat:
        by_game[rec["game_id"]].append(rec)

    # Sort players by skill descending; we'll swap "top" / "bottom" players in
    # demos.
    skill_series = pd.Series(predictor.s, index=players).sort_values(ascending=False)
    top_players = skill_series.head(5).index.tolist()
    mid_players = skill_series.iloc[len(skill_series) // 2 - 2:len(skill_series) // 2 + 3].index.tolist()

    rows = []
    games_iter = iter(sorted(by_game.keys()))
    examples_done = 0
    while examples_done < n_examples:
        try:
            gid = next(games_iter)
        except StopIteration:
            break
        recs = by_game[gid]
        if not recs:
            continue
        # Get the union of eligible players over the game
        elig_idxs = sorted({i for r in recs for i in r["eligible"]})
        elig_slugs = [players[i] for i in elig_idxs]
        if len(elig_slugs) < 4:
            continue
        # Pick a player to remove (not a top player, ideally) and a top player to insert
        candidates_remove = [p for p in elig_slugs if p in mid_players]
        if not candidates_remove:
            candidates_remove = elig_slugs[:1]
        remove = candidates_remove[0]
        candidates_add = [p for p in top_players if p not in elig_slugs]
        if not candidates_add:
            continue
        add = candidates_add[0]

        # Compute baseline expected wins for `add` IF added
        baseline_wins = 0.0
        swapped_wins = 0.0
        n_tossups = 0
        for rec in recs:
            ts = rec["tossup_slug"]
            res_base = predictor.p_win_outcome(add, elig_slugs + [add], ts)
            new_elig = [p for p in elig_slugs if p != remove] + [add]
            res_swap = predictor.p_win_outcome(add, new_elig, ts)
            baseline_wins += res_base["p_win"]
            swapped_wins += res_swap["p_win"]
            n_tossups += 1

        rows.append({
            "game_id": gid,
            "removed_player": remove,
            "added_player": add,
            "n_tossups": n_tossups,
            "added_expected_wins_in_full_field": baseline_wins,
            "added_expected_wins_after_swap": swapped_wins,
            "delta": swapped_wins - baseline_wins,
        })
        examples_done += 1
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Live prediction demo
# -----------------------------------------------------------------------------

def live_prediction_demo(predictor: Predictor, df: pd.DataFrame, players: list[str],
                         tossup_slugs: list[str]) -> pd.DataFrame:
    """A small set of (tossup, current_bin, eligible) -> next-event probabilities."""
    skill = pd.Series(predictor.s, index=players)
    top3 = skill.sort_values(ascending=False).head(3).index.tolist()
    eligibles = [top3, top3 + skill.sort_values(ascending=False).iloc[3:6].index.tolist()]

    # Pick 3 tossups: easy, medium, hard (by d_t)
    diff_series = pd.Series(predictor.d, index=tossup_slugs)
    sorted_d = diff_series.sort_values()
    picks = [sorted_d.index[len(sorted_d) // 4],
             sorted_d.index[len(sorted_d) // 2],
             sorted_d.index[3 * len(sorted_d) // 4]]

    rows = []
    for ts in picks:
        for elig in eligibles:
            for cur_bin in (5, 15, 25):
                res = predictor.p_event_at_bin(elig, ts, cur_bin)
                if "note" in res:
                    continue
                rows.append({
                    "tossup_slug": ts,
                    "tossup_difficulty_d": float(diff_series[ts]),
                    "n_eligible": len(elig),
                    "current_bin": cur_bin,
                    "p_no_event": res["p_no_event"],
                    "p_some_buzz": res["p_some_buzz_total"],
                    "p_dead_now": res["p_dead_now"],
                    "top_buzzer": max(res["per_player_p_buzz"], key=res["per_player_p_buzz"].get),
                    "top_buzzer_p": max(res["per_player_p_buzz"].values()),
                })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CLI commands
# -----------------------------------------------------------------------------

def cmd_predict(predictor: Predictor, args):
    elig = args.players.split(",")
    res = predictor.p_win_outcome(elig[0] if not args.player else args.player, elig, args.tossup)
    print(res)


def cmd_replay(predictor: Predictor, df: pd.DataFrame, players: list[str], args):
    sub = df[df["game_id"] == args.game]
    if len(sub) == 0:
        print(f"Game {args.game} not found"); return
    print(f"\nReplaying game {args.game}:")
    by_t = sub.groupby("tossup_slug")
    for ts, grp in by_t:
        elig = grp["player_slug"].unique().tolist()
        actual_winner = grp[grp["won"] == True]["player_slug"].iloc[0] if (grp["won"] == True).any() else None
        # Predict P(each player wins)
        ti = predictor._tossup_idx(ts)
        d_t = predictor.d[ti] if ti >= 0 else 0.0
        elig_idxs = np.array(predictor._player_idxs(elig))
        if len(elig_idxs) < 2:
            continue
        # Marginalised
        L = predictor.tossup_lengths.get(ts, predictor.n_bins)
        L = min(L, predictor.n_bins)
        per_p = np.zeros(len(elig_idxs))
        p_alive = 1.0
        p_dead = 0.0
        for b in range(L):
            ab = predictor.a[b]
            h_p = np.exp(predictor.s[elig_idxs] + ab)
            include_dead = (b == L - 1)
            h_d = np.exp(-d_t + ab) if include_dead else 0.0
            Z = 1.0 + h_p.sum() + h_d
            per_p += (h_p / Z) * p_alive
            p_dead += (h_d / Z) * p_alive
            p_alive *= 1.0 - (h_p.sum() + h_d) / Z
        order = np.argsort(-per_p)
        line = ", ".join(f"{elig[i]}:{per_p[i]:.0%}" for i in order[:3])
        flag = "  <- ACTUAL" if actual_winner else ""
        print(f"  {ts} (d={d_t:+.2f}) | actual={actual_winner or 'DEAD'} | top3 predicted: {line}{flag}")


def cmd_swap(predictor: Predictor, df: pd.DataFrame, players: list[str], args):
    sub = df[df["game_id"] == args.game]
    if len(sub) == 0:
        print(f"Game {args.game} not found"); return
    elig = sorted(sub["player_slug"].unique().tolist())
    if args.remove not in elig:
        print(f"Player {args.remove} not in game"); return
    if args.add not in predictor.player_to_idx:
        print(f"Player {args.add} unknown"); return
    new_elig = [p for p in elig if p != args.remove] + [args.add]
    print(f"\nSwap: -{args.remove} +{args.add} in {args.game}\n")
    delta_total = 0.0
    for ts in sorted(sub["tossup_slug"].unique()):
        r1 = predictor.p_win_outcome(args.add, elig + [args.add], ts)
        r2 = predictor.p_win_outcome(args.add, new_elig, ts)
        delta = r2["p_win"] - r1["p_win"]
        delta_total += delta
        print(f"  {ts}: {args.add}'s P(win) {r1['p_win']:.0%} -> {r2['p_win']:.0%} "
              f"(Δ {delta:+.0%})")
    print(f"\nTotal expected-wins shift for {args.add}: {delta_total:+.2f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    # Default (no subcmd) -> run full batch
    ap.add_argument("--raw", default="../buzzpoints_data.csv")
    ap.add_argument("--surv-dir", default="../survival-model/results-1")
    ap.add_argument("--output-dir", default="results-1")
    ap.add_argument("--bin-width", type=int, default=5)

    sp = sub.add_parser("predict", help="One-shot win-probability prediction")
    sp.add_argument("--tossup", required=True)
    sp.add_argument("--players", required=True, help="comma-separated player slugs")
    sp.add_argument("--player", help="If set, that player's P(win); else first")

    sr = sub.add_parser("replay", help="Print model's per-tossup P(actual winner wins) for a game")
    sr.add_argument("--game", required=True)

    ssw = sub.add_parser("swap", help="Swap one player out, another in")
    ssw.add_argument("--game", required=True)
    ssw.add_argument("--remove", required=True)
    ssw.add_argument("--add", required=True)

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data and survival fit...")
    df = load_data(args.raw)
    surv_overall = pd.read_csv(os.path.join(args.surv_dir, "overall_rankings.csv"))
    surv_diff = pd.read_csv(os.path.join(args.surv_dir, "tossup_difficulty.csv"))

    records, players, tossup_slugs, n_bins = build_event_records(df, bin_width=args.bin_width)
    pidx = {p: i for i, p in enumerate(players)}
    tidx = {t: i for i, t in enumerate(tossup_slugs)}

    s_map = dict(zip(surv_overall["player_slug"], surv_overall["skill_estimate"]))
    s = np.array([s_map.get(p, 0.0) for p in players])
    d_map = dict(zip(surv_diff["tossup_slug"], surv_diff["difficulty"]))
    d = np.array([d_map.get(t, 0.0) for t in tossup_slugs])

    flat = flatten_records(records, tidx)
    a = fit_baseline_hazard(flat, s, d, n_bins)

    # Tossup lengths (bin counts)
    tossup_length_bin = {}
    for rec in records:
        tossup_length_bin[rec["tossup_slug"]] = max(
            tossup_length_bin.get(rec["tossup_slug"], 0), rec["length_bin"])

    predictor = Predictor(s, d, a, pidx, tidx, tossup_length_bin)

    # Persist a_b
    pd.DataFrame({"bin": range(n_bins), "a_b": a}).to_csv(
        os.path.join(args.output_dir, "baseline_hazard.csv"), index=False)

    if args.cmd == "predict":
        cmd_predict(predictor, args); return
    if args.cmd == "replay":
        cmd_replay(predictor, df, players, args); return
    if args.cmd == "swap":
        cmd_swap(predictor, df, players, args); return

    # ---- Full batch ----
    print("\nComputing per-event predictions...")
    events = per_event_predictions(predictor, flat, players)
    out = os.path.join(args.output_dir, "per_event_predictions.csv")
    events.to_csv(out, index=False)
    print(f"Saved {out}")

    # Average predicted P(actual winner wins): higher is better calibration of point pred.
    won_events = events.dropna(subset=["predicted_p_actual"])
    won_events = won_events[won_events["actual_outcome"] != "dead"]
    print(f"  Mean predicted P(actual winner wins): "
          f"{won_events['predicted_p_actual'].mean():.3f} "
          f"(over {len(won_events)} won events)")

    calibration_plot(events, os.path.join(args.output_dir, "calibration_plot.png"))

    print("\nGenerating roster-swap examples...")
    swaps = roster_swap_examples(predictor, df, players, flat)
    out = os.path.join(args.output_dir, "roster_swap_examples.csv")
    swaps.to_csv(out, index=False)
    print(f"Saved {out}")

    print("\nLive prediction demo...")
    demo = live_prediction_demo(predictor, df, players, tossup_slugs)
    out = os.path.join(args.output_dir, "live_prediction_demo.csv")
    demo.to_csv(out, index=False)
    print(f"Saved {out}")

    # Brief markdown summary
    lines = ["# Counterfactuals & Live Prediction — Report\n",
             "Loads the survival-model fit (s, d) and re-derives a_b, then exposes three "
             "predictive primitives. See HANDOFF.md.\n",
             f"- {len(players)} players, {len(tossup_slugs)} tossups, {n_bins} bins.",
             f"- Mean predicted P(actual winner wins) on won events: "
             f"**{won_events['predicted_p_actual'].mean():.3f}** "
             f"(higher is better; random would be 1/n_eligible ≈ "
             f"{1.0 / events['n_eligible'].mean():.3f}).\n",
             "## Calibration (predicted vs observed P(dead))\n",
             "See `calibration_plot.png`. The 45° line is perfect calibration; "
             "deviation indicates the model is over- or under-predicting tossup-death rate.\n",
             "## Roster swap examples\n",
             "See `roster_swap_examples.csv`. Each row: a hypothetical swap of one player "
             "out, one in, computed across all tossups in that game.\n",
             "## Live prediction demo\n",
             "See `live_prediction_demo.csv`. Each row: at a snapshot of game state "
             "(current_bin, eligible set, tossup), what the model thinks comes next.\n",
             "## CLI\n",
             "```bash",
             "python model.py predict --tossup SLUG --players p1,p2,p3 --player p1",
             "python model.py replay --game GAME_ID",
             "python model.py swap --game GAME_ID --remove p1 --add p2",
             "```"]
    with open(os.path.join(args.output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {args.output_dir}/report.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
