# Survival Model — Buzz-Position-Aware Quizbowl Ranking

A discrete-time competing-risks hazard model on quizbowl buzzpoint data. Generalises the [Plackett–Luce model](../plackett-luce/README.md) by using *when* a buzz happened, not just *who* buzzed.

## What this models

For each `(tossup, game)`, multiple players are racing to buzz. The first correct buzz wins. The race is divided into discrete bins (default: 5 words wide, ~37 bins per question). At every bin in which the race is still alive, exactly one of these outcomes occurs:

1. **Win** — a still-in-race player buzzes correctly. Race ends.
2. **Exit** — a still-in-race player buzzes incorrectly (`buzz_value == -5` neg, or `buzz_value == 0` post-power wrong). They leave the race; others continue.
3. **No event** — the bin passes silently.
4. **Dead** — the question reaches its final bin without a winner.

### Likelihood

At every bin `b` reachable on `(tossup t, game g)`, define the multinomial-logit denominator:

```
Z(b, t)  =  1
          + sum_{j in risk(b)} exp(s_j + a_b)                # any in-race player buzzes
          + [b == B_t - 1] * exp(-d_t + a_b)                 # tossup goes dead at the final bin
```

where:
- `s_j` — player `j`'s log-hazard offset (the headline "skill" parameter)
- `a_b` — shared baseline log-hazard at bin `b`
- `d_t` — tossup `t`'s log-difficulty (larger = more dead-prone — sign matches PL's `d_t`)
- `B_t` — number of bins for tossup `t` (its length)
- `risk(b)` — players still racing at bin `b`

The likelihood contribution at bin `b`:

| Event at `b`                        | Numerator              |
|-------------------------------------|------------------------|
| no event                            | `1`                    |
| player `j` wins or exits            | `exp(s_j + a_b)`       |
| dead (only at final bin if no win)  | `exp(-d_t + a_b)`      |

Likelihood `= numerator / Z`. Negative log-likelihood is summed across all bin-observations and minimised with L-BFGS-B. L2 regularisation on `s`, `d`, `a` (default `lam_s=0.1`, `lam_d=0.5`, `lam_a=0.01`).

This reduces to the Plackett–Luce-with-difficulty model in the special case of a single bin (`B = 1`).

## Why this design (decisions made; see Open Questions for ones we punted)

- **Discrete-time grouped logistic, not continuous-time Cox.** Buzzpoints are integer word indices, the dataset is moderate-sized, and the discrete formulation lets us write the entire likelihood as a closed-form L-BFGS-B problem with no `lifelines`/`pymc` dependencies.
- **Shared baseline hazard.** Per-tossup hazard *shape* would be much more flexible but needs splines or low-rank decomposition to avoid overfitting (per HANDOFF item #2). v1 uses a single shared curve `a_b`. The shape is plotted at `results-*/baseline_hazard.png`.
- **Absolute (not normalised) position.** Tossup lengths are reasonably homogeneous (per-tossup max buzz: 135 ± 15 words). Powermarks and giveaways are at fixed word counts in QB, so absolute position respects the structure of the prompt. (HANDOFF item #3.)
- **Negs and `buzz_value == 0` rows treated identically.** Both exit the player from the race at their buzz position. v1 does *not* model neg-rate-vs-skill (HANDOFF: nice-to-have, not v1).
- **Tossup length proxy = max observed buzz across all games of that tossup.** For dead games with no buzzes at all, we fall back to this per-tossup maximum.
- **Identifiability.** `s` is mean-centred post-fit; `a` absorbs the global level; `d` is unconstrained but L2-regularised toward 0.
- **No reader-pace covariate.** HANDOFF item #4 — assessing pace–strength correlation is out of scope for v1; this is documented as an open question.

## Files

```
survival-model/
├── HANDOFF.md            – original design brief (read first)
├── README.md             – this file
├── model.py              – fit script (this is the only entry point)
└── results-*/            – per-run artefacts
    ├── overall_rankings.csv
    ├── bootstrap_overall.csv          – ci_low, ci_high, p_top5/10/20
    ├── tossup_difficulty.csv
    ├── player_speed_profile.csv       – residual buzz position by player
    ├── baseline_hazard.png            – fitted a_b curve
    ├── buzz_distribution.png          – raw empirical buzz histogram
    └── report.md                      – human-readable summary
```

## Running

```
cd survival-model
python model.py --input ../buzzpoints_data.csv --output-dir results
```

Useful flags:
- `--bin-width 5` – width of each discrete-time bin in words. Default 5 (gives ~37 bins).
- `--lam-d 0.5` – L2 on tossup difficulty. If omitted, picks via 5-fold CV (slow!).
- `--bootstrap 50` – number of cluster-bootstrap iterations (game-level).
- `--workers 0` – parallel bootstrap workers. `0` (default) = `cpu_count - 2`. Each worker peaks at ~2–4 GB on full data; lower this if OOMing. Capped automatically at `n_bootstrap`.
- `--skip-fit` – reuse the prior `overall_rankings.csv` and `tossup_difficulty.csv` from `--output-dir` instead of refitting (lets you kick off only the bootstrap stage if a prior run produced point estimates but the bootstrap died).
- `--lam-s 0.1`, `--lam-a 0.01` – player and baseline-hazard L2.
- `--no-aggregate` – disable category aggregation.

Bootstrap workers print one progress line per completed iteration, including elapsed wall time and ETA, so you can monitor the run.

Performance on a 14-core box, full ACF Nationals dataset (~63k rows, 400 tossups, 186 players, ~37 bins):
- Single fit: ~3 minutes.
- Bootstrap (50 iters, 8 workers): ~25 minutes (each worker holds ~2–4 GB, so total RAM ~20–30 GB).
- 5-fold CV across 4 lam_d values: ~70 minutes.

If you need faster fitting, the bottleneck is the per-bin Python loop in `neg_log_lik_and_grad`. Vectorising across bins (build a `(B, max_risk_set, ...)` tensor) would cut wall time substantially; not done in v1.

## Outputs (what each thing means)

- **`overall_rankings.csv`** – player skill `s` (log-hazard offset, mean-centred). Higher = more likely to buzz correctly first. Comparable in interpretation but *not* in scale to PL skills.
- **`bootstrap_overall.csv`** – game-level cluster-bootstrap 95% CI on each player's skill, plus `P(player ranks top-K)` for K∈{5,10,20}.
- **`tossup_difficulty.csv`** – `d_t` for each tossup, plus times-heard and dead-rate. Sorted hardest-first.
- **`player_speed_profile.csv`** – the headline output of this method *vs* the PL baseline. Each player's mean buzzpoint on correct buzzes, and their *mean residual* (player buzz minus per-tossup average buzz). Negative residual = consistently buzzes earlier than the field on the same questions; positive = consistently later. This is what separates "fast generalist" from "deep specialist" players with similar skill.
- **`baseline_hazard.png`** – the fitted `a_b` curve. Should be roughly increasing toward the giveaway with possibly a small bump near the powermark.

## Validation

To compare against the PL baseline:

1. Run PL: `cd ../plackett-luce && python model.py --input ../buzzpoints_data.csv --output-dir ../survival-model/results-pl-compare`
2. Join `overall_rankings.csv` from both folders on `player_slug`.
3. Compute Spearman/Pearson on the two `skill_estimate` columns. Expectation: r ≳ 0.95 for prolific players, lower for low-buzz players (more uncertainty).
4. Held-out per-bin log-likelihood on a game-level split (`cross_validate_lambdas` in `model.py`): should beat the equivalent grouped-logistic reduction of PL when position carries information.

A rigorous head-to-head CV with shared evaluation events is *not* yet implemented — see Open Questions #5.

## Open questions punted to v2

These are decisions that would introduce significant technical debt if guessed wrong:

1. **Headline use case.** The HANDOFF lists four motivations (ranking, profiling, question-quality analysis, counterfactuals). The current implementation produces outputs for #1 and #2 (ranking + speed profile). The model architecture is pulled in different directions by these — e.g. per-tossup baseline shape is needed for question-quality analysis (#3) but adds a lot of regularisation work; counterfactuals (#4) want full posteriors not point estimates. The current v1 is biased toward #1 and #2.
2. **Shared vs per-tossup baseline hazard.** Per-tossup curves (item #2 of HANDOFF "key design questions") could distinguish pyramidal questions from cliffs. v1 uses one shared `a_b`. Adding per-tossup needs splines + smoothing penalties, on the order of doubling the model's parameter count and changing the optimisation method.
3. **Reader pace as a confounder.** HANDOFF item #4. Not measured here. The first thing to do is plot mean buzzpoint per game vs the team strength of teams in that game and see whether there's correlation. If so, fold a reader-pace covariate into the hazard.
4. **Per-category survival models.** PL has per-category fits; we don't. Adding them is straightforward (subset `df` by category, refit). But it multiplies wall time by ~7 and the per-category buzzpoint distributions look different (especially Trash) so a per-category bin-width may be needed.
5. **Held-out comparison protocol with PL.** PL evaluates per-event held-out log-likelihood; the survival model evaluates per-bin. To compare them fairly, need to evaluate both on the same metric — e.g. predicted P(winner | eligible set) at the actual win-bin for the survival model, vs the PL probability.
6. **Optimiser performance.** v1 has a Python-side per-bin loop. For a usable bootstrap > 200 iterations or full CV, the inner loop should be vectorised (or moved to JAX). This is mechanical, not architectural — left as an optimisation opportunity.

## Reproducing this exact run

```bash
git checkout main
cd survival-model
python model.py \
    --input ../buzzpoints_data.csv \
    --output-dir results-1 \
    --bin-width 5 \
    --lam-s 0.1 \
    --lam-d 0.5 \
    --lam-a 0.01 \
    --bootstrap 50
```

Expected wall time: ~30 minutes on a 14-core machine. RNG seed for the bootstrap is fixed at 42 inside `model.py`.
