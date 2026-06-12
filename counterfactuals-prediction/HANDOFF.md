# Handoff: Counterfactual & Live Prediction from Buzzpoint Data

## Context

This repo (`buzzpoints-playground`) explores statistical methods on quizbowl
buzzpoint data. The Plackett-Luce and survival models in this repo each produce a
fitted parameter vector that *implicitly* defines a generative model:

- `../plackett-luce/` — given `(player_skills s, tossup difficulties d)`, the
  probability that a particular player wins a tossup with given eligible set is
  closed-form: `exp(s_winner) / (exp(d_t) + sum_k exp(s_k))`.
- `../survival-model/` — given `(s, d, a_b)`, the probability of every per-bin
  outcome is closed-form (see `model.py:296`).

That means we can use these models for *prediction*, not just *ranking*. The
existing folders fit the parameters but produce only static rankings; **this folder
extracts the predictive content of those parameters**.

The current folder (`counterfactuals-prediction/`) is for downstream prediction
tasks. It does not refit either model — it loads their saved parameters and
uses them.

## What this is for

Headline use cases:

1. **Player-swap counterfactuals.** "If alice had been on this team in this game,
   what's `P(alice wins each tossup)`?" Equivalently: "How many tossups would the
   roster `[alice, bob, carol]` have won against `[dave, eve, frank]`?"
2. **Live prediction.** Given a tossup is at word `p` with no buzz so far and risk
   set `R`, what's `P(carol buzzes next | her skill, the bin, the tossup
   difficulty)`?
3. **What-if rooms.** Simulate alternative bracket structures: "What's the expected
   tossup-conversion margin between team X and team Y on this packet?"
4. **Game-replay validation.** For each historical (tossup, game), compute the
   model's predicted probability the actual winner wins. Held-out
   per-event log-likelihood is the principled summary; per-event predictions are
   the visceral one.

## Inputs

We load fitted parameters from the survival model run (it strictly subsumes the PL
model). Specifically:

1. `../survival-model/results-1/overall_rankings.csv` → `s_i` per player
2. `../survival-model/results-1/tossup_difficulty.csv` → `d_t` per tossup
3. Baseline hazard `a_b` is *not* persisted in v1 of the survival model. Two options:
   (a) re-run the survival model with a flag that saves `a_b.csv`; (b) refit `a_b`
   on the joint records here. v1 uses **option (b)** — `a_b` is fast to re-derive
   conditional on `(s, d)` already being known (it's a 1-D fit).
4. `../buzzpoints_data.csv` — raw buzz events. Used as the (tossup, game)
   ground-truth corpus for evaluation.
5. (Optional) `../survival-model/results-1/bootstrap_overall.csv` — to propagate
   skill uncertainty through predictions.

## Predictive primitives

Given parameters, every prediction reduces to one of three closed-form computations.
See `survival-model/model.py:296` for the full likelihood; these are extracted
identities.

**Primitive 1 — `p_win_outcome(roster_i, eligible_set, tossup_t)`**

Probability player `i` wins this tossup against this field, marginalising over
positions. We approximate by integrating the per-bin hazards over the question's
length:

```
P(player i wins | risk set R, tossup t)
= sum_{b=0..B-1} h_i(b)/Z(b) * P(no event before bin b)
```

where `h_i(b) = exp(s_i + a_b)` and `Z(b)` is the per-bin denominator (HANDOFF
of survival-model has the formula).

**Primitive 2 — `p_event_at_bin(state)`**

State `= (current bin b, risk set R, tossup t, time-survived prefix)`. Returns the
multinomial probabilities over `{player_j wins, player_j negs/exits, no event,
dead (only at b = B_t-1)}`. Closed-form from the survival model.

**Primitive 3 — `simulate_game(roster_a, roster_b, packet)`**

Forward-simulate `n_sims` plays of an alternative game using primitive 2. Reports
expected conversion margin and its std error.

## Outputs

```
counterfactuals-prediction/
├── HANDOFF.md
├── README.md
├── model.py
└── results-*/
    ├── per_event_predictions.csv
    │     for every (tossup, game): predicted P(actual winner wins),
    │     log-likelihood, and brier score
    ├── live_prediction_demo.csv
    │     a few example "current state -> next event probabilities" rows
    │     to sanity-check primitive 2 against intuition
    ├── roster_swap_examples.csv
    │     for a small set of hand-picked swaps, the expected wins shift
    ├── calibration_plot.png
    │     reliability diagram (bucketed predicted-prob vs observed-frequency)
    └── report.md
```

`model.py` also exposes a CLI:

```
python model.py predict --tossup TOSSUP_SLUG --players p1,p2,p3
python model.py replay --game GAME_ID
python model.py swap --game GAME_ID --remove p1 --add p2
```

## Suggested first steps

1. **Re-derive `a_b` cheaply.** With `s, d` fixed at the survival-model fit, fit
   only `a_b` on the same data: a 1-D problem at each bin (Newton's method
   converges in 2–3 iterations). Compare your `a_b` to the survival-model's
   `baseline_hazard.png` as a sanity check.
2. **Per-event predictions on the training set first.** Don't worry about
   held-out yet. Get `P(actual winner wins)` for every (tossup, game), check
   it averages above 50% (it should — the actual winner *is* skewed strong).
3. **Calibration.** Bin the predicted probabilities into deciles, plot the
   observed frequency of "actual winner wins" in each decile. A 45° line is good
   calibration; deviations need investigation.
4. **One concrete swap.** Pick the 2026 ACF Nationals final, swap the lowest
   buzz-rate player on the winning team with the top buzzer from a mid-table team,
   simulate. The expected score swing is the headline number.

## Decisions baked into v1

- **Use the survival model's parameters, not the PL model's.** Survival is a
  superset. PL is the position-marginalised special case.
- **Re-derive `a_b` rather than persisting it from survival-model run.** Cleaner
  separation of concerns; survival-model output schema doesn't need to change.
- **Closed-form predictions, not Monte Carlo.** Primitives 1 and 2 are analytic.
  Only `simulate_game` (primitive 3) needs sampling.
- **Bootstrap-as-posterior for uncertainty.** No proper Bayesian posterior; we use
  the survival-model bootstrap CSV as a coarse proxy. Predictions can be
  marginalised over the 200 bootstrap samples (slow but trivial).
- **CLI is read-only / batch.** No interactive web UI in v1. The CLI prints
  predictions; no live-streaming of in-game state.
- **No retraining gate.** v1 trusts the survival-model fit. If the user wants
  proper held-out validation, they should split games before fitting (the survival
  model has a `cross_validate_lambdas` helper that exposes the right split logic).

## Open questions punted to v2

1. **Headline use case.** Are roster swaps the most interesting thing
   (counterfactuals), or is it live in-game prediction (commentary), or
   model validation (reliability diagrams)? v1 produces all three, but the design
   should bias toward the chosen primary use. Decision affects how the CLI is
   structured.
2. **Posterior treatment of uncertainty.** Bootstrap-as-posterior is a hack. A
   proper Laplace approximation around the MAP, or a HMC sample over the joint
   `(s, d, a)` posterior, would give honest predictive distributions. Punted: it
   would mean a substantial rewrite of the survival model fitting code.
3. **Reader-pace covariate.** Predictions ignore reader pace (same as the
   underlying model). For roster swaps this is irrelevant (pace is per-room,
   doesn't change with roster), but for live prediction the model assumes the
   pace of an "average" reader.
4. **Counterfactual identifiability under heavy interaction.** "Alice in this
   room" assumes alice's `s_alice` transfers cleanly to the new field. If alice's
   skill is partly *relative to* her usual opponents (e.g. she only buzzes when
   her teammate isn't faster), our prediction is biased. v1 doesn't model
   teammate-aware buzzing.
5. **How to score the live-prediction demo against ground truth.** Per-bin
   log-likelihood is the right metric; how to *display* it for end users is open.
6. **CLI vs notebook.** v1 uses a CLI. If editorial users would prefer a Jupyter
   notebook with widgets, that's a different output channel.
7. **Held-out evaluation protocol.** If predictions are computed on the same data
   the model was fit on, the quality numbers are optimistic. v1 documents this
   limitation; a leave-one-game-out evaluation would fix it but cost ~hundreds of
   refits.

## Pointers

- `../survival-model/model.py:296-379` — `neg_log_lik_and_grad` defines the
  probability model these primitives are derived from.
- `../survival-model/model.py:411-471` — bootstrap worker shows how to slice records
  by game; useful when implementing the per-event prediction batch.
- `../plackett-luce/HANDOFF.md` and `../survival-model/HANDOFF.md` — context for
  what each model captures.
- `../buzzpoints_data.csv` — raw events for ground truth.
- Match the existing per-method folder pattern: code at the top of this folder,
  results in `counterfactuals-prediction/results-*/`, paths in scripts relative
  to this folder.
