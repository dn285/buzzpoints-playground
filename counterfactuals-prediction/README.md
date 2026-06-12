# Counterfactual & Live Prediction

Wraps the survival-model fit as a closed-form predictor and exposes three
primitives:

1. **`p_win_outcome(player, eligible, tossup)`** — marginalised P(player wins)
2. **`p_event_at_bin(state)`** — per-bin multinomial probabilities given current state
3. **`simulate_game(roster_a, roster_b, packet)`** — forward simulation

See `HANDOFF.md` for the design rationale.

## How it works

The survival-model output persists `s` (player skill) and `d` (tossup difficulty) but
not `a_b` (baseline hazard). On startup, `model.py` loads `s` and `d` and re-derives
`a_b` by running L-BFGS-B on the 1-D conditional likelihood (≈ 4 minutes on the full
dataset). The fitted `(s, d, a)` parameters then drive all predictions in closed form.

## Run

```bash
cd counterfactuals-prediction
python model.py --output-dir results-1
```

This runs the full batch:
- Per-event predictions on every (tossup, game) in the data
- Calibration plot for predicted P(dead)
- 4 hand-picked roster-swap examples
- Live-prediction worked examples

## CLI subcommands

```bash
python model.py predict --tossup SLUG --players p1,p2,p3 --player p2
python model.py replay  --game GAME_ID
python model.py swap    --game GAME_ID --remove p1 --add p2
```

## Outputs

- `baseline_hazard.csv` — fitted `a_b` per bin (saved for reuse)
- `per_event_predictions.csv` — `predicted_p_actual`, `p_dead`, etc.
- `roster_swap_examples.csv` — Δ(expected wins) per swap
- `live_prediction_demo.csv` — primitive 2 worked examples
- `calibration_plot.png` — predicted vs observed P(dead) reliability diagram
- `report.md`

## Known limitations (v1)

- **`a_b` is re-derived, not loaded.** The survival-model run does not persist
  `a_b`; recovering it requires a separate fit. Adding `--save-baseline` to the
  survival-model would speed startup from ~4 minutes to instant.
- **Predictions assume the field never shrinks.** The actual race has exits (negs)
  that the predictor cannot anticipate — so per-event metrics under-credit the
  actual winner whenever they faced a smaller-than-eligible field. For
  *counterfactual* roster-swap predictions this is the right behaviour; for
  *historical* per-event evaluation it depresses the metric below uniform
  baseline.
- **Bootstrap-as-posterior is not yet implemented.** The bootstrap CSV is on
  disk but the prediction pipeline uses point estimates only.

## Reproducing

```bash
git checkout main
cd counterfactuals-prediction
python model.py --output-dir results-1
```

Wall time: ~5 minutes (dominated by the baseline-hazard refit).
