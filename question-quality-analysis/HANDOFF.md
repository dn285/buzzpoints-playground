# Handoff: Tossup Quality Analysis from Buzzpoint Data

## Context

This repo (`buzzpoints-playground`) explores statistical methods on quizbowl
buzzpoint data. The two existing models give one number per tossup:

- `../plackett-luce/` — outcome-only model, with a per-tossup difficulty `d_t`.
  `d_t` measures how dead-prone a tossup is.
- `../survival-model/` — discrete-time hazard model. Same `d_t` interpretation, plus
  a **shared** baseline hazard `a_b` that controls how the buzz density evolves with
  position (averaged across all tossups).

A single difficulty number captures *whether* a tossup converted but says nothing
about the *shape* of how it played: when buzzes happened, how clustered they were,
whether the giveaway was a cliff or a slope. **That shape is the editorially
interesting signal.** Two tossups with identical dead rate and mean conversion
position can have totally different shapes — one rewards depth across the whole
prompt (smooth pyramidal hazard curve), one is a sudden trigger (most buzzes at the
same word). Editors care about the second distinction; the existing models don't see
it.

The current folder (`question-quality-analysis/`) is the home for shape-aware tossup
analysis.

## What this is for

Concrete questions:

1. **Pyramidality.** Is the buzz density monotonically increasing toward the
   giveaway, the way the form intends? Or are there flat stretches / weird humps
   in the middle?
2. **Cliff vs slope.** Do buzzes cluster at one trigger word (low variance,
   high kurtosis), or are they spread across the prompt?
3. **Power-conversion structure.** What fraction of buzzes happen before the power
   mark (≤ word 75 in standard ACF format)? Is the power "earnable" or essentially
   theoretical?
4. **Discrimination power.** Does the tossup separate strong from weak players
   (high information at relevant skill levels) or is it indifferent (everyone
   buzzes the same word, or no one buzzes at all)?
5. **Editorial classification.** Map each tossup to one of: `pyramidal`,
   `front-loaded`, `cliff`, `back-loaded`, `dud`, `gimme`. Headline output of the
   folder.

The motivation is editorial review of question sets, the kind of work an editorial
team does manually after a tournament. This folder automates the first pass.

## Inputs

1. `../buzzpoints_data.csv` — raw buzz events.
2. `../survival-model/results-1/tossup_difficulty.csv` — `d_t` per tossup. We
   reuse this as our difficulty axis instead of refitting.
3. `../survival-model/results-1/baseline_hazard.png` — for visual sanity check;
   no programmatic input.
4. `../survival-model/results-1/overall_rankings.csv` — optional, used for the
   discrimination metric (correlate buzz-or-not with skill).

If the survival-model results don't exist, run that folder first.

## Quality metrics (computed per tossup)

For each tossup `t` with `times_heard >= 3`:

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| `n_heard`         | # games this tossup was used in              | Sample-size guard |
| `dead_rate`       | dead games / heard games                     | How often it failed to convert |
| `mean_buzz`       | mean correct-buzz word index                 | Where it typically converts |
| `std_buzz`        | sd of correct-buzz word indices              | Cliff (low) vs slope (high) |
| `iqr_buzz`        | 75th – 25th pctile of correct buzz position  | Robust spread |
| `power_rate`      | correct buzzes at `bp ≤ 75` / correct buzzes | Power earnability |
| `pyramidality`    | Spearman of (cumulative buzz fraction) vs (position) — clipped to [0, 1] | 1 = perfect monotonic, ~0 = degenerate |
| `slope_score`     | OLS slope of buzz-cumulative on relative position, normalised | + = front-loaded, − = back-loaded |
| `cliff_score`     | `1 − (iqr_buzz / total_length)`              | High = trigger word, low = smooth |
| `discrimination`  | `corr(buzzed_correct, player_skill_pl)` across players who heard it | Skill differentiation |
| `difficulty_d`    | from `tossup_difficulty.csv`                 | Reuse, not refit |

`pyramidality` is the headline shape metric: take the empirical CDF of buzz
positions on the tossup (over all games), evaluate at the bin midpoints, and check
whether the CDF is monotonically increasing in shape (it always is, by definition,
but the *rate* of increase tells us pyramidality). Concretely: regress the per-bin
empirical hazard on bin index and report the standardised positive slope; clip to
[0,1] for readability.

## Editorial classification

Combine the metrics into a categorical label per tossup:

- **gimme**: `mean_buzz / total_length < 0.4` AND `dead_rate < 5%`
  (everyone gets it, fast)
- **dud**: `dead_rate > 50%`
  (nobody got it; difficulty doesn't matter, the question failed)
- **cliff**: `cliff_score > 0.85` AND `dead_rate < 30%`
  (everyone buzzes at the same word — a single trigger)
- **front-loaded**: `slope_score > +0.5`
  (most buzzes happen in the first half, but it's not a single trigger)
- **back-loaded**: `slope_score < −0.5`
  (most buzzes happen in the last quarter — possibly a "needs the giveaway" question)
- **pyramidal**: anything else with `pyramidality > 0.6` AND `dead_rate < 30%`
  (well-formed)
- **noisy**: catch-all

Thresholds are empirical; they should be tuned to the dataset distribution. v1 picks
defaults from the percentile distributions of each metric and emits a histogram so
the reader can re-tune.

## Outputs

```
question-quality-analysis/
├── HANDOFF.md
├── README.md
├── model.py
└── results-*/
    ├── tossup_quality.csv              all metrics + label per tossup
    ├── classification_counts.csv        # of tossups per label
    ├── per_category_quality.csv         summary stats per category
    ├── shape_examples_<label>.png       grid: 6 example empirical buzz CDFs per label
    ├── pyramidality_vs_difficulty.png   scatter, hue = label
    ├── std_vs_mean.png                  scatter
    └── report.md
```

## Suggested first steps

1. **Plot the empirical buzz CDF for a handful of tossups by hand first.** Pick 3
   famously pyramidal ones, 3 cliffs, 3 duds (eyeball from the data); look at the
   shape. This will sanity-check your metric definitions before you trust them.
2. **Compute metrics in pandas, no model fits required for v1.**
3. **Calibrate the classification thresholds to the dataset.** Don't hardcode 0.85
   — use the empirical distribution.
4. **Sanity-check against `tossup_difficulty.csv`.** `dead_rate` and `d_t` should
   correlate. If they don't, something's wrong upstream or here.

## Decisions baked into v1

- **No per-tossup hazard refit.** This was open question #2 of the survival-model
  HANDOFF: per-tossup baseline hazards would let us write down the *shape* directly,
  but it doubles the model parameter count and needs splines + smoothing. v1 uses
  empirical-CDF metrics that don't need the model. The model-based version is a
  better v2 if any of these metrics turn out to be too noisy at low `n_heard`.
- **Reuse `d_t` from the survival model.** Don't refit difficulty here.
- **Absolute positions, not normalised.** Same justification as the survival model:
  ACF-format tossups have known landmark words (powermark at 75, giveaway around
  130). Normalising would obscure those landmarks.
- **`times_heard >= 3` filter.** Below 3 there's not enough data for shape metrics.
- **`pyramidality` is a coarse rank-based measure.** Real pyramidality testing
  would compare the empirical hazard curve against a strictly-monotonic null. The
  rank-correlation version captures the rough idea without parametric assumptions.
- **No reader-pace adjustment.** Same as the survival model. Buzzpoints are
  word-indexed, not time-indexed, so reader pace partially cancels.

## Open questions punted to v2

1. **Pyramidality formalism.** What's the *right* mathematical definition? Options
   include: rank-based (current), per-bin hazard slope from a per-tossup spline,
   Kullback–Leibler divergence between observed buzz CDF and an idealised reference
   CDF. Each gives a different ranking. v1 uses the rank-based version.
2. **Per-tossup hazard fit.** A natural extension of the survival model — would
   give a principled *shape* parameter per tossup. Decision: punt, because adding
   per-tossup baseline hazards is a major architectural change in `survival-model/`
   and reasonable people would split on whether it's worth it for the marginal
   gain over empirical metrics.
3. **Editorial labels — who validates?** The proposed taxonomy (`pyramidal`,
   `cliff`, `dud`...) reflects one editorial sensibility. A real editor would want
   the labels they use, not these.
4. **Power-mark position.** Hardcoded to 75 here. Different formats use different
   marks; a per-tournament metadata file would generalise this.
5. **"Dud" vs "hard but fair" distinction.** A high-`d_t` tossup is hard. A `dud`
   may be high-`d_t` because it's bad, or because it's hard-but-fair. Disentangling
   needs editorial input — possibly a flag: "would I have written this differently?".
6. **Confidence intervals on metrics.** All metrics are point estimates. Bootstrap
   over games would give CIs but multiplies wall time linearly.
7. **Cross-tournament comparison.** Same tossup heard at multiple tournaments would
   give better shape estimates. Not modelled — only ACF Nationals here.

## Pointers

- `../survival-model/results-1/tossup_difficulty.csv` — primary difficulty input.
- `../buzzpoints_data.csv` — raw buzz positions per (tossup, game, player).
- `../survival-model/HANDOFF.md`, "What this is most useful for / Question quality
  analysis" paragraph — direct motivation for this folder.
- Match the existing per-method folder pattern: code at the top of this folder,
  results in `question-quality-analysis/results-*/`, paths in scripts relative
  to this folder.
