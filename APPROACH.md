# Approach: Three downstream analyses on top of the buzzpoint models

This document is the single-source-of-truth for the three top-level folders that
were added on top of `plackett-luce/` and `survival-model/`:

- `player-profiling/` — multi-dimensional player profiles + archetype clustering
- `question-quality-analysis/` — tossup *shape* metrics + editorial classification
- `counterfactuals-prediction/` — predictive primitives (win prob, live, swap)

It is intended to be sufficient on its own to reproduce all three from a fresh
checkout: read this file, run the documented commands, get the same artefacts.

## Why these three exist

The end of `plackett-luce/2026-acf-nationals-results/report.md` says:

> I believe the far more interesting use for buzzpoints would be in answering a
> number of other questions, such as player profiling, question quality analysis
> and counterfactuals prediction.

Each of the three folders maps to one of those questions. They are layered on top of
the existing fits — they do not refit the PL or survival models — so they can be
re-run cheaply when the upstream models change.

## Repository layout

```
buzzpoints-playground/
├── buzzpoints_data.csv                  raw events (one row per heard tossup)
├── plackett-luce/                       outcome-only model (existed already)
│   ├── model.py
│   └── results-2/{overall_rankings, category_skills, tossup_difficulty}.csv
├── survival-model/                      buzz-position-aware model (existed already)
│   ├── model.py
│   └── results-1/{overall_rankings, tossup_difficulty, player_speed_profile}.csv
├── player-profiling/                    NEW
│   ├── HANDOFF.md
│   ├── README.md
│   └── model.py
├── question-quality-analysis/           NEW
│   ├── HANDOFF.md
│   ├── README.md
│   └── model.py
├── counterfactuals-prediction/          NEW
│   ├── HANDOFF.md
│   ├── README.md
│   └── model.py
└── APPROACH.md                          this file
```

Each new folder mirrors the existing-folder pattern: `HANDOFF.md` (design brief),
`README.md` (how to run + interpret), `model.py` (entry point), `results-*/`
(per-run artefacts). HANDOFF and README are deliberately separated: HANDOFF is the
design rationale and the open questions; README is the operational manual.

## Reading order

If you are coming to this fresh:

1. `plackett-luce/2026-acf-nationals-results/report.md` — context and motivation.
2. `survival-model/HANDOFF.md` — the framework that buzz position adds.
3. The three new HANDOFF.md files in any order — they are independent.
4. The three new README.md files for run instructions.

## Dependencies between folders

Both upstream models must be run before any of the three new folders, because the
new folders consume their saved CSVs:

```
plackett-luce/results-2/
    └── consumed by: player-profiling, question-quality-analysis (discrimination metric)
survival-model/results-1/
    └── consumed by: player-profiling, question-quality-analysis,
                     counterfactuals-prediction
```

The three new folders are independent of each other.

## End-to-end reproduction

This is the canonical sequence to reproduce every artefact in this repo from a fresh
checkout. All paths are relative to the repo root.

```bash
# 1. Plackett-Luce model (existing)
cd plackett-luce
python model.py --input ../buzzpoints_data.csv --output-dir results-2 --bootstrap 200
cd ..

# 2. Survival model (existing)
cd survival-model
python model.py \
    --input ../buzzpoints_data.csv \
    --output-dir results-1 \
    --bin-width 5 --lam-s 0.1 --lam-d 0.5 --lam-a 0.01 --bootstrap 50
cd ..

# 3. Player profiling
cd player-profiling
python model.py --output-dir results-1 --k 5 --n-correct-min 10
cd ..

# 4. Tossup quality analysis
cd question-quality-analysis
python model.py --output-dir results-1 --power-mark 75 --n-heard-min 3
cd ..

# 5. Counterfactuals & live prediction
cd counterfactuals-prediction
python model.py --output-dir results-1
cd ..
```

Total wall time on a 14-core box: ~30–40 minutes (dominated by the survival-model
bootstrap and the counterfactuals baseline-hazard refit).

## Design choices common to all three folders

- **No new model fits in the new folders.** Player profiles and tossup-quality
  metrics are derived stats on top of the saved fits + raw CSV. The
  counterfactuals folder re-derives only the baseline hazard `a_b` (a 1-D
  conditional fit, ~4 minutes), reusing `s` and `d` from disk. This keeps the new
  layer cheap to iterate on, and it keeps the upstream model code unmodified.
- **Aggregated categories.** All folders use the same 7-category aggregation as
  the upstream models (Fine Arts, History, Literature, Science, Beliefs, Thought,
  Other). The mapping lives in each `model.py`; do not introduce a third copy.
- **Markdown reports without `tabulate`.** `pandas.DataFrame.to_markdown` is
  avoided (extra dep); each folder uses an inline `df_to_md` helper or string
  composition.
- **No internet, no caching layer.** All inputs are local CSVs. Re-running
  produces identical artefacts modulo random-seeded operations (k-means,
  bootstrap), which are seeded.

## Per-folder design summary

### `player-profiling/`

**Goal**: produce a richer description of each player than a single skill scalar.

**Approach**: derive 9 dimensions per player by joining the PL/survival CSVs with
the raw buzz data. Standardise; k-means cluster (k=5 default); auto-label clusters
with a heuristic; render radar charts for the top 30 players, plus a PCA scatter
coloured by cluster.

The 9 dimensions intentionally mix outcome-style scalars (`skill_pl`,
`skill_surv`), behavioural rates (`accuracy`, `neg_rate`, `power_rate`), shape
descriptors (`speed`, `hard_skill`), and category structure (`breadth`,
`specialization`). Clustering them gives a coarse-grained taxonomy of player
"types" — not a substitute for ranking.

**Headline output**: `report.md` — silhouette across `k=3..8`, cluster centroids
+ labels, top-30 ranking with cluster annotation, exemplar players per cluster.

### `question-quality-analysis/`

**Goal**: editorial-grade per-tossup analysis. "Was this a well-formed pyramidal
tossup, or a cliff, or a dud?"

**Approach**: model-free metrics on the empirical buzz distribution per tossup.
The headline metric is `pyramidality` (Spearman ρ between bin index and bin
buzz-count, clipped to [0, 1]), backed up by `cliff_score`, `slope_score`,
`mean_buzz`, `std_buzz`, `power_rate`, and `discrimination`. A priority-ordered
classifier maps each tossup to one of `{dud, gimme, cliff, back-loaded,
pyramidal, front-loaded, noisy}`.

We do **not** fit per-tossup hazard shapes (open question #2 of the survival
model HANDOFF). Per-tossup splines would give a model-based pyramidality but
need careful regularisation; v1 is purely empirical.

**Headline output**: `report.md` and `tossup_quality.csv` (one row per tossup
with all metrics + label).

### `counterfactuals-prediction/`

**Goal**: make the survival-model parameters useful for prediction, not just
ranking.

**Approach**: load `(s, d)` from disk, refit `a_b` conditional on them, then
derive three closed-form primitives:

- `p_win_outcome` integrates per-bin hazards over the question length to give a
  marginalised win probability for an arbitrary roster.
- `p_event_at_bin` is the per-bin multinomial for "what happens next" given a
  state.
- `simulate_game` forward-simulates a packet between two rosters.

The CLI exposes `predict`, `replay`, and `swap` subcommands.

**Headline outputs**:
- `per_event_predictions.csv` — model's predicted P(actual winner wins) per game
- `roster_swap_examples.csv` — 4 hand-picked roster swaps with ΔE(wins)
- `calibration_plot.png` — reliability diagram for `p_dead`
- `report.md`

## Significant blockers / decision points (require user input)

These are decisions where guessing wrong introduces real technical debt. They
are surfaced here and **not** silently picked. Each is also called out in the
relevant HANDOFF.md:

1. **(player-profiling) Cluster count `k`.** Defaulted to 5; silhouette is flat
   across k=3..8 (range 0.19–0.25), so the choice is somewhat arbitrary. Should
   be picked based on intended use: 3 archetypes for a blog-post taxonomy, 7+
   for a fantasy-draft tool. **Default chosen: k=5.**
2. **(player-profiling) Headline format.** Editorial radar charts vs.
   numerical-comparison CSV vs. live commentary. v1 produces all three but
   biases toward editorial. The user's intended audience determines which is
   the "real" output.
3. **(question-quality-analysis) Pyramidality formalism.** Rank-based Spearman
   ρ, hazard-slope, KL-divergence to an idealised CDF — each gives a different
   ranking. v1 uses the rank-based version; a model-based replacement is a v2
   ask. **Default: Spearman ρ of bin index vs bin density.**
4. **(question-quality-analysis) Editorial label thresholds.** The cutoffs
   (`dead_rate > 0.5` for `dud`, `cliff_score > 0.85` for `cliff`, etc.) are
   chosen to give a sensible distribution at ACF Nationals 2026. They are not
   universally correct. A real editor would pick their own.
5. **(counterfactuals-prediction) Headline use case.** Roster-swap
   counterfactuals, in-game live prediction, or model validation? v1 produces
   all three but the CLI is structured around one-off lookups. A primary
   audience would dictate whether the CLI should become a notebook, a
   web-server, or a streaming feed.
6. **(counterfactuals-prediction) `a_b` not persisted by survival-model.** The
   counterfactual model re-derives `a_b` (4 minutes per startup). Easy fix:
   add `--save-baseline` to `survival-model/model.py`; the counterfactuals
   model would then load it directly and start instantly.

## Significant blockers / areas of ambiguity (technical, not user-facing)

These are limitations or gotchas that future maintainers should be aware of:

7. **Predictions assume static fields.** The counterfactuals model integrates
   per-bin hazards as if every eligible player races the whole tossup. In reality
   players neg out, shrinking the field. For *counterfactual* questions ("what
   if alice were here") this is correct (we don't know who would neg). For
   *evaluating* historical predictions, this systematically depresses the
   predicted probability of the actual winner — the per-event mean P(actual
   winner wins) is 0.10, just below the uniform baseline of 1/n_eligible ≈ 0.17.
   This is documented in the report; it is not a bug, but it is a metric to
   interpret with care.
8. **`d_t` sign convention.** Higher `d_t` in the survival model means *harder /
   more dead-prone*, but it appears as `exp(-d_t)` in the dead-competitor
   hazard. The sign flip is internal to the model and consistent — but if you
   add new prediction code, double-check which direction your `d_t` should push
   probabilities.
9. **No held-out evaluation.** Per-event predictions are on the same data the
   underlying model was fit on, so the calibration is in-sample. A leave-one-game-out
   evaluation would be honest but cost ~hundreds of refits.
10. **No bootstrap propagation through the new layer.** The PL and survival
    bootstrap CSVs on disk are not consumed by the three new models. All
    derived statistics (profile dimensions, quality metrics, predictions) are
    point estimates. Honest CIs would require running the whole pipeline once
    per bootstrap sample.
11. **Cluster auto-labels are heuristic.** `label_clusters` in
    `player-profiling/model.py` uses a hand-coded rule cascade. The labels are
    indicative; they should not be quoted as ground truth.
12. **`tabulate` not assumed installed.** Each new `model.py` includes its own
    inline `df_to_md` helper rather than relying on `pandas.to_markdown`.
13. **Single-tournament dataset.** Everything is fit on 2026 ACF Nationals
    only. Cross-tournament generalisation is out of scope for v1.

## How to extend (rough roadmap)

If a future maintainer wants to push this further:

- **Persist `a_b` from the survival model.** One-line change in
  `survival-model/model.py`; saves 4 minutes per counterfactuals run.
- **Per-tossup hazard shapes.** Open question #2 of the survival model HANDOFF.
  Would replace the model-free metrics in `question-quality-analysis/` with a
  proper shape parameter per tossup. Adds splines + smoothing; ~doubles the
  parameter count.
- **Bootstrap propagation through the new layer.** Re-run each new model on
  every bootstrap sample of the underlying PL/survival fit; report 95% CIs on
  every derived statistic. Linear in bootstrap iterations.
- **Cross-tournament profiles.** Aggregate multiple tournaments into a single
  player-profile run, with care about category-label harmonisation and
  per-tournament difficulty normalisation.
- **Web frontend for live prediction.** Wrap `Predictor.p_event_at_bin` in an
  HTTP service.

## Closing note

This document, the three HANDOFF.md files, and the three README.md files are
intended to be self-contained: someone reading them in order should be able to
recreate everything in this repo from the raw CSV. The three folders deliberately
*do not* edit the existing `plackett-luce/` or `survival-model/` code — they are
strictly downstream consumers.
