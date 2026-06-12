# Handoff: Player Profiling from Buzzpoint Data

## Context

This repo (`buzzpoints-playground`) explores statistical methods for ranking and analysing
quizbowlers from per-tossup data. Two models already live here:

- `../plackett-luce/` — Plackett-Luce model with per-tossup difficulty. Outcomes only
  (who won / dead). Produces a single skill scalar per player.
- `../survival-model/` — discrete-time competing-risks hazard model. Adds buzz position.
  Produces skill, plus a `player_speed_profile.csv` with mean residual buzzpoint.

Both reduce a player to one or two scalars. **Player profiling is what they leave on
the table.** Two players with identical "skill" (buzz-first rate) often look very
different when you decompose along multiple axes — early-vs-late, accurate-vs-trigger,
deep-vs-broad, generalist-vs-specialist.

The current folder (`player-profiling/`) is the home for that decomposition. It is
strictly downstream of the existing models: it consumes their outputs and the raw CSV,
and does not refit the survival or PL hazard.

## What this is for

Headline question: **what does this player look like?** Not just "are they good", but
"how are they good". Concretely:

- Are they fast or slow on conversions, holding tossup difficulty fixed?
- Are they accurate or buzz-happy (correct rate among buzzes / neg rate)?
- Are they a generalist (skill flat across categories) or a specialist (skill spikes
  in 1–2 categories)?
- Where do they buzz on the difficulty curve — easy questions in the giveaway, or
  hard questions in the powermark?
- Which other players cluster with them (archetype assignment)?

The motivation is editorial / narrative: writing about a player or comparing two,
choosing draft picks for an all-star team, talking about the "shape" of a roster.

## Inputs (already produced upstream)

1. `../plackett-luce/results-2/overall_rankings.csv` — overall skill `s_PL`
2. `../plackett-luce/results-2/category_skills.csv` — per-category skills (NaN
   below threshold)
3. `../survival-model/results-1/overall_rankings.csv` — survival skill `s_surv`
4. `../survival-model/results-1/player_speed_profile.csv` — `mean_residual_bp`
5. `../survival-model/results-1/tossup_difficulty.csv` — `d_t` per tossup
6. `../buzzpoints_data.csv` — raw buzz events (one row per heard tossup)

If those folders don't exist yet, run them first per their own READMEs. The fits are
deterministic; this analysis layer just slices and joins.

## Profile dimensions

For each player with `n_correct >= 10` we compute:

| Dimension | Definition | What it captures |
|-----------|------------|------------------|
| `skill_pl`         | PL skill (with difficulty), centred | Who-wins propensity |
| `skill_surv`       | Survival skill, centred             | Same, position-aware |
| `speed`            | `-mean_residual_bp` (sign flip so + = faster) | Fast guesser vs late convertor |
| `accuracy`         | `correct / (correct + neg + wrong-non-neg)` over all buzzes | Trigger discipline |
| `neg_rate`         | `neg / heard` | How often they're wrong |
| `power_rate`       | `correct buzzes at `bp <= 75` / correct` | Strength of early signal |
| `breadth`          | `# categories with skill estimate >= threshold` | Generalist vs specialist (count) |
| `specialization`   | `max(category_skill) − mean(category_skill)` | Generalist vs specialist (continuous) |
| `hard_skill`       | mean residual buzz on tossups with `d_t > 75th pctile`, sign-flipped | Conversion depth on hard tossups |

All dimensions are computed on aggregated categories (Fine Arts, History, Lit, Sci,
Beliefs, Thought, Other) — same as the upstream models.

## Archetype clustering

After standardising the dimensions above per-player, k-means with `k = 5` (default)
groups players into archetypes. The choice of `k` is somewhat arbitrary — a silhouette
plot is also produced so the reader can see the trade-off. Suggested labels (assigned
by inspection post-hoc, not hardcoded):

- *fast generalist* — high `speed`, low `specialization`, mid-high `skill`
- *deep specialist* — high `specialization`, low `speed`, high `skill` in 1 category
- *trigger-happy* — high `power_rate`, lower `accuracy`, often higher `neg_rate`
- *balanced* — close to mean on every axis
- *late convertor* — low `speed`, high `accuracy`, mid `skill`

Cluster centroids (in standardised coordinates) are written to a CSV; the reader
labels them.

## Outputs

```
player-profiling/
├── HANDOFF.md
├── README.md
├── model.py
└── results-*/
    ├── player_profiles.csv              one row per qualifying player; all dims
    ├── archetypes.csv                   cluster_id + centroid coords
    ├── archetype_assignments.csv        player -> cluster
    ├── radar_<player_slug>.png          radar chart per top-30 player
    ├── pca_scatter.png                  2D PCA of profile space, coloured by cluster
    ├── speed_vs_skill.png               scatter, useful sanity-check view
    ├── specialization_vs_breadth.png
    └── report.md
```

## Suggested first steps

1. **Wire up data joins.** Load all five upstream CSVs + raw buzz data; left-join on
   `player_slug`. Sanity-check counts. Players who appear in raw data but not in
   skill tables are below the buzz threshold for one or both upstream models — keep
   them in raw stats but mark `skill_pl` / `skill_surv` as NaN.
2. **Compute the dimensions.** Each of the 9 dimensions is a one-liner over the joined
   frame. Watch for divide-by-zero on rare-buzzer players (gate to `n_correct >= 10`).
3. **Standardise + cluster.** Z-score each dimension; impute NaN with 0 (the prior mean
   under L2). k-means with random restarts; report inertia and silhouette across `k=3..8`.
4. **Visualise.** Radar chart per top-30 player; PCA scatter coloured by cluster.
5. **Report.** Markdown summary with a paragraph per cluster citing exemplar players.

## Decisions baked into v1 (see "Open questions" for ones we punted)

- **No new model fit.** All dimensions are derived from upstream model outputs + raw
  CSV summaries. Adding a structured profile model (e.g. multi-output GP over
  `category × position`) is deferred — see open questions.
- **k = 5 archetypes.** Round-number default. The output includes silhouette across
  `k = 3..8` so a different value can be picked post-hoc.
- **Standardisation by z-score, no PCA whitening before clustering.** Z-score is
  robust to outliers via `RobustScaler`-style winsorisation at ±3σ; PCA whitening
  would mix the dimensions before clustering, defeating interpretability.
- **k-means, not hierarchical / GMM.** k-means clusters are convex and interpretable;
  GMM components are easier to misread on small samples.
- **Sign convention: positive = "good".** `speed`, `accuracy`, `power_rate` are flipped
  if needed so radar charts read as "outward = better player".

## Open questions punted to v2

These would introduce technical debt if guessed wrong, so flag them before deciding:

1. **Headline output format.** Is this primarily for editorial blog posts (radar
   charts + paragraphs), for fantasy-draft style numerical comparison (CSV + ranking),
   or for live commentary (interactive)? v1 produces all three artefacts but biases
   toward editorial.
2. **Cluster count `k`.** k=5 is a guess. Domain knowledge would pick this; e.g. the
   user might want exactly 3 archetypes for a blog post, or 7 for a draft tool.
3. **Whether to fit a joint profile model.** Composing existing model outputs is
   simple but loses statistical efficiency. A joint multi-output model (e.g. survival
   model with per-player position-shape coefficients) would let `speed` and
   `specialization` borrow strength from each other. Big v2 ask.
4. **Cross-tournament profiles.** Only ACF Nationals 2026 is loaded. A player's
   profile probably stabilises only across an entire season; aggregating multiple
   tournaments needs harmonised category labels and a normalisation across difficulty
   levels (open / regs / nationals).
5. **Confidence intervals.** Profile dimensions are point estimates. The upstream
   models have bootstrap CIs for skill but not for the derived dimensions. A proper
   bootstrap of the whole pipeline would be honest but expensive.
6. **Temporal effects within a tournament.** Are players faster/slower in late rounds?
   A player's profile averages over rounds; a per-round breakdown could add `fatigue`
   as a tenth dimension. Likely under-powered on a single tournament.

## Pointers

- `../plackett-luce/model.py:60-67` — `aggregate_category` mapping. Reuse it.
- `../survival-model/model.py:578-601` — `player_speed_profile` definition. The
  `speed` dimension here is a sign-flip of that same residual.
- `../survival-model/HANDOFF.md` — original framework doc; this folder picks up its
  player-profiling motivation directly.
- Match the existing per-method folder pattern: code at the top of this folder,
  results in `player-profiling/results-*/`, paths in scripts relative to this folder.
