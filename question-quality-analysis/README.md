# Tossup Quality Analysis

Empirical shape metrics per tossup at 2026 ACF Nationals + a coarse editorial
classification. Strictly downstream of the survival-model fit (reuses its `d_t`).

See `HANDOFF.md` for design rationale and open questions.

## Metrics computed per tossup (with `n_heard >= 3`)

| Metric | Meaning |
|--------|---------|
| `dead_rate`      | dead games / total games |
| `mean_buzz`      | mean correct-buzz word index |
| `std_buzz`, `iqr_buzz` | spread of correct buzzes |
| `power_rate`     | fraction of correct buzzes at `bp ≤ 75` |
| `pyramidality`   | Spearman ρ of (bin index, bin density), clipped to [0,1] |
| `slope_score`    | + = front-loaded relative-position distribution; − = back-loaded |
| `cliff_score`    | `1 − iqr_buzz / length`, high = trigger word |
| `discrimination` | corr(buzzed-correct, PL skill) across players who heard it |
| `difficulty_d`   | survival-model `d_t` (reused, not refit) |

## Editorial labels (priority order)

1. `dud` — `dead_rate > 0.5`
2. `gimme` — `rel_mean_buzz < 0.4` AND `dead_rate < 0.05`
3. `cliff` — `cliff_score > 0.85`
4. `back-loaded` — `rel_mean_buzz > 0.85`
5. `pyramidal` — `pyramidality > 0.5` AND `dead_rate < 0.3`
6. `front-loaded` — `rel_mean_buzz < 0.55`
7. `noisy` — everything else

## Run

```bash
cd question-quality-analysis
python model.py --output-dir results-1
```

## Outputs

- `tossup_quality.csv` — every metric + label per tossup
- `classification_counts.csv` — count per label
- `per_category_quality.csv` — mean metrics per category
- `pyramidality_vs_difficulty.png`, `std_vs_mean.png`
- `shape_examples_<label>.png` — empirical buzz CDFs for 6 example tossups per label
- `report.md`

## Reproducing

```bash
git checkout main
cd question-quality-analysis
python model.py --output-dir results-1 --power-mark 75 --n-heard-min 3
```

Wall time: ~10 seconds.
