# Player Profiling

Multi-dimensional profile of every player at 2026 ACF Nationals + k-means archetype
clustering. Strictly downstream of the Plackett-Luce and survival-model fits — it
joins their saved CSVs with the raw buzz data and derives 9 profile dimensions per
player.

See `HANDOFF.md` for design rationale and open questions.

## Profile dimensions

| Dimension | Source | Meaning (high values mean…) |
|-----------|--------|-----------------------------|
| `skill_pl`         | PL `overall_rankings.csv` | Conventional outcome-only skill |
| `skill_surv`       | Survival `overall_rankings.csv` | Position-aware skill |
| `speed`            | `-mean_residual_bp` | Buzzes earlier than average for the same tossup |
| `accuracy`         | raw CSV | High correct-rate among buzzes |
| `neg_rate`         | raw CSV | Negs more often per tossup heard |
| `power_rate`       | raw CSV | Higher fraction of correct buzzes are powers |
| `breadth`          | PL `category_skills.csv` | Qualifies in more categories |
| `specialization`   | same | Best category much better than average |
| `hard_skill`       | raw + survival difficulty | Buzzes earlier on hard tossups |

## Run

```bash
cd player-profiling
python model.py --output-dir results-1 --k 5
```

Defaults assume `../plackett-luce/results-2/` and `../survival-model/results-1/`
already exist.

## Outputs

- `player_profiles.csv` — all 9 dims per player
- `archetypes.csv` — cluster centroids in standardised space, with auto-generated label
- `archetype_assignments.csv` — player → cluster, with cluster label
- `pca_scatter.png` — 2D PCA, coloured by cluster
- `speed_vs_skill.png`, `specialization_vs_breadth.png`
- `radars/radar_<player_slug>.png` — top-30 players by survival skill
- `report.md` — narrative summary

## Reproducing

```bash
git checkout main
cd player-profiling
python model.py --output-dir results-1 --k 5 --n-correct-min 10
```

Wall time: ~5 seconds.
