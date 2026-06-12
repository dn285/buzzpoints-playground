# Handoff: Buzz-Position-Aware Quizbowl Ranking Model

## Context

This repo (`buzzpoints-playground`) explores statistical methods for ranking quizbowlers from per-tossup data. The existing approach lives in `../plackett-luce/` and uses a Plackett-Luce model with per-tossup difficulty parameters fit via L-BFGS-B; see `../plackett-luce/README.md` for the full methodology. It models *outcomes* only — who won each tossup, or whether it went dead — and ignores *when* the buzz happened.

This handoff is for building a new method that uses buzz-position data. The current folder (`survival-model/`) is the home for the new code — rename if a different framework feels more natural once you've sketched the design.

## What buzz position adds

The current model treats "alice won by 5 words" and "alice won by 30 words" as identical outcomes. Position adds:
- *Margin* on conversions — useful for distinguishing fast guessers from deep-knowledge players
- A per-tossup *information curve* (when does enough info appear?), separate from the binary "did anyone get it"
- Better discrimination among very close players where outcomes are noisy but speeds differ systematically

## What this is most useful for

**Ranking**: Modest gains. Outcome data already captures most of the signal. The top/middle/bottom of the rankings probably won't reshuffle much; some close gaps near the top might tighten or flip.

**Player profiling**: This is where position really shines. Two players with identical skill estimates can have very different buzz-time distributions — early-buzzing generalists vs late-converting specialists. The current model can't distinguish them.

**Question quality analysis**: Tossup *shape*. Two tossups with the same dead rate can look very different — one rewards depth across the whole question (smooth difficulty curve), one is a sudden cliff (most buzzes at the same position). This is editorially valuable, separate from rankings.

**Counterfactuals and live prediction**: "If alice had been in this room, when would she have buzzed?" or "Given the question is 30 words in with no buzz, what's P(carol buzzes next)?"

So if the goal is purely "who's better," the existing model is probably enough. The motivation for going further is one of the other use cases — make sure to be clear about which one is driving the design.

## Data

`../buzzpoints_data.csv` has one row per (tossup, game, player) triple:
- `tossup_slug`, `game_id`, `category`, `player_slug`, `team_slug`, `opponent_slug`
- `buzzpoint` — position of the buzz (NaN if no buzz)
- `buzz_value` — `-5` for neg, `10` for correct, `15` for powermark
- `won` — boolean, true for the first correct buzz in each game

Things to check up front:
- Distribution of `buzzpoint` (verify the unit — word index? character offset?)
- Whether tossup length is known (need it to set the maximum position / to censor dead tossups)
- How dead tossups appear in the data — does `buzzpoint` get set to max length, or stay NaN?
- Whether reader pace is recoverable from the data (it's a confounder — see below)

## Modeling framework

This is a **competing risks survival problem**: for each (game, tossup), multiple players are racing to buzz, the first one wins, and the tossup itself has a "no one buzzes by the end" hazard.

The natural starting point is a **Cox proportional hazards model**:
- Each player has a hazard rate parameter exp(s_i)
- The baseline hazard h_0(p) is shared across tossups (or per-category)
- Each tossup has its own difficulty multiplier exp(-d_t)

The hazard for player i on tossup t at position p (given they haven't buzzed):
```
h_i(p) = h_0(p) * exp(s_i - d_t)
```

This is a direct continuous-time generalization of the current Plackett-Luce model — at any single position, the relative hazard ratios reduce to PL. The new information comes from *when* events happen, not just their order.

Alternative formulations worth considering:
- Discrete-time logistic per word/bin (simpler to fit, more parameters)
- Latent-variable model where each player has a buzzpoint drawn from some distribution (Thurstone-ish)
- Per-tossup baseline hazard curves (more flexible, way more parameters — needs regularization)

Negs need careful handling: they're a separate competing risk (player buzzes incorrectly, exits the race, but tossup continues). The simplest start is to treat a neg as right-censoring for the negger — they exit at position p, the others keep racing — and not model neg-rate-vs-skill at all. Modeling neg propensity as a function of skill is a nice-to-have, not a v1 requirement.

## Key design questions to resolve early

1. **Continuous vs discrete time.** Cox PH is continuous; discretizing into per-word bins is easier to fit and lets you use standard logistic regression machinery, at the cost of some precision.
2. **Shared baseline hazard, or per-tossup curve.** Shared = far fewer parameters but can't capture per-tossup *shape* (pyramidal vs sudden cliff). Per-tossup needs regularization (smoothing splines, low-rank decomposition) to avoid overfitting.
3. **Tossup length variability.** Different questions are different lengths. Either normalize position to [0, 1] of question length, or model length explicitly. The right choice depends on whether the "information curve" is in absolute or relative terms.
4. **Reader pace as a confounder.** Some readers are faster than others. If pace correlates with field strength (e.g., better teams in faster rooms), it'll bias skill estimates. Worth checking the empirical correlation before deciding whether to adjust.
5. **Identifiability.** Like the PL model, the s/d split has an additive constant. Center s post-fit. With per-tossup baselines, there's an additional shape-vs-difficulty identifiability issue — be explicit about what's being absorbed by what.

## Suggested first steps

1. **Exploratory analysis first, model second.** Plot the distribution of `buzzpoint` overall, by category, and conditional on (correct vs neg vs dead). Plot the empirical hazard rate as a function of position (for a strong player, weak player, easy tossup, hard tossup). This will heavily inform whether per-tossup shapes are needed or whether a shared baseline is enough.
2. **Simplest viable model.** Cox PH with shared baseline, single difficulty per tossup. Use `lifelines` for the prototype, switch to a custom scipy/JAX fit if performance matters. Compare skill estimates to the PL model — they should be highly correlated for prolific players, with more uncertainty for low-buzz players.
3. **Player profiling output.** For each player, compute residual mean buzzpoint after controlling for tossup difficulty. This separates "skill" from "speed" and is the headline output that motivates the whole exercise.
4. **Validation.** Held-out per-event log-likelihood, splitting by game (same as the PL model's CV). Compare to PL baseline — does the position-aware model predict the *order* of buzzes meaningfully better, or are the ranking-relevant gains marginal?

## Pointers

- `../plackett-luce/model.py` — current model implementation. Useful to crib `load_data` and `build_tossup_records`, plus the game-level cluster bootstrap structure (you'll want the same for CIs).
- `../plackett-luce/README.md` — full methodology writeup for the existing approach. The "What this doesn't model" section is the direct motivation for this work.
- Cluster bootstrap by game (not by event) is non-negotiable — within-game observations share players, reader, pace.
- Match the existing per-method folder pattern: code at the top of this folder, results in `survival-model/<tournament-name>-results/`, paths in scripts relative to this folder (`../buzzpoints_data.csv`).
