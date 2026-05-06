# 2026 ACF Nationals Report

I ranked quizbowlers based on per-question performance at 2026 ACF Nationals. My analysis is based on the Plackett-Luce model, an extension of the Bradley-Terry model, which itself generalizes the Elo system. Of particular note is that, unlike other analyses, this does not factor in buzzpoint data whatsoever -- only who got which tossup and against which opponents.

## Overview
For each tossup, in each game, a player is the "winner" if they buzzed correctly first. Every other player in the room is a "loser" for that tossup. The model uses these per-tossup outcomes to estimate a single skill parameter for each player.

## The basic model
For tossup `t` with eligible player set `S_t`, the probability that player `i` wins is:

```
P(i wins | S_t) = exp(s_i) / sum_{k in S_t} exp(s_k)
```

where `s_i` is player `i`'s skill parameter. The log-likelihood across all tossups is:

```
L = sum_t [ s_{w_t} - log( sum_{k in S_t} exp(s_k) ) ]
```

where `w_t` is the winner of tossup `t`. We maximize this with L-BFGS-B (`scipy.optimize.minimize`) and an L2 penalty on `s` to keep the problem well-posed (the model is identified only up to an additive constant — we then center the skills at zero post-fit). The penalty strength is `lambda = 0.1`. Players with very few observations are pulled toward the prior mean of zero, which is the desired behavior.

## Handling negs
A naive implementation drops any tossup with a neg, on the grounds that the model has no explicit way to penalize incorrect buzzes. I did this initially, but it threw away a lot of data, many of which were on tossups with hard-to-convert answerlines, which are a powerful differentiator among top players. Instead, I dropped only the negging player from the eligible set for that tossup, meaning the rebound winner faced a smaller field, which Plackett-Luce handles natively. 

## Question difficulty
The original model also dropped dead tossups, since there is no winner to credit. However, throwing those events away means systematically underestimating the skill of players in tough fields and ignoring the strongest signal we have about which questions are difficult.

We extend the model with a per-tossup difficulty parameter `d_t`. Each unique tossup gets one `d_t` shared across every game it was read in (the question's difficulty doesn't change between rooms). Conceptually, the tossup itself is a "virtual competitor" in the race:

```
P(player i wins) = exp(s_i)  / ( exp(d_t) + sum_{k in S_t} exp(s_k) )
P(tossup goes dead) = exp(d_t) / ( exp(d_t) + sum_{k in S_t} exp(s_k) )
```

Higher `d_t` means a harder question. The log-likelihood adds one term per event:

- For events with a winner: `s_{w_t} - log( exp(d_t) + sum_{k in S_t} exp(s_k) )`
- For dead events:         `d_t      - log( exp(d_t) + sum_{k in S_t} exp(s_k) )`

Skill and difficulty parameters are concatenated and fit jointly. Difficulty gets its own L2 regularization with strength `lambda_d` (separate from the skill `lambda`). Post-fit, we center the player skills at zero and the difficulties end up on the same scale.

This change allows us to use every game-tossup event as data -- including dead-after-neg events, where the negging player is excluded but the question still goes dead -- instead of silently throwing away ~30% of the dataset.

## Choosing lambda_d
The right value of `lambda_d` is not obvious. Excessively low values would overfit difficulty estimates from low-frequency tossups, while excessively high values would reduce to the no-difficulty model. I used 5-fold cross-validation, splitting at the game level (so within-game dependence doesn't leak between train and validation). The skill `lambda` is left at 0.1; in principle the same procedure could choose it, but the rankings are not very sensitive to it within the reasonable range.

When this was performed, the CV scores were:

```
lam_d=0.01 : -1.9089
lam_d=0.05 : -1.9058
lam_d=0.1  : -1.9043
lam_d=0.5  : -1.9022  <-- best
lam_d=1.0  : -1.9085
lam_d=5.0  : -1.9397
```

As such, `lam_d=0.5` was selected. It is worth noting that the curve is fairly flat between 0.05 and 1.0 (about 0.007 nats of difference per event). This is reassuring, as it indicates the rankings don't hinge on a regularization choice. The drop-off at `lam_d=5.0` confirms that strong shrinkage starts to wash out the difficulty signal.

## Confidence intervals
A point estimate is not sufficient, we want to know whether the gap between two players is real or could plausibly reverse on a different sample of games. I used a cluster bootstrap, resampling whole games with replacement and refitting the joint skill+difficulty model on each resample. I resampled at the game level, rather than the tossup level, because tossups within a game are not independent. Similarly, event-level resampling would produce artificially narrow intervals.

For each player we report the 2.5th–97.5th percentile of skill across resamples, plus the empirical probability of finishing in the top 5 / 10 / 20. The bootstrap refits both skill and difficulty parameters but only reports CIs for skills (the difficulty CIs aren't part of the analysis output). 

## Per-category models
For each of the seven aggregated categories (Fine Arts, History, Literature, Science, Beliefs, Thought, Other), we fit the same skill+difficulty model on just that category's tossups. Each player ends up with one overall skill estimate and up to seven per-category estimates.

To reduce noise from small samples, players are excluded from a category's ranking if they have fewer than ~10 correct buzzes in that category (the threshold varies a bit by category — categories with fewer total tossups use a lower bar). The bootstrap also runs per category and reports `P(top 3)` and `P(top 5)` instead of top 5/10/20.

## What this doesn't model
The current model uses only the outcome of each tossup (who won, or whether it went dead). It ignores buzz position — alice winning by 5 words and alice winning by 30 words count identically. Position would add real information about *margin* and could distinguish fast guessers from deep-knowledge players, but it requires a different modeling framework (survival / Cox proportional hazards, with a per-tossup baseline hazard curve) and is partly captured indirectly by the difficulty parameters anyway. For ranking purposes, outcome is most of the signal.

## According to this model…

**Q: Who are the best players?**

A: [These players.](2026-acf-nationals-results\player-rankings.md)

Performed the analysis above and ranked players by their skill parameters (both overall and per-category). Note that skill parameters should not be compared across tournaments or subjects, as only the relative difference between players within a given ranking is meaningful. The value of the parameter itself is also somewhat arbitrary, as the model just centers it at the mean (this is why many top players in the subject rankings have "negative" overall skill).

**Q: Which tossups are the hardest?**

A: [These tossups.](2026-acf-nationals-results\hardest-tossups.md)

Sorted the tossups using the difficulty estimates. The difficulty parameters double as a question-quality signal. Tossups that went dead in most rooms get high `d_t`; tossups that were quickly converted get low `d_t`. This is a useful sanity check (genuinely hard tossups should rank near the top) and a standalone output in its own right.

**Q: Which categories are the most specialized?**

A: Science and nothing else comes close.
![](2026-acf-nationals-results\residual_variance.png)

For each category, compute the variance of (category skill - overall skill) across players. Higher variance indicates a more specialist-driven category.

**Q: Is there any correlation or trade-off between categories?**

A: Probably, but this would be better-determined from a larger dataset. 
![](2026-acf-nationals-results\category_correlation.png)

Computed pairwise Pearson correlations between category skill vectors across players. This reveals which categories tend to co-occur in players (positive correlation) and which trade off (negative correlation). 

It is worth noting that this analysis is more accurate than PCA, which would have functionally just an approximation of the cross-category correlation matrix.

## Thoughts

If one were to only rank players based on 2026 ACF Nationals performance and if there were such a thing as an "objective" ranking, I believe it would be more or less the rankings produced by this method. While many player placements here were initially extremely surprising to me, my intuition was consistently proven "incorrect" when I actually inspected that player's buzzpoints and compared it against the other players ranked above and below them. 

I put "incorrect" in quotes because, of course, there are many other factors to consider. There was a whole season worth of other tournaments and two other nationals to consider. Additionally, it is impossible for any method to fairly or reasonable account for things like the shadow effect, boogeymen and biking. Again, I am not saying these rankings are somehow objectively correct, just that I believe they accurately reflect which players had the best buzzes at this tournament.

Finally, on the topic of buzz positions, those were excluded intentionally. For pure ranking purposes, I was fairly confident that question outcome carries the large majority of the signal. This belief is significantly reinforced by inspection of the rankings produced by this buzzpoint-blind model. We could test this more rigorously by producing a position-aware model and comparing the resulting rankings to these, then comparing rank correlations. However, I believe the far more interesting use for buzzpoints would be in answering a number of other questions, such as player profiling, question quality analysis and counterfactuals prediction. 👀😉