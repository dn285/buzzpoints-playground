# [name tbd]

[name tbd] is a statistical method used to rank quizbowlers based on per-question performance. It is based on the Plackett-Luce model, an extension of the Bradley-Terry model, which itself generalizes the Elo system. 

## Overview
For each tossup, a player is a “winner” for that tossup if they buzzed correctly in their room, with the other players in the room being the “losers” for that tossup. 

## Methodology
Both dead and negged tossups were dropped from the data. I dropped dead tossups because there is no sensible way to declare a “winner” in that situation. I dropped negged tossups because this model does not explicitly penalize negging and I do not think there is enough signal in a bounceback conversion for it to be meaningful. Either way, the results were quite similar and given we are primarily concerned with top players, they have enough live buzzes where dropping these data points does not diminish the statistical validity of our analysis. 

We fit the filtered data using a Plackett-Luce model. For tossup t with eligible player set S_t, the probability that player i wins is:
P(i wins | S_t) = exp(s_i) / sum_{k in S_t} exp(s_k)
where s_i is player i’s skill parameter. 

The log-likelihood across all tossups is:
L = sum_t [ s_{w_t} - log( sum_{k in S_t} exp(s_k) ) ]
where w_t is the winner of tossup t.

The log-likelihood was maximized with L-BFGS-B via scipy.optimize.minimize and L2 regularization, with lambda set to 0.1 (can cross-validate). 

The same model was then separately for each category, using only tossups in that category. This gives each player both an overall skill estimate, as well as a skill estimate per category. For the downstream analysis, to reduce noise, I only included a player in a category’s model if they had at least 5 correct buzzes in that category. 

## According to this model…
Q: Who are the best players?
A: These people:

Performed the analysis above and ranked players by their skill parameters (both overall and per-category). Note that skill parameters should not be compared across tournaments or subjects, as only the relative difference between players within a given ranking is meaningful. The value of the parameter itself is also somewhat arbitrary, as the model just centers it at the mean (this is why many top players in the subject rankings have “negative” overall skill). 

Q: Which categories are the most specialized?
A: Science and nothing else comes close. 

For each category, compute the variance of (category skill - overall skill) across players. Higher variance indicates a more specialist-driven category. 

Q: Is there any correlation or trade-off between categories?
A: Probably.

Computed pairwise Pearson correlations between category skill vectors across players. This reveals which categories tend to co-occur in players (positive correlation) and which trade off (negative correlation). It is worth noting that this analysis is more accurate than PCA, which in this case is functionally just an approximation of the cross-category correlation matrix. 