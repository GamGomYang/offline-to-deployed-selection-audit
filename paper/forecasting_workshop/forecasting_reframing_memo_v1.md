# Forecasting Reframing Memo v1

## Paper Identity

This is not a paper about learning a better predictor. It is a paper about how forecasting systems should be evaluated when targets and realized actions diverge under frictions. The central question is whether forecasting-driven systems should be judged at the level of proposed targets or at the level of the realized decisions they actually induce. Cost-sensitive portfolio decisions are the example domain, but the workshop-facing contribution is methodological: holding predictive information fixed, the forecast-to-execution interface can change realized decision quality even when forecast-side information is unchanged or only weakly changed.

## Three Reviewer-facing Key Messages

1. The main contribution is an evaluation argument: under frictions, forecast quality and realized decision quality are not the same object, so executed-path evaluation should be primary.
2. The main empirical result shows that changing only the forecast-to-execution interface changes realized outcomes in positive-cost regimes while the zero-cost row remains near-flat, which supports an implementation-side reading rather than a stronger-predictor claim.
3. The supporting same-forecast table strengthens the forecasting-workshop fit by showing that forecast-side movement can stay small while realized decision-side consequences become large.

## Why This Fits a Forecasting Workshop

1. The paper studies how forecasting outputs should be evaluated after they pass through a downstream decision layer, which is a forecasting-systems problem rather than a domain-only finance problem.
2. The paper makes the contrast between forecast quality and realized decision quality explicit, which is directly relevant to forecasting pipelines whose outputs are acted on under operational constraints.
3. The paper uses finance only as a narrow case study to illustrate a broader workshop theme: when forecasts are consumed by decision systems, evaluation should follow realized actions rather than targets alone.

## What Must Not Dominate the Main Narrative

1. The paper must not read as a portfolio RL paper or a new RL-algorithm paper.
2. The paper must not read as a better predictor, stronger alpha, or finance benchmark-dominance paper.
3. The paper must not let dense-friction provenance, comparator robustness detail, or rebuild history overshadow the main forecasting-to-decision evaluation story.

## Internal Drafting Tagline

`Forecast Quality Is Not Yet Realized Decision Quality`
