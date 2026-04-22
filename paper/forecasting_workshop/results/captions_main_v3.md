# Main Text Captions v3

## Figure 1. Conceptual Pipeline

Figure 1: Forecasts are converted into proposed target actions and then translated into realized actions through an interface operating under friction, which creates two distinct evaluation questions: whether the same frozen proposal can yield different realized outcomes under different interfaces (Q1), and whether forecast-metric ranking matches deployed operational ranking under a fixed interface (Q2). This separation changes the evaluation object and can therefore alter benchmark interpretation and model-selection stability under frictions.

## Table 1. Q2 Selection Drift

Table 1: Forecast-side vs. deployed-side model selection under a fixed replenishment interface. Winner columns report the most frequent seed-level best model at each friction level. Positive mean deployed gap means the forecast-selected model underperforms the deployed-selected model by that amount. Table 1 shows the operational selection consequence in the required inventory domain; Figure 3 retains the cross-domain ranking-drift pattern in both synthetic and inventory. In inventory, the selection consequence becomes recurrent at moderate-to-high friction.

## Figure 2. Q1 Results

Figure 2: Q1 --- the same proposal can yield different realized outcomes under different interfaces. At zero friction, target-side and realized-side quantities coincide or nearly coincide. Under positive friction, interface-mediated differences emerge. The synthetic benchmark shows the controlled phenomenon cleanly, while the inventory domain shows a threshold pattern: low friction can remain mixed, but sufficiently high friction makes the tempered interface beneficial through reduced adjustment costs. The two panels use different y-axis quantities because the synthetic and inventory Q1 evidence emphasize controlled gap emergence and operational score effects, respectively.

## Figure 3. Q2 Results

Figure 3: Q2 --- forecast ranking need not match deployed ranking under a fixed interface. In both required domains, zero-friction ranking mismatch is low, while positive friction induces repeated ranking flips and lower rank correlation. Ranking drift is interesting, but selection drift is the operational consequence: Figure 3 shows the cross-domain ranking pattern, while Table 1 shows the inventory-side selection consequence.
