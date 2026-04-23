# Q2 Pivot Frontmatter v1

## Default Title

`Benchmarking Forecast-Ranking Robustness Under Frictional Deployment`

## Title Options

1. `Benchmarking Forecast-Ranking Robustness Under Frictional Deployment`
2. `Forecast-Ranking Robustness Under Deployment Frictions`
3. `When Forecast Rankings Fail Deployed Selection Under Friction`
4. `Selection-Robust Forecasting Benchmarks Under Deployment Friction`
5. `From Forecast Ranking to Deployed Selection: Benchmarking Under Friction`

## Abstract A (Default)

Forecasting benchmarks often rank systems by forecast-side quality. Under frictional deployment, however, forecasts pass through a fixed interface before actions and utility are realized, so the ranking appropriate for prediction need not be the ranking appropriate for deployed model selection. We therefore study Q2 as the main benchmark question---whether forecast-side rankings remain deployed-selection robust under a fixed frictional interface---while using Q1 only as mechanism support for why divergence can arise. Synthetic provides the exact zero-friction sanity anchor, event micro provides the main forecasting-native evidence, and inventory provides the main operational corroboration. Event-micro and inventory then show recurrent deployed misselection at moderate-to-high friction. Forecast-side metrics may remain valid for prediction, but forecasting systems under frictional deployment interfaces should report deployed-selection robustness, not forecast-side ranking alone.

## Abstract B (Backup)

Deployment-time model selection for forecasting systems is often inherited from forecast-side rankings. But when forecasts are converted into actions through a frictional interface, the forecast-side winner may not be the deployed-best system. We make this deployed-selection question (Q2) the paper's main object and use Q1 only to explain the mechanism by which fixed proposals and realized actions can diverge. Synthetic provides the exact zero-friction sanity anchor; event micro supplies the main forecasting-native Q2 evidence; inventory provides the main operational corroboration. The results show recurrent deployed misselection at moderate-to-high friction while keeping forecast-side metrics in their role as valid prediction measures. Forecasting systems under frictional deployment interfaces should report deployed-selection robustness, not forecast-side ranking alone.

## Intro First Two Paragraphs

Forecasting benchmarks often evaluate systems by forecast-side quality alone. Under deployment, however, forecasts are translated through fixed interfaces into realized actions and utility. The benchmark-design question is therefore not only whether predictions are accurate, but whether forecast-side ranking remains selection-robust once deployment friction intervenes. A benchmark is unreliable for deployment if the forecast-side winner recurrently fails to be the deployed-best system under the fixed interface that is actually used.

We study this issue through two evaluation questions. Q2 is the main benchmark question: under one fixed deployed interface, does ranking systems by forecast quality still identify the system that is best under realized-action evaluation? Q1 is mechanism support only: if a proposed action path is held fixed, can different interfaces produce different realized outcomes? This separation keeps forecast metrics in their proper role as prediction measures while isolating when the deployment-time selection object changes under friction.

## Related-Work Framing Paragraph

Forecasting benchmark work often evaluates forecast quality, calibration, and live predictive performance. Our question is complementary: whether forecast-side ranking remains trustworthy for deployed model selection under a fixed frictional interface. Unlike decision-aware forecasting or control work, we do not propose new training objectives or policies; we study the evaluation object used for model selection after forecasts are mediated by actions and friction.

## Q2-First Contributions

1. Define deployed-selection robustness as a benchmark-design object for forecasting systems under fixed frictional interfaces.
2. Show the Q2 failure mode in which forecast-side winners become recurrently deployed-suboptimal.
3. Separate Q1 mechanism support from Q2 benchmark consequence while keeping forecast-side metrics in their role as valid prediction measures.
