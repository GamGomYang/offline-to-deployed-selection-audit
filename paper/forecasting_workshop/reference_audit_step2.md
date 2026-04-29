# Step 2 Reference Map and Citation Verification

Date: 2026-04-29

Local finding: no `.bib` file exists in `paper/forecasting_workshop/` or the repository tree. The paper currently uses an inline `thebibliography` block in `paper_forecasting_workshop_v2.tex`. Therefore, "Found in .bib?" is "No" for every candidate below. Existing inline keys are recorded where applicable, but no new citation should be inserted into the paper until a real `.bib` entry is added or the inline bibliography is deliberately updated.

## A. Citation Audit Table

| # | Candidate | Found in .bib? | Citation key | Verified? | Main/Appendix | Action |
|---|---|---|---|---|---|---|
| 1 | ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities | No (.bib absent) | inline: `forecastbench`; TODO: `karger2025forecastbench` | Verified | Main | Already cited inline; add BibTeX before BibTeX-based citation. |
| 2 | LLM-as-a-Prophet: Understanding Predictive Intelligence with Prophet Arena | No (.bib absent) | inline: `prophetarena`; TODO: `yang2026prophetarena` | Verified | Main | Already cited inline; update metadata to ICLR 2026 before BibTeX use. |
| 3 | GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation | No (.bib absent) | inline: `gifteval`; TODO: `aksu2024gifteval` | Verified | Main | Already cited inline; add BibTeX before BibTeX-based citation. |
| 4 | Monash Time Series Forecasting Archive | No (.bib absent) | TODO: `godahewa2021monash` | Verified | Main | Add to TODO BibTeX; do not cite in paper yet. |
| 5 | The M4 Competition: 100,000 time series and 61 forecasting methods | No (.bib absent) | TODO: `makridakis2020m4` | Verified | Main | Add to TODO BibTeX; use only as one-line benchmark lineage if later added. |
| 6 | M5 accuracy competition: Results, findings, and conclusions | No (.bib absent) | TODO: `makridakis2022m5accuracy` | Verified | Main candidate | Add to TODO BibTeX; likely omit from 4-page main text to avoid overload. |
| 7 | A decoder-only foundation model for time-series forecasting | No (.bib absent) | TODO: `das2024timesfm` | Verified | Main | Add to TODO BibTeX; useful for foundation-model evaluation motivation. |
| 8 | Chronos: Learning the Language of Time Series | No (.bib absent) | TODO: `ansari2024chronos` | Verified | Main | Add to TODO BibTeX; useful for foundation-model forecasting context. |
| 9 | Context is Key: A Benchmark for Forecasting with Essential Textual Information | No (.bib absent) | TODO: `williams2025contextkey` | Verified | Main | Add to TODO BibTeX; useful for benchmark-design neighborhood. |
| 10 | Strictly Proper Scoring Rules, Prediction, and Estimation | No (.bib absent) | inline: `gneiting2007`; TODO: `gneiting2007proper` | Verified | Main | Already cited inline; add BibTeX before BibTeX-based citation. |
| 11 | What Is a Good Forecast? | No (.bib absent) | inline: `murphy1993`; TODO: `murphy1993goodforecast` | Verified | Main | Already cited inline; add BibTeX before BibTeX-based citation. |
| 12 | Evaluating Weather Forecasts from a Decision Maker's Perspective | No (.bib absent) | inline: `raeth2025`; TODO: `raeth2025decisionmaker` | Verified via arXiv; preprint | Main | Already cited inline; keep narrow and label as arXiv if used. |
| 13 | Task-based End-to-end Model Learning in Stochastic Optimization | No (.bib absent) | inline: `donti2017task`; TODO: `donti2017taskbased` | Verified | Main candidate | Already cited inline; consider appendix or one compact contrast only. |
| 14 | Smart "Predict, then Optimize" | No (.bib absent) | inline: `elmachtoub2021spo`; TODO: `elmachtoub2022spo` | Verified | Main candidate | Already cited inline; use only as contrast. |
| 15 | Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities | No (.bib absent) | TODO: `mandi2024decisionfocused` | Verified from arXiv/DOI metadata | Main candidate | Add to TODO BibTeX; could replace multiple DFL citations with one survey. |
| 16 | There are no Champions in Supervised Long-Term Time Series Forecasting | No (.bib absent) | TODO: `brigato2026nochampions` | Verified | Main candidate | Add to TODO BibTeX; use only if ranking-robustness framing needs it. |
| 17 | Accounting for Variance in Machine Learning Benchmarks | No (.bib absent) | TODO: `bouthillier2021variance` | Verified | Main or appendix | Add to TODO BibTeX; useful for statistical/seed reporting. |
| 18 | A Meta-Analysis of Overfitting in Machine Learning | No (.bib absent) | TODO: `roelofs2019overfitting` | Verified | Appendix only | Add to TODO BibTeX; appendix reliability discussion only. |
| 19 | The Ladder: A Reliable Leaderboard for Machine Learning Competitions | No (.bib absent) | TODO: `blum2015ladder` | Verified | Appendix only | Add to TODO BibTeX; appendix leaderboard discussion only. |
| 20 | How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments | No (.bib absent) | TODO: `colas2018seeds` | Verified | Appendix only | Add to TODO BibTeX; appendix seed/statistical power only. |
| 21 | On Reporting Robust and Trustworthy Conclusions from Model Comparison Studies Involving Neural Networks and Randomness | No (.bib absent) | TODO: `gundersen2023robust` | Verified | Appendix only | Add to TODO BibTeX; appendix robustness reporting only. |
| 22 | Online Learning with Switching Costs and Other Adaptive Adversaries | No (.bib absent) | TODO: `cesabianchi2013switching` | Verified | Appendix only | Add to TODO BibTeX; footnote/appendix only. |
| 23 | Bandits with Switching Costs: T^(2/3) Regret | No (.bib absent) | TODO: `dekel2014bandits` | Verified | Appendix only | Add to TODO BibTeX; footnote/appendix only. |
| 24 | Sample-Efficient Reinforcement Learning with loglog(T) Switching Cost | No (.bib absent) | TODO: `qiao2022switching` | Verified | Appendix only | Add to TODO BibTeX; appendix only, avoid main RL framing. |
| 25 | Optimal Inventory Policy | No (.bib absent) | TODO: `arrow1951inventory` | Verified | Appendix only | Add to TODO BibTeX; inventory mechanism appendix only. |
| 26 | The Optimality of (S,s) Policies in the Dynamic Inventory Problem | No (.bib absent) | TODO: `scarf1960ss` | Partially verified; verify publisher details before use | Appendix only | Add TODO as VERIFY BEFORE USE. |
| 27 | Optimal Execution of Portfolio Transactions | No (.bib absent) | TODO: `almgren2000execution` | Verified | Appendix only | Add to TODO BibTeX; do not place in main text. |
| 28 | Decision-aware training of spatiotemporal forecasting models to select a top K subset of sites for intervention | No (.bib absent) | TODO: `heuton2025topk` | Verified | Appendix only | Add to TODO BibTeX; extended related work only. |
| 29 | Decision-Focused Retraining of Forecast Models for Optimization Problems in Smart Energy Systems | No (.bib absent) | TODO: `beichter2024retraining` | Verified | Appendix only | Add to TODO BibTeX; extended related work only. |
| 30 | Decision-Focused Fine-Tuning of Time Series Foundation Models for Dispatchable Feeder Optimization | No (.bib absent) | TODO: `beichter2025finetuning` | Verified | Appendix only | Add to TODO BibTeX; extended related work only. |
| 31 | Goal-Oriented Time-Series Forecasting: Foundation Framework Design | No (.bib absent) | TODO: `fechete2026goaloriented` | VERIFY BEFORE USE | Appendix only | Add TODO only; do not cite until official AAAI/OJS metadata is verified. |

## B. Proposed Main-Text Citation Set

Use at most 12 references in the 4-page main text. Recommended set after adding BibTeX entries:

1. Karger et al. 2025, ForecastBench.
2. Yang et al. 2026, Prophet Arena.
3. Aksu et al. 2024, GIFT-Eval.
4. Godahewa et al. 2021, Monash Time Series Forecasting Archive.
5. Makridakis et al. 2020, M4 Competition.
6. Das et al. 2024, TimesFM.
7. Ansari et al. 2024, Chronos.
8. Williams et al. 2025, Context is Key.
9. Gneiting and Raftery 2007, proper scoring rules.
10. Murphy 1993, forecast quality/value.
11. Raeth and Ludwig 2025, decision-maker-perspective forecast evaluation.
12. Mandi et al. 2024, decision-focused learning survey as a single compact contrast.

Omit M5, Donti, Elmachtoub, Brigato, and Bouthillier from the 4-page main text unless space opens. If a predict-then-optimize contrast is needed, prefer one survey citation rather than multiple algorithmic citations.

## C. Proposed Appendix Citation Set

Appendix only:

- Benchmark reliability and seed reporting: Bouthillier et al. 2021; Roelofs et al. 2019; Blum and Hardt 2015; Colas et al. 2018; Gundersen et al. 2023.
- Switching-cost background: Cesa-Bianchi et al. 2013; Dekel et al. 2014; Qiao et al. 2022.
- Inventory/order-cost background: Arrow et al. 1951; Scarf 1960 after publisher verification.
- Transaction-cost/finance background: Almgren and Chriss 2000 only if absolutely necessary; do not place in main text.
- Extended decision-focused forecasting related work: Heuton et al. 2025; Beichter et al. 2024; Beichter et al. 2025; Fechete et al. 2026 only after official verification.

## D. Missing BibTeX TODO List

Every candidate is missing from a `.bib` file because no `.bib` exists locally.

- TODO `karger2025forecastbench`: Ezra Karger, Houtan Bastani, Chen Yueh-Han, Zachary Jacobs, Danny Halawi, Fred Zhang, Philip E. Tetlock. "ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities." ICLR 2025. URL: https://openreview.net/forum?id=lfPkGWXLLf ; arXiv: https://arxiv.org/abs/2409.19839
- TODO `yang2026prophetarena`: Qingchuan Yang, Simon Mahns, Sida Li, Anri Gu, Jibang Wu, Haifeng Xu. "LLM-as-a-Prophet: Understanding Predictive Intelligence with Prophet Arena." ICLR 2026. URL: https://openreview.net/pdf?id=VpiHkMSPqI ; arXiv: https://arxiv.org/abs/2510.17638
- TODO `aksu2024gifteval`: Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, Doyen Sahoo. "GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation." NeurIPS 2024 TSALM Workshop. URL: https://openreview.net/forum?id=Z2cMOOANFX ; arXiv: https://arxiv.org/abs/2410.10393
- TODO `godahewa2021monash`: Rakshitha Godahewa, Christoph Bergmeir, Geoffrey I. Webb, Rob J. Hyndman, Pablo Montero-Manso. "Monash Time Series Forecasting Archive." arXiv 2021 / NeurIPS Datasets and Benchmarks 2021. URL: https://arxiv.org/abs/2105.06643
- TODO `makridakis2020m4`: Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos. "The M4 Competition: 100,000 time series and 61 forecasting methods." International Journal of Forecasting 36(1):54-74, 2020. DOI: https://doi.org/10.1016/j.ijforecast.2019.04.014
- TODO `makridakis2022m5accuracy`: Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos. "M5 accuracy competition: Results, findings, and conclusions." International Journal of Forecasting 38(4):1346-1364, 2022. DOI: https://doi.org/10.1016/j.ijforecast.2021.11.013
- TODO `das2024timesfm`: Abhimanyu Das, Weihao Kong, Rajat Sen, Yichen Zhou. "A decoder-only foundation model for time-series forecasting." ICML 2024. URL: https://openreview.net/forum?id=jn2iTJas6h ; arXiv: https://arxiv.org/abs/2310.10688
- TODO `ansari2024chronos`: Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, Yuyang Wang. "Chronos: Learning the Language of Time Series." TMLR 2024. URL: https://openreview.net/forum?id=gerNCVqqtR ; arXiv: https://arxiv.org/abs/2403.07815
- TODO `williams2025contextkey`: Andrew Robert Williams, Arjun Ashok, Etienne Marcotte, Valentina Zantedeschi, Jithendaraa Subramanian, Roland Riachi, James Requeima, Alexandre Lacoste, Irina Rish, Nicolas Chapados, Alexandre Drouin. "Context is Key: A Benchmark for Forecasting with Essential Textual Information." ICML 2025, PMLR 267:66887-66944. URL: https://proceedings.mlr.press/v267/williams25a.html
- TODO `gneiting2007proper`: Tilmann Gneiting, Adrian E. Raftery. "Strictly Proper Scoring Rules, Prediction, and Estimation." Journal of the American Statistical Association 102(477):359-378, 2007. DOI: https://doi.org/10.1198/016214506000001437
- TODO `murphy1993goodforecast`: Allan H. Murphy. "What Is a Good Forecast? An Essay on the Nature of Goodness in Weather Forecasting." Weather and Forecasting 8(2):281-293, 1993. DOI: https://doi.org/10.1175/1520-0434(1993)008<0281:WIAGFA>2.0.CO;2
- TODO `raeth2025decisionmaker`: Kornelius Raeth, Nicole Ludwig. "Evaluating Weather Forecasts from a Decision Maker's Perspective." arXiv 2025. URL: https://arxiv.org/abs/2512.14779
- TODO `donti2017taskbased`: Priya L. Donti, Brandon Amos, J. Zico Kolter. "Task-based End-to-end Model Learning in Stochastic Optimization." NeurIPS 2017. URL: https://papers.nips.cc/paper/7132-task-based-end-to-end-model-learning-in-stochastic-optimization ; arXiv: https://arxiv.org/abs/1703.04529
- TODO `elmachtoub2022spo`: Adam N. Elmachtoub, Paul Grigas. "Smart 'Predict, then Optimize'." Management Science 68(1):9-26, 2022. DOI: https://doi.org/10.1287/mnsc.2020.3922
- TODO `mandi2024decisionfocused`: Jayanta Mandi, James Kotary, Senne Berden, Maxime Mulamba, Victor Bucarey, Tias Guns, Ferdinando Fioretto. "Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities." Journal of Artificial Intelligence Research 81:1623-1701, 2024. DOI: https://doi.org/10.1613/jair.1.15320 ; arXiv: https://arxiv.org/abs/2307.13565
- TODO `brigato2026nochampions`: Lorenzo Brigato, Rafael Morand, Knut Joar Strommen, Maria Panagiotou, Markus Schmidt, Stavroula Mougiakakou. "There are no Champions in Supervised Long-Term Time Series Forecasting." TMLR 2026. URL: https://openreview.net/forum?id=yO1JuBpTBB ; arXiv: https://arxiv.org/abs/2502.14045
- TODO `bouthillier2021variance`: Xavier Bouthillier, Pierre Delaunay, Mirko Bronzi, Assya Trofimov, Brennan Nichyporuk, Justin Szeto, Naz Sepah, Edward Raff, Kanika Madan, Vikram Voleti, Samira Ebrahimi Kahou, Vincent Michalski, Dmitriy Serdyuk, Tal Arbel, Chris Pal, Gael Varoquaux, Pascal Vincent. "Accounting for Variance in Machine Learning Benchmarks." MLSys 2021. URL: https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf ; arXiv: https://arxiv.org/abs/2103.03098
- TODO `roelofs2019overfitting`: Rebecca Roelofs, Vaishaal Shankar, Benjamin Recht, Sara Fridovich-Keil, Moritz Hardt, John Miller, Ludwig Schmidt. "A Meta-Analysis of Overfitting in Machine Learning." NeurIPS 2019. URL: https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning
- TODO `blum2015ladder`: Avrim Blum, Moritz Hardt. "The Ladder: A Reliable Leaderboard for Machine Learning Competitions." ICML 2015, PMLR 37:1006-1014. URL: https://proceedings.mlr.press/v37/blum15.html ; arXiv: https://arxiv.org/abs/1502.04585
- TODO `colas2018seeds`: Cedric Colas, Olivier Sigaud, Pierre-Yves Oudeyer. "How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments." arXiv 2018. URL: https://arxiv.org/abs/1806.08295
- TODO `gundersen2023robust`: Odd Erik Gundersen, Saeid Shamsaliei, Hakon S. Kjaernli, Helge Langseth. "On Reporting Robust and Trustworthy Conclusions from Model Comparison Studies Involving Neural Networks and Randomness." ACM REP 2023, pages 37-61. DOI: https://doi.org/10.1145/3589806.3600044
- TODO `cesabianchi2013switching`: Nicolo Cesa-Bianchi, Ofer Dekel, Ohad Shamir. "Online Learning with Switching Costs and Other Adaptive Adversaries." NeurIPS 2013. URL: https://papers.nips.cc/paper/5151-online-learning-with-switching-costs-and-other-adaptive-adversaries ; arXiv: https://arxiv.org/abs/1302.4387
- TODO `dekel2014bandits`: Ofer Dekel, Jian Ding, Tomer Koren, Yuval Peres. "Bandits with Switching Costs: T^(2/3) Regret." STOC 2014. URL: https://arxiv.org/abs/1310.2997
- TODO `qiao2022switching`: Dan Qiao, Ming Yin, Ming Min, Yu-Xiang Wang. "Sample-Efficient Reinforcement Learning with loglog(T) Switching Cost." ICML 2022, PMLR 162:18031-18061. URL: https://proceedings.mlr.press/v162/qiao22a.html ; arXiv: https://arxiv.org/abs/2202.06385
- TODO `arrow1951inventory`: Kenneth J. Arrow, Theodore Harris, Jacob Marschak. "Optimal Inventory Policy." Econometrica 19(3):250-272, 1951. URL: https://www.jstor.org/stable/1906813
- TODO VERIFY BEFORE USE `scarf1960ss`: Herbert E. Scarf. "The Optimality of (S,s) Policies in the Dynamic Inventory Problem." In Mathematical Methods in the Social Sciences, 1959, Stanford University Press, 1960, pages 196-202. Verify publisher metadata before citing.
- TODO `almgren2000execution`: Robert Almgren, Neil Chriss. "Optimal Execution of Portfolio Transactions." Journal of Risk 3:5-39, 2000. URL: https://www.risk.net/journal-of-risk/technical-paper/2161150/optimal-execution-portfolio-transactions
- TODO `heuton2025topk`: Kyle Heuton, Frederick Samuel Muench, Shikhar Shrestha, Thomas J. Stopka, Michael C. Hughes. "Decision-aware training of spatiotemporal forecasting models to select a top K subset of sites for intervention." ICML 2025. URL: https://openreview.net/forum?id=8eQKjsVnN3 ; arXiv: https://arxiv.org/abs/2503.05622
- TODO `beichter2024retraining`: Maximilian Beichter, Dorina Werling, Benedikt Heidrich, Kaleb Phipps, Oliver Neumann, Nils Friederich, Ralf Mikut, Veit Hagenmeyer. "Decision-Focused Retraining of Forecast Models for Optimization Problems in Smart Energy Systems." e-Energy 2024, pages 170-181. DOI: https://doi.org/10.1145/3632775.3661952
- TODO `beichter2025finetuning`: Maximilian Beichter, Nils Friederich, Janik Pinter, Dorina Werling, Kaleb Phipps, Sebastian Beichter, Oliver Neumann, Ralf Mikut, Veit Hagenmeyer, Benedikt Heidrich. "Decision-Focused Fine-Tuning of Time Series Foundation Models for Dispatchable Feeder Optimization." Energy and AI 21:100533, 2025. DOI: https://doi.org/10.1016/j.egyai.2025.100533 ; arXiv: https://arxiv.org/abs/2503.01936
- TODO VERIFY BEFORE USE `fechete2026goaloriented`: Luca-Andrei Fechete, Mohamed Sana, Fadhel Ayed, Nicola Piovesan, Wenjie Li, Antonio De Domenico, Tareq Si Salem. "Goal-Oriented Time-Series Forecasting: Foundation Framework Design." Claimed AAAI 2026 / Proceedings of the AAAI Conference on Artificial Intelligence 40(25):21065-21073, DOI 10.1609/aaai.v40i25.39249. Official AAAI/OJS metadata not verified in this pass; do not cite until verified.

## E. Insertion Rule

No unverified citations were inserted into `paper_forecasting_workshop_v2.tex`. Since there is no `.bib` file, new citations should not be added to the paper until the selected entries are added to a real bibliography source or the inline bibliography is intentionally updated.
