# Reviewer Attack Points v1

This note stress-tests the forecasting workshop submission from the perspective of a skeptical but fair reviewer.

## Top 10 Criticisms Ranked by Risk

| Rank | Risk | Classification | Reviewer Attack |
| --- | --- | --- | --- |
| 1 | High | Must fix before submission | **Is this really a forecasting paper, or is it still a finance execution paper in forecasting clothing?** The core experiment is a portfolio case study, the main metric is net Sharpe, and the paper still risks being read as domain-specific execution work rather than a forecasting-systems evaluation paper. |
| 2 | High | Must fix before submission | **Why must executed-path evaluation be primary?** A skeptical reviewer can say this is partly definitional: you choose executed-path metrics, then conclude executed-path metrics matter. The paper needs the disagreement logic to read as evidence, not merely as a preferred accounting convention. |
| 3 | High | Must fix before submission | **The same-forecast support may be too weak.** The paper says `similar forecasting information`, but only metric-level similarity is shown. Without raw forecast-output identity or near-identity, the reviewer can argue that the support is suggestive but not strong enough for the rhetorical role it currently plays. |
| 4 | High | Must fix before submission | **Why is this not just a smoothing or regularization effect?** A skeptical reader can say the gains may simply come from reducing turnover mechanically, not from any deeper forecast-to-decision evaluation insight. If that alternative explanation is not handled crisply, the paper can feel like a finance implementation trick rather than a forecasting contribution. |
| 5 | Medium | Acceptable limitation | **The case study is narrow.** One frozen split, one fixed 27-name snapshot, and one domain make it hard to know how much the lesson transfers beyond this setting. A reviewer may accept this, but only if the paper stays disciplined about scope. |
| 6 | Medium | Acceptable limitation | **Frozen-policy identification may feel incomplete.** A reviewer can ask whether the result survives end-to-end retraining, or whether the interface effect is only visible because the policy is held fixed. This does not kill the paper if the claim stays narrow, but the limitation is real. |
| 7 | Medium | Must fix before submission | **The dense-friction package still sounds like internal rebuild documentation.** Terms such as `regenerated canonical`, `locked protocol`, and provenance-heavy appendix wording can distract from the science and make the paper feel like a reconstruction report instead of a clean workshop submission. |
| 8 | Medium | Appendix-only issue | **The CC-TA-LBIP comparator may confuse more than it helps.** A reviewer may see it as half-formed external validation: not central enough to persuade, but present enough to dilute the story. If it stays, it must remain visibly auxiliary. |
| 9 | Medium | Acceptable limitation | **Why should a forecasting audience care if the main metric is domain-specific net Sharpe?** Even if the evaluation argument is valid, the reviewer may ask whether the message generalizes beyond portfolio decisions or whether it is too tightly tied to finance performance metrics. |
| 10 | Low | Must fix before submission | **The draft still contains internal process language in places.** Phrases like `audit`, `locked protocol`, `regenerated canonical`, or `workshop bundle` can trigger skepticism even if they are not technically wrong. They make the submission sound like an internal build package. |

## What These Attacks Target

- Attacks 1, 2, 3, and 4 target the paper's core identity and whether the workshop fit is intellectually real.
- Attacks 5, 6, and 9 target scope and transfer, which are limitations rather than fatal flaws if handled honestly.
- Attacks 7, 8, and 10 target narrative discipline and page economy.

## Bottom Line

The paper's likely rejection mode is not "the numbers are wrong." It is more likely "the framing is not yet tight enough to make this feel like a forecasting-paper contribution rather than a finance execution paper with careful caveats."
