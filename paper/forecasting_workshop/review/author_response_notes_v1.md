# Author Response Notes v1

These are short author-side response notes for the strongest expected reviewer attacks. They are intentionally brief and non-defensive.

## Ranked Responses

### 1. Forecasting paper or finance execution paper?

**Response note:** The paper does not claim a better portfolio policy or a new RL method. Its central question is how forecasting outputs should be evaluated once frictions separate proposed targets from realized actions. Portfolio decisions are the concrete case study, not the contribution category.

### 2. Why is executed-path evaluation primary?

**Response note:** The paper does not argue this by preference alone. The target-versus-executed comparison shows that positive-cost conclusions can differ materially across the two views, while the accounting diagnostic explains why that mismatch arises. The paper therefore treats target-level quantities as diagnostic and executed-path quantities as primary.

### 3. Is the same-forecast support strong enough?

**Response note:** The paper uses conservative wording on purpose. It does not claim exact forecast identity; it claims only that forecast-side movement is small relative to decision-side movement in the compared arms. This is supporting evidence, not the main identification result.

### 4. Why is this not just a smoothing effect?

**Response note:** Reduced turnover is part of the mechanism, not a contradiction. The paper's point is that once frictions intervene, the forecast-to-execution interface changes realized decision quality through realized trading and realized cost. The zero-cost near-flat row and the target-versus-executed disagreement help separate this from a generic "better predictor" story.

### 5. The case study is too narrow.

**Response note:** Agreed; the paper is intentionally narrow. The claim is a case-study evaluation argument, not a universality claim. The contribution is to isolate one forecasting-to-decision evaluation issue cleanly rather than to establish broad empirical generality.

### 6. Frozen-policy evidence may be incomplete.

**Response note:** Also agreed. The frozen-policy setup is chosen to isolate the interface effect while holding predictive information fixed. The paper does not claim that end-to-end retraining would produce the same magnitude or ranking.

### 7. Dense-friction sounds like internal rebuild machinery.

**Response note:** This can be cleaned up. In the submission-facing version, dense friction should be described only as appendix sensitivity support that moves in the same qualitative direction as the main result. Provenance details should remain factual but minimal.

### 8. The auxiliary comparator muddies the story.

**Response note:** That is why it should stay auxiliary and appendix-only in emphasis. It is there to show that the implementation-side direction is not unique to one package, not to function as a second main result.

### 9. Why should forecasting readers care about net Sharpe?

**Response note:** Net Sharpe is the case-study metric, but the paper's logic is not about that metric itself. The general point is that once a decision layer and frictions intervene, evaluation must follow realized actions rather than proposed targets alone.

### 10. Internal process language weakens credibility.

**Response note:** Agreed. This is a presentational issue, not a scientific one, and should be cleaned before submission. Submission-facing text should avoid audit/build jargon unless strictly necessary.

## Response Discipline

- Do not answer by widening the claim.
- Do not answer by pretending the same-forecast support is stronger than it is.
- Do not answer by promoting dense-friction or CC-TA-LBIP above supporting status.
- Keep answers short enough that they sound like reviewer responses, not excuses.
