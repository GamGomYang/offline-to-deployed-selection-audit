# Paper Forecasting Workshop Outline v1

## Page Budget

1. **Page 1**
   - Title and abstract
   - Introduction problem statement
   - Conceptual figure placement

2. **Page 2**
   - Forecast-to-execution interface section
   - Experimental setup
   - Start of Results with Table 1

3. **Page 3**
   - RL main result interpretation
   - Accounting diagnostic with Figure 2

4. **Page 4**
   - Brief same-forecast supporting mention
   - Brief appendix pointer
   - Discussion and limitations
   - Conclusion

## Section Outline

### 1. Introduction

- Start with the forecasting problem, not the finance application.
- Introduce the target-versus-realized-action gap under frictions.
- State the narrow claim and the workshop contribution.

### 2. Forecast-to-Execution Interface

- Use the conceptual figure to define the paper's object of study.
- Explain why proposed targets and realized actions are different evaluation objects.
- Keep notation light and workshop-facing.

### 3. Experimental Setup

- Describe the frozen-policy, fixed-signal case study.
- Lock the main comparison to `eta=0.5` versus `eta=1.0`.
- State the primary metric as executed-path net Sharpe.

### 4. Results

- Open with the RL selected-point comparison as the only main empirical result.
- Use accounting to justify executed-path evaluation.
- Keep same-forecast as brief conservative support and move the table to the appendix.
- Move dense-friction, comparator, and robustness details to appendix references.

### 5. Discussion and Limitations

- Re-state the narrow implementation-side reading.
- Keep limitations to four compact points.
- Make the forecasting relevance explicit again.

### 6. Conclusion

- End on the evaluation principle, not on the finance case study.

## Appendix Handoff

- Dense-friction diagnostic bundle
- CC-TA-LBIP auxiliary table
- `c`-ablation robustness table
- Target-versus-executed evaluation note
- Forecast similarity audit

## Assembly Guardrails

- The first page must surface the forecasting problem statement clearly.
- Table 1 is the only main empirical result.
- Figure 2 must read as a mechanism explanation, not a second result.
- The same-forecast package must stay conservative and supporting only.
- The main claim must still stand even if every appendix item is removed.
