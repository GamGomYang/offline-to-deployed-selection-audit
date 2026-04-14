# Generalization Decision Rules v1

This memo pre-registers the decision rules for the two highest-priority generalization checks in the forecasting workshop paper:

1. cost-regime sweep
2. multi-split temporal robustness

Its purpose is to keep interpretation stable before results are seen. The paper remains a narrow forecasting-to-decision evaluation paper. These checks can strengthen or weaken confidence in the same implementation-side reading, but they do not authorize broader claims than the claim-freeze document allows.

## Global Guardrails

- Do not claim a universal forecasting theorem.
- Do not claim general benchmark dominance.
- Do not claim a stronger predictive signal.
- Do not promote any generalization check into a second main empirical result.
- The RL selected-point comparison in the documented case study remains the main result unless the whole paper is explicitly re-scoped.
- Executed-path evaluation remains primary; target-level quantities remain diagnostic only.

## Experiment 1: Cost-Regime Sweep

### Core Question

When cost levels are varied more finely, does the documented selected-point interpretation remain aligned with the paper's implementation-side reading?

### Green Outcome

All of the following hold:

- the zero-cost row, or the lowest-cost regime, remains near-flat or clearly weaker than the positive-cost rows
- positive-cost rows show an executed-path advantage for the selected interface in the same qualitative direction as the main result
- the target-versus-executed disagreement remains visible or becomes more pronounced as cost increases
- turnover reduction remains directionally consistent with the implementation-side reading

### Yellow Outcome

One or more of the following hold, without a direct contradiction of the main result:

- the zero-cost row is noisy rather than clearly near-flat
- positive-cost rows are mixed in magnitude but mostly preserve direction
- target-versus-executed disagreement is present but uneven across the sweep
- turnover reduction remains but the executed-path advantage is small or unstable in some intermediate rows

### Red Outcome

Any of the following hold:

- the zero-cost row shows a strong positive effect comparable to or larger than the positive-cost rows
- positive-cost rows frequently lose the executed-path advantage or reverse direction
- target-versus-executed disagreement disappears or flips in a way that undermines the evaluation argument
- the cost sweep mainly suggests a generic monotone smoothing story with no friction-specific interpretation

### Paper-Writing Consequences

#### If Green

- promote the cost sweep to appendix support with a compact reference in the main text or discussion
- allowed wording:
  - `A denser cost sweep preserves the same friction-sensitive direction.`
  - `The implementation-side reading remains aligned with stronger frictions.`
- do not call it a new main result
- do not widen the claim beyond `can materially change realized decision quality under frictions`

#### If Yellow

- keep the cost sweep in the appendix only
- describe it as mixed but directionally compatible support
- allowed wording:
  - `The denser cost sweep is broadly compatible with the main interpretation, but its strength is uneven across cost levels.`
- do not use it to strengthen the abstract or title

#### If Red

- do not use the cost sweep as supporting evidence for the current framing
- remove or weaken any sentence implying that the friction-sensitive direction is robust across cost regimes
- allowed wording:
  - `The denser cost sweep does not consistently preserve the same interpretation, so the workshop paper remains centered on the documented case-study result only.`

## Experiment 2: Multi-Split Temporal Robustness

### Core Question

Across additional temporal splits, does the selected interface preserve the positive-cost direction often enough to support a narrow robustness claim?

### Green Outcome

All of the following hold:

- at least 3 of 4 splits preserve the positive-cost executed-path direction for the selected interface
- turnover reduction is preserved in most splits
- the zero-cost row is flat, near-flat, or mixed/noisy rather than systematically positive
- no split shows a strong contradiction in which the selected interface clearly underperforms across the positive-cost rows

### Yellow Outcome

One or more of the following hold, without collapsing the main case-study reading:

- only 2 of 4 splits preserve the positive-cost direction
- turnover reduction remains common, but executed-path advantage is unstable across splits
- zero-cost rows are mixed and harder to interpret
- one split materially disagrees while the others are compatible

### Red Outcome

Any of the following hold:

- fewer than 2 of 4 splits preserve the positive-cost direction
- turnover reduction is inconsistent or frequently absent
- the zero-cost row is often strongly positive in a way that weakens the friction-specific interpretation
- split-to-split variation is large enough that the current workshop framing would overstate robustness

### Paper-Writing Consequences

#### If Green

- add a brief robustness sentence in the discussion or conclusion
- place the split table or figure in the appendix
- allowed wording:
  - `Across additional temporal splits, the positive-cost direction is preserved in most tested cases.`
  - `The temporal-robustness check supports the same narrow implementation-side reading.`
- do not promote this to a broad domain-general claim

#### If Yellow

- keep the multi-split results in the appendix only
- describe them as mixed robustness evidence
- allowed wording:
  - `Additional temporal splits show partial but not uniform support for the documented case-study pattern.`
- keep the main paper anchored to the original split

#### If Red

- do not present the multi-split experiment as supporting evidence
- explicitly narrow the paper back to the documented split only
- allowed wording:
  - `The additional temporal splits do not support a broader robustness statement, so the paper should be read strictly as a documented case study.`

## Promotion Rules: Main Text vs Appendix

### Main Text

Only the following can enter the main text from these generalization checks:

- at most one short sentence per experiment
- only if the outcome is Green
- only if the sentence sharpens the existing evaluation argument without creating a second empirical centerpiece

### Appendix

The appendix is the default home for:

- cost-sweep figures or tables
- multi-split robustness tables or figures
- mixed-result explanations
- any diagnostic detail needed to interpret Yellow outcomes

### Keep Out

The following should not be promoted anywhere beyond an internal note:

- noisy exploratory results with no stable interpretation
- Red outcomes presented as if they were supportive
- any wording that implies universality from a small number of splits or cost levels

## Allowed Claims by Outcome

### If Both Checks Are Green

- allowed:
  - `The documented implementation-side reading remains compatible with a denser cost sweep and with most tested temporal splits.`
  - `The evidence extends beyond a single cost row and a single split, while remaining a narrow case study.`
- not allowed:
  - `This effect is universal.`
  - `This establishes a general theorem about forecasting systems.`

### If One Green and One Yellow

- allowed:
  - `The broader checks provide partial support for the same interpretation, but robustness remains limited.`
- keep all robustness details in the appendix

### If Both Yellow

- allowed:
  - `The broader checks are mixed, so the paper should remain centered on the documented case-study result.`
- no strengthening of the abstract, title, or conclusion

### If Either Check Is Red

- allowed:
  - `The additional checks do not justify a broader robustness statement.`
- required:
  - lower the paper back to a single-case documented evaluation result
  - remove any language suggesting that the pattern generalizes across regimes or splits

## Claim-Lowering Rules if Results Are Mixed or Negative

### If Cost Sweep Is Yellow or Red

- keep the main claim unchanged but avoid saying the friction-sensitive direction is broadly preserved
- if Red, say only that the main documented case supports the evaluation argument

### If Multi-Split Is Yellow or Red

- keep the paper framed as a documented case study
- if Red, remove any language implying temporal robustness beyond the original split

### If Both Are Red

- lower the paper to:
  - `a narrow documented case study showing that forecast-to-execution interfaces can change realized decision quality under frictions in one fixed setting`
- remove any wording that suggests broader robustness

## Writing Strategy for Mixed Results

If results are mixed, the paper should:

- keep the abstract and title unchanged unless they overstate robustness
- leave the main result and accounting argument intact
- move mixed evidence to the appendix
- describe Yellow outcomes as `partial` or `mixed`
- describe Red outcomes as `not supportive of a broader robustness statement`

## Final Rule

These checks are intended to calibrate confidence, not to manufacture a larger claim. Even under the strongest outcome, the forecasting workshop paper remains a narrow forecasting-to-decision evaluation paper with portfolio decisions as a concrete case study.
