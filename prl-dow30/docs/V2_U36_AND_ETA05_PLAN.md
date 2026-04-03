# V2 Pilot Plan: Second Universe + Execution-Aligned Retraining

This note fixes the next two reviewer-facing extensions in the safest order:

1. a second-universe replication that is likely to preserve the execution effect, and
2. an execution-aligned retraining pilot that answers the most direct frozen-policy objection.

## Why These Two

### 1) Second universe

Plain-language purpose:

- If the same execution frontier appears on a different but comparable asset basket, the result is harder to dismiss as a lucky 27-name snapshot.
- This extension is the safer first move because it keeps the original paper question unchanged: same market, same signals, same model family, different fixed universe.

Why this design is favorable:

- It stays in U.S. daily large-cap equities, where the current execution effect is already visible.
- It increases dimension from 27 to 36 without jumping to a high-risk setting such as S\&P 500 full cross-sections.
- The fixed list is broad and liquid enough to make transaction costs and execution smoothing still matter, while avoiding small-cap or thin-history names that would add instability.

Selected universe (`U36 sector-balanced large-cap`):

- ABT, ACN, ADBE, ADP, AEP, APD
- BDX, BLK, BMY
- C, CMCSA, COP, COST
- DE, DHR
- EMR
- FDX
- GILD
- LLY, LOW, LMT
- MA, MDT, MET, MS
- NEE
- ORCL
- PEP, PFE
- RTX
- SBUX, SCHW
- TGT, TMO, TXN
- XOM

Success condition in simple terms:

- We do not need the exact same Sharpe level.
- We only need the same qualitative story:
  - positive-cost interior `eta < 1` exists,
  - executed turnover falls,
  - net Sharpe improves relative to immediate execution.

### 2) Execution-aligned retraining

Plain-language purpose:

- Reviewer question: "If `eta=0.5` is the selected operating point, why not train the policy directly under that execution rule?"
- This pilot answers that question without exploding the experiment matrix.

Why this design is favorable:

- We retrain only the already-selected operating point `eta=0.5`, so the choice is justified before seeing new test results.
- We keep the universe, dates, signals, architecture, and transaction-cost setting unchanged.
- That means any difference is easier to read as "what changes when training is aligned to the execution rule" rather than as a whole new modeling paper.

Success condition in simple terms:

- Best case: retrained `eta=0.5` beats frozen `eta=0.5` and frozen `eta=1.0`.
- Good enough: retrained `eta=0.5` is at least not worse than frozen `eta=0.5`, while still beating immediate execution.
- Even that weaker outcome is useful because it shows the execution effect is not an artifact of one mismatched training setup.

## Recommended Order

1. Run the `U36` frozen pilot first.
2. If the validation frontier still contains a useful interior point, keep it and move on.
3. Then run the `U27 eta=0.5` execution-aligned retraining pilot.
4. Only expand to 5 or 10 seeds after the 3-seed pilot is directionally clean.

## Why This Order Is Safer

- The second-universe pilot is more likely to preserve the current paper story with minimal interpretation risk.
- The retraining pilot is more reviewer-critical, but it is also more likely to change the empirical ranking.
- Running the lower-risk replication first gives us an extra win even if retraining turns out mixed.

## What We Expect If Things Go Well

### U36 pilot

Easy reading:

- `eta=1.0` should still trade the most.
- `eta=0.5` or a nearby interior point should cut turnover materially.
- Under positive `kappa`, that lower turnover should improve net Sharpe even if gross changes little.

### U27 eta=0.5 retraining pilot

Easy reading:

- Training directly with `eta=0.5` should make the policy less surprised by partial execution.
- That can improve the `eta=0.5` operating point itself, or at least stabilize it.
- If that happens, the paper can say:
  - the frozen-policy result identifies the execution effect, and
  - a first retraining check suggests the effect does not disappear once the execution rule is incorporated at training time.

## Pilot Gate Before Full Expansion

Use `seeds = {0,1,2}` first.

### U36 gate

Advance only if:

- the positive-cost validation frontier contains an interior point,
- executed turnover drops by at least about 20\% relative to `eta=1.0`,
- validation net Sharpe is not uniformly worse across the positive-cost kappas.

### U27 eta=0.5 retraining gate

Advance only if:

- no seed collapses,
- retrained `eta=0.5` is not clearly worse than frozen `eta=0.5` on validation,
- retrained `eta=0.5` still improves on `eta=1.0` in the positive-cost regimes.

## Run Commands

From `prl-dow30/`:

```bash
bash scripts/run_paper_u36_sector_frozen_pilot.sh
bash scripts/run_paper_u27_eta05_retrain_pilot.sh
```

Override seeds if needed:

```bash
SEEDS="0 1 2 3 4" bash scripts/run_paper_u36_sector_frozen_pilot.sh
SEEDS="0 1 2 3 4" bash scripts/run_paper_u27_eta05_retrain_pilot.sh
```

## Files Added For These Pilots

### U36 frozen replication

- `configs/exp/paper_u36_sector_snapshot_control.yaml`
- `configs/exp/paper_u36_sector_validation_eta.yaml`
- `configs/exp/paper_u36_sector_final_eta.yaml`
- `scripts/run_paper_u36_sector_frozen_pilot.sh`

### U27 execution-aligned retraining

- `configs/exp/paper_u27_eta05_snapshot_control.yaml`
- `configs/exp/paper_u27_eta05_validation_main_vs_baseline.yaml`
- `configs/exp/paper_u27_eta05_final_main_vs_baseline.yaml`
- `scripts/run_paper_u27_eta05_retrain_pilot.sh`

## How To Read The Outcomes

### If both work

- Best outcome.
- We gain one external replication and one training-alignment answer.

### If only U36 works

- Still valuable.
- The generalization criticism softens, even if retraining remains future work.

### If only retraining works

- Also valuable.
- The strongest frozen-policy criticism softens, even if broad generalization stays limited.

### If both are mixed

- Then we should not force them into the main paper.
- In that case the better move is to keep the v1 paper focused and mention these as exploratory extensions.
