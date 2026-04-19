# Universe Construction Note

These universe specifications are fixed reproducible snapshots for the generalization support package.

They are **not** intended as survivorship-free historical universes, and they should not be described that way in the workshop paper or in any paper-facing note. Their role is narrower: they provide stable, reusable support settings that let us ask whether the evaluation-object discrepancy is specific to the current U27 composition or whether the same executed-path reading recurs across other fixed baskets.

## What These Specs Are For

The main purpose of these universes is to reduce the `U27-only artifact` concern without changing the identity of the workshop paper.

They exist to test whether:

- the executed-path interpretation remains informative outside the exact current U27
- the target-vs-executed disagreement repeats across fixed support baskets
- the qualitative friction-sensitive pattern is tied only to one ticker composition or recurs across several reproducible snapshots

## What These Specs Are Not For

These specs are not designed to support claims about:

- survivorship-free historical market universes
- benchmark-representative universe construction
- broad cross-market universality
- exhaustive sector or index replication

They are support-only fixed snapshots for workshop-facing robustness checks.

## Universe Roles

- `u27_current.yaml`: exact current main-paper U27 control universe
- `u27_alt_largecap.yaml`: alternative fixed large-cap snapshot with partial overlap allowed
- `u27_sector_balanced.yaml`: more sector-even fixed 27-name support basket
- `u27_random_seed17.yaml`: reproducible seeded random support basket, appendix-only

## Writing Constraint

Paper-facing wording should say that these are fixed reproducible snapshots used to test whether the evaluation-object discrepancy is U27-specific. It should not say that they are survivorship-free historical universes or broad market-representative samples.
