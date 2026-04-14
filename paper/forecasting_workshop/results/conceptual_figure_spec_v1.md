# Conceptual Figure Spec v1

## Figure Title

From Forecasts to Realized Decisions Under Frictions

## Figure Goal

Show, in one panel, that forecasting outputs become proposed targets first and realized actions only after an execution interface and frictions intervene. The figure should make clear that evaluation should attach to the realized path, not only to the proposed target.

## One-Panel Layout

Left-to-right schematic with five main boxes:

1. **Forecast model**
2. **Proposed target**
3. **Execution interface**
4. **Executed decision**
5. **Realized metrics**

Place a small friction marker between `Proposed target` and `Executed decision`, visually attached to the `Execution interface`.

## Box Labels

### Box 1

**Forecast model**

Small subtitle:
`Predictive information`

### Box 2

**Proposed target**

Small subtitle:
`What the system would like to do`

### Box 3

**Execution interface**

Small subtitle:
`How targets become actions under constraints`

### Box 4

**Executed decision**

Small subtitle:
`What is actually implemented`

### Box 5

**Realized metrics**

Small subtitle:
`Turnover, cost, realized decision quality`

## Arrow Labels

- `Forecast model -> Proposed target`
  - label: `forecast output`
- `Proposed target -> Execution interface`
  - label: `desired action`
- `Execution interface -> Executed decision`
  - label: `implemented action`
- `Executed decision -> Realized metrics`
  - label: `realized path`

## Friction Annotation

Place a callout near the execution interface:

`Frictions and implementation constraints can make proposed targets diverge from realized actions.`

## Required Visual Emphasis

- Add a visible contrast between `Proposed target` and `Executed decision`.
- Add a small note under the middle of the figure:
  - `Target != realized action under frictions`
- Add a highlighted evaluation tag attached only to `Realized metrics`:
  - `Primary evaluation point`
- Optionally add a lighter diagnostic tag near `Proposed target`:
  - `Diagnostic only`

## Design Constraints

- Use plain language that a general ML or forecasting reader can understand.
- Avoid portfolio-specific notation such as weights, holdings, or Greek symbols.
- Avoid multi-panel structure.
- Avoid equations.
- Keep the figure distinct from the accounting figure: this is a conceptual framing schematic, not an empirical diagnostic.

## Intended Reader Takeaway

The reader should understand the figure in one glance:

1. Forecasts first become proposed targets.
2. Frictions intervene through an execution interface.
3. Realized actions can differ from targets.
4. Evaluation should therefore follow realized actions and realized metrics.
