# GARA-D Bounded Tiny Proxy Eval

Source cluster:

- `analysis/eval_suite_garad_bounded_tiny_20260519/risk_report.md`
- `analysis/eval_suite_garad_bounded_tiny_20260519/summary.json`
- registry rows:
  - `garad_bounded_tiny_shrink_a002_20260519`
  - `garad_bounded_tiny_extend_a002_20260519`
  - `garad_bounded_tiny_extend_a005_20260519`

## Result

- The shrink variant passed package/risk checks and reduced movement slightly.
- The extend variants also passed package/risk checks, but they increased
  movement more strongly and are best kept as contrast cases.

## Boundary decision

- Treat the shrink variant as candidate-only evidence.
- Treat the extend variants as archive/contrast only.
- Do not present any of them as an official submission recommendation without
  new GT or official-eval evidence.
