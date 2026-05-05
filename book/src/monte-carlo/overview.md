# Monte Carlo — `aprender-monte-carlo` Finance + Business Simulation

Recipes for [`aprender-monte-carlo`](https://crates.io/crates/aprender-monte-carlo) v0.31.2 — Monte Carlo simulation primitives optimized for financial and business forecasting (Geometric Brownian Motion, jump-diffusion, parametric VaR, scenario simulations).

Closes the **≥3 recipes per sister crate** requirement from [expand-cookbooks/subcrate-coverage.md](../../specifications/expand-cookbooks/subcrate-coverage.md).

## Recipes

| # | Recipe | Citation |
|---|--------|----------|
| MC.1 | [`mc_stock_price_simulation_gbm`](https://github.com/paiml/apr-cookbook/blob/main/examples/monte-carlo/mc_stock_price_simulation_gbm.rs) | Black & Scholes (1973). DOI: 10.1086/260062 |
| MC.2 | [`mc_business_revenue_forecast`](https://github.com/paiml/apr-cookbook/blob/main/examples/monte-carlo/mc_business_revenue_forecast.rs) | Savage (2009). The Flaw of Averages. ISBN: 978-0471381976 |
| MC.3 | [`mc_value_at_risk_historical_vs_parametric`](https://github.com/paiml/apr-cookbook/blob/main/examples/monte-carlo/mc_value_at_risk_historical_vs_parametric.rs) | Jorion (2007). Value at Risk (3rd ed). ISBN: 978-0071464956 |

## API surface exercised

- `aprender::monte_carlo::prelude::{MonteCarloEngine, MonteCarloRng, GeometricBrownianMotion, TimeHorizon, VarianceReduction, percentile, VaR}`
- All deterministic via seeded RNG per IIUR
- Asserts known properties: GBM mean ≈ analytical, P50 ≤ P90, |hist_VaR − param_VaR| < ε

## Provenance

Added during PMAT-082 (expand-cookbooks initiative, v6.1.0).
