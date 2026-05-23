# Subcrate Coverage

The hard requirement: **≥3 recipes per published sister crate**. This document fully specifies each recipe — name, path, what it demonstrates, the public API it exercises, the IIUR isolation strategy, and the citation. Recipe authors implement directly from this spec.

All sister crates are at `^0.31.2`. Cargo.toml additions consolidated in [tickets.md](tickets.md) §"Cargo.toml additions".

---

## `aprender-mcp` v0.31.2 — embedded MCP server library

**Crate purpose**: MCP (Model Context Protocol) server library used internally by `apr-cli`'s `apr mcp` subcommand. Published as a standalone crate so external Rust apps can embed an MCP server with the same tool dispatcher logic.

### MCP-EMB.1 — `mcp_embedded_server_minimal`

**Path**: `examples/mcp/mcp_embedded_server_minimal.rs`
**Demonstrates**: smallest possible aprender-mcp embed — register one custom tool (`echo`), spin up the stdio dispatcher, accept one JSON-RPC request, return a response, exit. NOT going through `apr-cli`.
**API surface**: `aprender_mcp::Server::builder().with_tool(...).serve_stdio()`
**IIUR isolation**: dispatcher reads from a `Cursor<&[u8]>` request and writes to a `Vec<u8>` response — no actual stdin/stdout. Test asserts the response shape.
**Citation**: Anthropic (2024). Model Context Protocol Specification. https://spec.modelcontextprotocol.io

### MCP-EMB.2 — `mcp_embedded_register_custom_tool`

**Path**: `examples/mcp/mcp_embedded_register_custom_tool.rs`
**Demonstrates**: register a custom tool (e.g. `square_root`) with full JSON Schema Draft 7 input schema, exercise the tool through the dispatcher, assert schema validation rejects malformed input.
**API surface**: `aprender_mcp::Tool::new("square_root", schema, handler).register(...)`
**IIUR isolation**: same as MCP-EMB.1; in-memory I/O.
**Citation**: JSON Schema Draft 7 + apr-mcp-tool-schemas-v1.yaml FALSIFY-MCP-002 strict.

### MCP-EMB.3 — `mcp_embedded_byte_parity_pmcp`

**Path**: `examples/mcp/mcp_embedded_byte_parity_pmcp.rs`
**Demonstrates**: FALSIFY-MCP-009 byte-identical parity test between the hand-rolled stdio dispatcher and the new pmcp-based delegation (M5 scaffold). Same request → same response bytes.
**API surface**: `aprender_mcp::Server` with both `pmcp-dispatcher` feature on and off.
**IIUR isolation**: in-memory request/response; assert byte-identical equality.
**Citation**: aprender PR #908 (M5 scaffold) + FALSIFY-MCP-009.

---

## `aprender-tsp` v0.31.2 — local TSP optimization with personalized .apr models

**Crate purpose**: Traveling-Salesman-Problem solver that uses a personalized `.apr` model to bias edge selection from user history (e.g., delivery routes, daily commutes). All-local, no network.

### TSP.1 — `tsp_personalized_route_apr`

**Path**: `examples/tsp/tsp_personalized_route_apr.rs`
**Demonstrates**: load a personalized `.apr` model (synthetic, generated inline), feed a 10-city graph with edge weights, get an optimized route biased by the model's learned preferences.
**API surface**: `aprender_tsp::Solver::new(model: BundledModelV2).solve(graph: &Graph) -> Route`
**IIUR isolation**: graph + model both generated inline from `RecipeContext` deterministic RNG; tempfile for the .apr.
**Citation**: Lin & Kernighan (1973). An Effective Heuristic Algorithm for the Traveling-Salesman Problem. Operations Research 21(2). DOI: 10.1287/opre.21.2.498

### TSP.2 — `tsp_local_2opt_optimization`

**Path**: `examples/tsp/tsp_local_2opt_optimization.rs`
**Demonstrates**: run aprender-tsp's 2-opt local search on a 20-city instance without any model — pure classical algorithm. Compare initial random tour cost vs. optimized tour cost.
**API surface**: `aprender_tsp::two_opt::optimize(tour: Tour, graph: &Graph) -> Tour`
**IIUR isolation**: graph generated from RecipeContext seed; assertion: optimized tour cost ≤ initial cost.
**Citation**: Croes (1958). A Method for Solving Traveling Salesman Problems. Operations Research 6(6). DOI: 10.1287/opre.6.6.791

### TSP.3 — `tsp_train_personalized_apr_model`

**Path**: `examples/tsp/tsp_train_personalized_apr_model.rs`
**Demonstrates**: train a personalized `.apr` model from a synthetic history of past routes (e.g., 50 prior tours with edge frequencies). Save as `.apr`; reload; verify deterministic.
**API surface**: `aprender_tsp::Trainer::new().fit(history: &[Tour]).save_apr(path)`
**IIUR isolation**: history + tempdir; assertion: hash(round-trip) == hash(original).
**Citation**: Bello, I. et al. (2017). Neural Combinatorial Optimization with Reinforcement Learning. arXiv:1611.09940

---

## `aprender-shell` v0.31.2 — AI-powered shell completion trained on history

**Crate purpose**: lightweight shell-completion engine that trains a local `.apr` model from the user's shell history (zsh `.zsh_history`, bash `.bash_history`, fish `~/.local/share/fish/fish_history`) and proposes completions inline.

### SH.1 — `shell_history_to_apr_corpus`

**Path**: `examples/shell/shell_history_to_apr_corpus.rs`
**Demonstrates**: parse a synthetic shell history file (zsh format), extract command tokens, produce a training corpus suitable for aprender-shell's tokenizer.
**API surface**: `aprender_shell::history::parse_zsh(content: &str) -> Vec<Command>` + `aprender_shell::corpus::Builder::from_commands(...).build()`
**IIUR isolation**: synthetic history string inline; tempdir for corpus output.
**Citation**: Davison, A. (2008). Shell command-line history as a corpus for completion. (Note: this exact paper may not exist; PMAT-077 ticket should resolve to a concrete cite during recipe authoring.)

### SH.2 — `shell_completion_train_local`

**Path**: `examples/shell/shell_completion_train_local.rs`
**Demonstrates**: train a `.apr` shell-completion model from the corpus produced by SH.1. Show the loss curve, save the model, verify size < 10 MB (lightweight constraint).
**API surface**: `aprender_shell::Trainer::new(config).fit(corpus).save_apr(path)`
**IIUR isolation**: training data + tempdir; assertion: saved file size < 10 MB.
**Citation**: Brown, T. B. et al. (2020). Language Models are Few-Shot Learners. arXiv:2005.14165 (LM training reference; aprender-shell is a small autoregressive LM)

### SH.3 — `shell_completion_serve_inline`

**Path**: `examples/shell/shell_completion_serve_inline.rs`
**Demonstrates**: load the trained model from SH.2, call the inline-completion API with a partial command (`git che`), receive ranked completion candidates (`checkout`, `cherry-pick`, ...).
**API surface**: `aprender_shell::Completer::load_apr(path).complete(prefix: &str, k: usize) -> Vec<Completion>`
**IIUR isolation**: load from tempfile produced inline; assertion: top-k contains expected completions.
**Citation**: aprender-shell crate docs (→ pin to specific docs.rs page during recipe authoring)

---

## `aprender-monte-carlo` v0.31.2 — Monte Carlo simulations for finance/business

**Crate purpose**: Monte Carlo simulation primitives optimized for financial and business forecasting use cases. Geometric Brownian Motion, jump-diffusion, parametric VaR, scenario simulations.

### MC.1 — `mc_stock_price_simulation_gbm`

**Path**: `examples/monte-carlo/mc_stock_price_simulation_gbm.rs`
**Demonstrates**: simulate 1000 paths of stock-price evolution under Geometric Brownian Motion (drift μ, volatility σ, time horizon T). Compute mean terminal price + 95th percentile.
**API surface**: `aprender_monte_carlo::gbm::simulate(s0, mu, sigma, t, paths, seed) -> Array2<f64>`
**IIUR isolation**: deterministic seed from RecipeContext; assertion: terminal mean within tolerance of analytical expectation `s0 * exp(mu * t)`.
**Citation**: Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy 81(3). DOI: 10.1086/260062

### MC.2 — `mc_business_revenue_forecast`

**Path**: `examples/monte-carlo/mc_business_revenue_forecast.rs`
**Demonstrates**: simulate 12-month revenue forecast as a sum of N customer cohorts each with random churn rate, ARPU, and acquisition channel. Output P50/P90 revenue ranges with confidence intervals.
**API surface**: `aprender_monte_carlo::scenario::simulate_cohorts(...)` + `aprender_monte_carlo::stats::percentiles(...)`
**IIUR isolation**: deterministic cohort generation from RecipeContext seed; assertion: P50 ≤ P90, P90 ≤ max-observed.
**Citation**: Savage, S. L. (2009). The Flaw of Averages: Why We Underestimate Risk in the Face of Uncertainty. Wiley. ISBN: 978-0471381976

### MC.3 — `mc_value_at_risk_historical_vs_parametric`

**Path**: `examples/monte-carlo/mc_value_at_risk_historical_vs_parametric.rs`
**Demonstrates**: compute 1-day 99% VaR on a synthetic returns series two ways — (a) historical (5th percentile of empirical returns), (b) parametric (μ ± 2.326σ assuming Normal). Show convergence as paths → ∞.
**API surface**: `aprender_monte_carlo::var::{historical, parametric}` + `aprender_monte_carlo::stats::convergence_test`
**IIUR isolation**: synthetic returns from RecipeContext seed; assertion: |historical_VaR - parametric_VaR| < tolerance for normally-distributed input.
**Citation**: Jorion, P. (2007). Value at Risk: The New Benchmark for Managing Financial Risk (3rd ed). McGraw-Hill. ISBN: 978-0071464956

---

## `aprender-cgp` v0.31.2 — Compute-GPU-Profile unified perf CLI

**Crate purpose**: cross-backend kernel profiler. Run the same kernel through scalar / SIMD / wgpu / CUDA paths, get a unified report with throughput, latency, energy estimate, and roofline-model placement.

### CGP.1 — `cgp_unified_kernel_profile_scalar_simd_wgpu_cuda`

**Path**: `examples/cgp/cgp_unified_kernel_profile_scalar_simd_wgpu_cuda.rs`
**Demonstrates**: profile a vector dot-product across all 4 backends (scalar always; SIMD if x86_64; wgpu if `wgpu` feature; CUDA if `cuda` feature). Output unified report.
**API surface**: `aprender_cgp::Profiler::new().backends(BackendSet::ALL).profile(kernel: K) -> Report`
**IIUR isolation**: kernel + input generated from RecipeContext seed; tempdir for report output. GPU backends `#[cfg_attr(not(any(...)), ignore)]` for tests.
**Citation**: Williams, S., Waterman, A., Patterson, D. (2009). Roofline: An Insightful Visual Performance Model. CACM 52(4). DOI: 10.1145/1498765.1498785

### CGP.2 — `cgp_perf_regression_gate_ci`

**Path**: `examples/cgp/cgp_perf_regression_gate_ci.rs`
**Demonstrates**: run aprender-cgp against a baseline JSON; if current run is >5% slower than baseline on the scalar backend, exit 1. Use as a CI gate.
**API surface**: `aprender_cgp::regression::compare(baseline: Path, current: Report, threshold: 0.05)`
**IIUR isolation**: baseline JSON inline; current Report inline; assertion: 0% drift passes, 6% drift fails.
**Citation**: Mytkowicz, T. et al. (2009). Producing Wrong Data Without Doing Anything Obviously Wrong! ASPLOS. DOI: 10.1145/1508244.1508275

### CGP.3 — `cgp_cross_backend_comparison_report`

**Path**: `examples/cgp/cgp_cross_backend_comparison_report.rs`
**Demonstrates**: render a markdown comparison table from a Report — backend × throughput × latency × energy. Useful for PR descriptions.
**API surface**: `aprender_cgp::report::Report::to_markdown(opts: TableOpts) -> String`
**IIUR isolation**: synthetic Report inline; assertion: rendered markdown matches expected snapshot.
**Citation**: aprender-cgp crate docs (→ pin)

---

## `aprender-contracts-macros` v0.31.2 — `#[contract]` proc-macros

**Crate purpose**: compile-time enforcement of preconditions/postconditions/invariants via attribute macros. The runtime YAML validator (`aprender-contracts`) handles authoring/audit; this crate handles compile-time application.

### CM.1 — `contracts_macros_attribute_basic`

**Path**: `examples/contracts-macros/contracts_macros_attribute_basic.rs`
**Demonstrates**: apply `#[contract(precondition = "x > 0", postcondition = "result > x")]` to a `fn double(x: i32) -> i32`. Show that the macro expands to runtime checks; show negative tests trigger panic.
**API surface**: `#[contract(...)]` attribute on a function
**IIUR isolation**: pure-function tests; deliberate panic in `#[should_panic]` test.
**Citation**: Meyer, B. (1992). Applying "Design by Contract". IEEE Computer 25(10). DOI: 10.1109/2.161279

### CM.2 — `contracts_macros_compile_time_precondition`

**Path**: `examples/contracts-macros/contracts_macros_compile_time_precondition.rs`
**Demonstrates**: use `#[contract]` with const-time preconditions (e.g., `const_assert!(N > 0)`) on a generic function. Show that violating the constraint at the call site is a compile error (in a `compile_fail` doc-test).
**API surface**: `#[contract(const_precondition = "N > 0")]` on `fn foo<const N: usize>()`
**IIUR isolation**: pure compile-time check; runtime side is a no-op.
**Citation**: Findler, R. B. & Felleisen, M. (2002). Contracts for higher-order functions. ICFP. DOI: 10.1145/581478.581484

### CM.3 — `contracts_macros_yaml_codegen_roundtrip`

**Path**: `examples/contracts-macros/contracts_macros_yaml_codegen_roundtrip.rs`
**Demonstrates**: generate Rust contract code from a YAML contract file (the bridge from runtime YAML validator to compile-time enforcement). Show YAML → Rust → applied attribute roundtrip with byte-identical re-generation.
**API surface**: `aprender_contracts_macros::codegen::from_yaml(path) -> TokenStream`
**IIUR isolation**: YAML input inline; tempdir for generated `.rs`; assertion: parse(generated) == parse(re-generated).
**Citation**: aprender-contracts-macros crate docs (→ pin)

---

## Coverage matrix

| Crate | Recipes | All API-stable? | All offline-friendly? | All IIUR-clean? |
|-------|---------|-----------------|-----------------------|-----------------|
| `aprender-mcp` | 3 (MCP-EMB.1, .2, .3) | ✅ post-M5 (FALSIFY-MCP-009 byte-parity) | ✅ in-memory I/O | ✅ tempdir + RecipeContext |
| `aprender-tsp` | 3 (TSP.1, .2, .3) | ✅ | ✅ inline graph generation | ✅ tempdir for .apr |
| `aprender-shell` | 3 (SH.1, .2, .3) | ✅ | ✅ synthetic history | ✅ tempdir for corpus + model |
| `aprender-monte-carlo` | 3 (MC.1, .2, .3) | ✅ | ✅ pure RNG simulation | ✅ deterministic seed |
| `aprender-cgp` | 3 (CGP.1, .2, .3) | ⚠️ wgpu/CUDA optional features | ✅ scalar baseline always available | ✅ tempdir for reports |
| `aprender-contracts-macros` | 3 (CM.1, .2, .3) | ✅ proc-macros are stable surface | ✅ compile-time + pure-function tests | ✅ no I/O |

**Total subcrate recipes: 18**, satisfying the ≥3-per-crate requirement with no padding.

## Authoring notes

1. **Citation resolution**: 4 recipes use `→ resolve` / `→ pin` placeholders. Recipe authors verify the citation against docs.rs / arXiv at recipe-authoring time. Acceptable to fall back to `Citation: aprender-<subcrate> crate v0.31.2 docs.rs reference` if no concrete academic cite exists.

2. **Subcrate stability concerns**: `aprender-tsp`, `aprender-shell`, `aprender-monte-carlo` are pre-1.0 and may have API churn. Per [scope.md](scope.md) Risk Register, recipe failure on minor bumps surfaces as a cookbook test fail — that's the intended canary.

3. **GPU-required recipes**: CGP.1 has GPU backends gated behind `cfg`. The CPU-only smoke (scalar backend) is the floor for CI green; full backend matrix is opt-in via local `cargo test --features cuda,wgpu`.
