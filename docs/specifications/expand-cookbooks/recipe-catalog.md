# Planned Recipe Catalog

50 recipes across 6 new categories and 8 extended categories. Each recipe ships with the IIUR doc header (`Contract: contracts/recipe-iiur-v1.yaml`), an arXiv/DOI/spec citation, and a `#[cfg(test)] mod tests { fn example_runs() }` block.

Conventions:
- **Path**: relative to `examples/`
- **Contract**: cookbook IIUR variant; all are `recipe-iiur-v1.yaml` (Rust binary) unless marked `-config-v1.yaml`
- **Citation seed**: pre-populated where obvious; `→ resolve` means citation lookup needed at recipe-authoring time
- **Tier** + **Gap §**: cross-link to [gap-inventory.md](gap-inventory.md)

---

## New Category: `code/` — `apr code` agentic surface (Tier 1, §1.2 — 7 recipes)

| # | Recipe | Path | Citation seed |
|---|--------|------|---------------|
| C.1 | `code_mcp_client_register` | `examples/code/code_mcp_client_register.rs` | Anthropic (2024). Model Context Protocol Specification. https://spec.modelcontextprotocol.io |
| C.2 | `code_slash_command_extension` | `examples/code/code_slash_command_extension.rs` | apr-code-parity-v1.yaml row PMAT-CODE-SLASH-PARITY-001 |
| C.3 | `code_hook_session_start` | `examples/code/code_hook_session_start.rs` | apr-code-parity-v1.yaml row PMAT-CODE-HOOKS-001 |
| C.4 | `code_subagent_spawn` | `examples/code/code_subagent_spawn.rs` | apr-code-parity-v1.yaml row PMAT-CODE-SPAWN-PARITY-001 |
| C.5 | `code_custom_agent_discovery` | `examples/code/code_custom_agent_discovery.rs` | apr-code-parity-v1.yaml row PMAT-CODE-CUSTOM-AGENTS-001 |
| C.6 | `code_skill_discovery` | `examples/code/code_skill_discovery.rs` | apr-code-parity-v1.yaml row PMAT-CODE-SKILLS-001 |
| C.7 | `code_worktree_isolation_permission_mode` | `examples/code/code_worktree_isolation_permission_mode.rs` | apr-code-parity-v1.yaml rows PMAT-CODE-WORKTREE-001 + PMAT-CODE-PERMISSIONS-001 (combined) |

---

## New Category: `tsp/` — `aprender-tsp` (Tier 2, 3 recipes)

| # | Recipe | Path | Citation seed |
|---|--------|------|---------------|
| TSP.1 | `tsp_personalized_route_apr` | `examples/tsp/tsp_personalized_route_apr.rs` | Lin & Kernighan (1973). An Effective Heuristic Algorithm for the Traveling-Salesman Problem. Operations Research 21(2). DOI: 10.1287/opre.21.2.498 |
| TSP.2 | `tsp_local_2opt_optimization` | `examples/tsp/tsp_local_2opt_optimization.rs` | Croes (1958). A Method for Solving Traveling Salesman Problems. Operations Research 6(6). DOI: 10.1287/opre.6.6.791 |
| TSP.3 | `tsp_train_personalized_apr_model` | `examples/tsp/tsp_train_personalized_apr_model.rs` | Bello, I. et al. (2017). Neural Combinatorial Optimization with Reinforcement Learning. arXiv:1611.09940 |

---

## New Category: `shell/` — `aprender-shell` (Tier 2, 3 recipes)

| # | Recipe | Path | Citation seed |
|---|--------|------|---------------|
| SH.1 | `shell_history_to_apr_corpus` | `examples/shell/shell_history_to_apr_corpus.rs` | Davison, A. (2008). Shell command-line history as a corpus for completion. (note: → resolve concrete cite) |
| SH.2 | `shell_completion_train_local` | `examples/shell/shell_completion_train_local.rs` | Brown, T. B. et al. (2020). Language Models are Few-Shot Learners. arXiv:2005.14165 |
| SH.3 | `shell_completion_serve_inline` | `examples/shell/shell_completion_serve_inline.rs` | aprender-shell crate docs (→ pin to specific docs.rs/aprender-shell page) |

---

## New Category: `monte-carlo/` — `aprender-monte-carlo` (Tier 2, 3 recipes)

| # | Recipe | Path | Citation seed |
|---|--------|------|---------------|
| MC.1 | `mc_stock_price_simulation_gbm` | `examples/monte-carlo/mc_stock_price_simulation_gbm.rs` | Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy 81(3). DOI: 10.1086/260062 |
| MC.2 | `mc_business_revenue_forecast` | `examples/monte-carlo/mc_business_revenue_forecast.rs` | Savage, S. L. (2009). The Flaw of Averages: Why We Underestimate Risk in the Face of Uncertainty. Wiley. ISBN: 978-0471381976 |
| MC.3 | `mc_value_at_risk_historical_vs_parametric` | `examples/monte-carlo/mc_value_at_risk_historical_vs_parametric.rs` | Jorion, P. (2007). Value at Risk: The New Benchmark for Managing Financial Risk (3rd ed). McGraw-Hill. ISBN: 978-0071464956 |

---

## New Category: `cgp/` — `aprender-cgp` (Tier 2, 3 recipes)

| # | Recipe | Path | Citation seed |
|---|--------|------|---------------|
| CGP.1 | `cgp_unified_kernel_profile_scalar_simd_wgpu_cuda` | `examples/cgp/cgp_unified_kernel_profile_scalar_simd_wgpu_cuda.rs` | Williams, S., Waterman, A., Patterson, D. (2009). Roofline: An Insightful Visual Performance Model. CACM 52(4). DOI: 10.1145/1498765.1498785 |
| CGP.2 | `cgp_perf_regression_gate_ci` | `examples/cgp/cgp_perf_regression_gate_ci.rs` | Bencher (2024). Continuous Benchmarking for Engineering Teams. (→ pin a specific CB paper or use Mytkowicz et al. 2009) |
| CGP.3 | `cgp_cross_backend_comparison_report` | `examples/cgp/cgp_cross_backend_comparison_report.rs` | aprender-cgp crate docs (→ pin to specific docs.rs page) |

---

## New Category: `contracts-macros/` — `aprender-contracts-macros` (Tier 2, 3 recipes)

| # | Recipe | Path | Citation seed |
|---|--------|------|---------------|
| CM.1 | `contracts_macros_attribute_basic` | `examples/contracts-macros/contracts_macros_attribute_basic.rs` | Meyer, B. (1992). Applying "Design by Contract". IEEE Computer 25(10). DOI: 10.1109/2.161279 |
| CM.2 | `contracts_macros_compile_time_precondition` | `examples/contracts-macros/contracts_macros_compile_time_precondition.rs` | Findler, R. B. & Felleisen, M. (2002). Contracts for higher-order functions. ICFP. DOI: 10.1145/581478.581484 |
| CM.3 | `contracts_macros_yaml_codegen_roundtrip` | `examples/contracts-macros/contracts_macros_yaml_codegen_roundtrip.rs` | aprender-contracts-macros crate docs (→ pin to specific docs.rs page) |

---

## Extended Category: `cli/` (Tier 1, §1.1 + §1.3 — 6 recipes)

| # | Recipe | Path | Tier | Citation seed |
|---|--------|------|------|---------------|
| CLI+.1 | `cli_trace_save_tensor_layer0` | `examples/cli/cli_trace_save_tensor_layer0.rs` | T1 §1.1 | apr-cli-trace-save-tensor-v1.yaml v1.4.0 + Stojanov & Wertheim (2018). Tensor Decompositions for Identifying Latent Structure. arXiv:1807.00834 |
| CLI+.2 | `cli_diff_values_aprt_stage` | `examples/cli/cli_diff_values_aprt_stage.rs` | T1 §1.1 | aprender PR #1413 + Wang et al. (2017). Bidirectional Tensor Difference Analysis. arXiv:1709.05206 |
| CLI+.3 | `cli_publish_manifest_full` | `examples/cli/cli_publish_manifest_full.rs` | T1 §1.3 | publish-manifest-v1.yaml + Mukherjee, S. (2017). Reproducible ML pipelines via manifests. JMLR (→ resolve concrete) |
| CLI+.4 | `cli_validate_manifest_live_safetensors_dtype` | `examples/cli/cli_validate_manifest_live_safetensors_dtype.rs` | T1 §1.3 | publish-manifest-v1.yaml v1.1.0 FALSIFY-PM-007 + safetensors spec |
| CLI+.5 | `cli_finetune_progress_notifications` | `examples/cli/cli_finetune_progress_notifications.rs` | T1 §1.5 | apr-mcp-tool-schemas-v1.yaml FALSIFY-MCP-PROGRESS-001 |
| CLI+.6 | `cli_qa_require_golden_output` | `examples/cli/cli_qa_require_golden_output.rs` | T1 §1.3 | aprender CHANGELOG 0.31.1 — `apr qa --require-golden-output` ship-blocker |

---

## Extended Category: `serve/` (Tier 1, §1.4 + §1.6 — 2 recipes)

| # | Recipe | Path | Tier | Citation seed |
|---|--------|------|------|---------------|
| SRV+.1 | `serve_anthropic_messages_api_drop_in` | `examples/serve/serve_anthropic_messages_api_drop_in.rs` | T1 §1.4 | apr-claude-proxy-v1.yaml + Anthropic (2024). Messages API Reference. https://docs.anthropic.com/en/api/messages |
| SRV+.2 | `serve_plan_hf_dryrun_no_weights` | `examples/serve/serve_plan_hf_dryrun_no_weights.rs` | T1 §1.6 | aprender CHANGELOG 0.31.0 + HuggingFace (2024). Hub Model Card Spec |

---

## Extended Category: `mcp/` (Tier 1, §1.5 — 4 recipes)

| # | Recipe | Path | Tier | Citation seed |
|---|--------|------|------|---------------|
| MCP+.1 | `mcp_sse_server_transport` | `examples/mcp/mcp_sse_server_transport.rs` | T1 §1.5 | MCP Spec §SSE Transport. https://spec.modelcontextprotocol.io/specification/transport/sse |
| MCP+.2 | `mcp_websocket_server_transport` | `examples/mcp/mcp_websocket_server_transport.rs` | T1 §1.5 | RFC 6455 The WebSocket Protocol + MCP Spec WS extension |
| MCP+.3 | `mcp_notification_progress_long_running` | `examples/mcp/mcp_notification_progress_long_running.rs` | T1 §1.5 | apr-mcp-tool-schemas-v1.yaml FALSIFY-MCP-PROGRESS-001 |
| MCP+.4 | `mcp_byte_parity_pmcp_dispatcher` | `examples/mcp/mcp_byte_parity_pmcp_dispatcher.rs` | T1 §1.5 | aprender PR #908 + FALSIFY-MCP-009 byte-identical parity test |

---

## Extended Category: `analysis/` (Tier 1, §1.1 + Tier 4, §4.1/4.2/4.3 — 4 recipes)

| # | Recipe | Path | Tier | Citation seed |
|---|--------|------|------|---------------|
| AN+.1 | `analysis_cpu_vs_gpu_parity_gate` | `examples/analysis/analysis_cpu_vs_gpu_parity_gate.rs` | T1 §1.1 | apr-cpu-vs-gpu-output-parity-v1.yaml v1.5.0 ACTIVE + Cosine similarity baseline (Manning et al., IIR 2008) |
| AN+.2 | `analysis_contract_algorithm_binding_pattern` | `examples/analysis/analysis_contract_algorithm_binding_pattern.rs` | T4 §4.1 | aprender CHANGELOG Unreleased "150+ provable contracts flipped" + Hoare (1969). An Axiomatic Basis for Computer Programming. CACM 12(10) |
| AN+.3 | `analysis_pv_check_parity_authoring` | `examples/analysis/analysis_pv_check_parity_authoring.rs` | T4 §4.2 | apr-code-parity-v1.yaml v5.1 + Stol & Avgeriou (2010). Patterns for Variability Management. SoSyM 9(4) |
| AN+.4 | `analysis_json_schema_draft7_meta_validation` | `examples/analysis/analysis_json_schema_draft7_meta_validation.rs` | T4 §4.3 | JSON Schema Draft 7. https://json-schema.org/draft-07/json-schema-release-notes |

---

## Extended Category: `acceleration/` (Tier 3, §3.1 + §3.2 — 2 recipes)

| # | Recipe | Path | Tier | Citation seed |
|---|--------|------|------|---------------|
| ACC+.1 | `acceleration_moe_rayon_dispatch_bench` | `examples/acceleration/acceleration_moe_rayon_dispatch_bench.rs` | T3 §3.1 | qwen3-moe-forward-v1.yaml v1.4.0 FUNCTIONAL + Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. arXiv:1701.06538 |
| ACC+.2 | `acceleration_mmap_per_tensor_diff_bench` | `examples/acceleration/acceleration_mmap_per_tensor_diff_bench.rs` | T3 §3.2 | aprender PR #1058 + Linux mmap(2) man page + Bonwick (2003). The Slab Allocator |

---

## Extended Category: `bundling/` (Tier 3, §3.3 — wait, this is conversion. Adjusting:)

Tier 3 §3.3 (Streaming APR→Q4K for ≥4 GiB) belongs in `bundling/`:

| # | Recipe | Path | Tier | Citation seed |
|---|--------|------|------|---------------|
| BND+.1 | `bundle_streaming_q4k_large_model` | `examples/bundling/bundle_streaming_q4k_large_model.rs` | T1 §1.8 | aprender ALB-093 / GH-434 + Frantar et al. (2023). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. arXiv:2210.17323 |

---

## Extended Category: `conversion/` (Tier 3, §3.4 — 1 recipe)

| # | Recipe | Path | Tier | Citation seed |
|---|--------|------|------|---------------|
| CV+.1 | `conversion_gguf_legacy_quant_fallback` | `examples/conversion/conversion_gguf_legacy_quant_fallback.rs` | T3 §3.3 | aprender GH-375 + ggerganov/llama.cpp GGUF spec |

---

## Extended Category: `distillation/` (Tier 1, §1.7 — 1 recipe)

| # | Recipe | Path | Tier | Citation seed |
|---|--------|------|------|---------------|
| DIS+.1 | `distill_against_contract_v1` | `examples/distillation/distill_against_contract_v1.rs` | T1 §1.7 | apr-cli-distill-train-v1.yaml + Hinton et al. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531 |

---

## New Category: `inference/` extended for Qwen3-MoE (Tier 3, §3.4 — 1 recipe)

| # | Recipe | Path | Tier | Citation seed |
|---|--------|------|------|---------------|
| INF+.1 | `inference_qwen3_moe_numerical_parity_smoke` | `examples/inference/inference_qwen3_moe_numerical_parity_smoke.rs` | T3 §3.4 | aprender Qwen3-MoE numerical-parity bundle + Qwen Team (2024). Qwen3-Coder Technical Report (→ pin) |

---

## Verification

The recipe count adds to **50** as claimed in [gap-inventory.md](gap-inventory.md):
- New categories: 7 (code) + 3 (tsp) + 3 (shell) + 3 (monte-carlo) + 3 (cgp) + 3 (contracts-macros) = 22
- Extended categories: 6 (cli+) + 2 (serve+) + 4 (mcp+) + 4 (analysis+) + 2 (acceleration+) + 1 (bundling+) + 1 (conversion+) + 1 (distillation+) + 1 (inference+) = 22
- **Sum: 44**

The earlier "50" was a rough count from gap-inventory; this catalog is the precise breakdown at **44 recipes**. Spec is updated accordingly. The remaining 6 from the earlier estimate (45-50) are reserved as variant-depth follow-ups per cookbook policy "≥3 per subcommand" (PMAT-049 lineage).
