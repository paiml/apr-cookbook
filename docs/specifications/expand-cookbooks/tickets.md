# PMAT Ticket Breakdown

44 recipes decomposed into **12 PMAT tickets** (PMAT-072..083), one per category cluster. Ticket numbers are placeholders; actual numbers assigned by `pmat work add` at execution time.

Each ticket is independently shippable and testable. Tickets within Tier 1 are P1 priority; sister-crate tickets are P2; perf/authoring tickets are P3.

---

## Cargo.toml additions (PMAT-072 prerequisite)

Six new dev-dependencies required across all sister-crate tickets. Land in PMAT-072; subsequent tickets depend on this:

```toml
[dev-dependencies]
# Sister crates (PMAT-072 expand-cookbooks)
aprender-mcp = "0.31.2"
aprender-tsp = "0.31.2"
aprender-shell = "0.31.2"
aprender-monte-carlo = "0.31.2"
aprender-cgp = "0.31.2"
aprender-contracts-macros = "0.31.2"
```

If any subcrate has incompatible features or unavailable on the target platform, gate behind a feature flag in apr-cookbook's `[features]` section.

---

## PMAT-072 — Cargo.toml + new categories scaffolding

**Priority**: P1
**Estimate**: 0.5 day
**Depends on**: spec acceptance

### Scope

1. Add 6 sister-crate dev-dependencies (above).
2. Create 6 new category directories: `examples/{code,tsp,shell,monte-carlo,cgp,contracts-macros}/`.
3. Add empty `[[example]]` placeholders or skip — wire each example as it lands.
4. Update `book/src/SUMMARY.md` to add 6 new top-level sections.
5. Update `docs/specifications/components/recipe-catalog.md` to add 6 new category tables.

### Definition of Done

- `cargo build --lib` clean (new deps resolve)
- `cargo tree | grep aprender-` shows 6 new entries
- mdbook builds with 6 new (placeholder) section headers

---

## PMAT-073 — `apr code` agentic recipes (Tier 1, §1.2)

**Priority**: P1
**Estimate**: 3 days (most complex new surface)
**Depends on**: PMAT-072

### Scope

7 recipes per [recipe-catalog.md](recipe-catalog.md) §"code/":
- `code_mcp_client_register`
- `code_slash_command_extension`
- `code_hook_session_start`
- `code_subagent_spawn`
- `code_custom_agent_discovery`
- `code_skill_discovery`
- `code_worktree_isolation_permission_mode`

Each ships with the IIUR doc header, citation to `apr-code-parity-v1.yaml` row + an upstream paper, `#[cfg(test)] mod tests` block.

### Definition of Done

- All 7 build clean
- All 7 pass `cargo test --example <name>`
- Each demonstrates one row from `apr-code-parity-v1.yaml` v5.1
- Book chapters under `book/src/code/` for each recipe

---

## PMAT-074 — GPU/CPU oracle bisection (Tier 1, §1.1)

**Priority**: P1
**Estimate**: 2 days
**Depends on**: PMAT-072

### Scope

3 recipes:
- `cli/cli_trace_save_tensor_layer0` — capture per-stage tensor dumps in APRT byte format
- `cli/cli_diff_values_aprt_stage` — diff captured tensors element-wise
- `analysis/analysis_cpu_vs_gpu_parity_gate` — assert cosine ≥ threshold; below → fail closed

### Definition of Done

- All 3 build + test green on CPU-only (GPU-path tests `#[cfg_attr(...)] ignore`)
- `apr-cpu-vs-gpu-output-parity-v1.yaml` v1.5.0 ACTIVE referenced in docs
- Recipe walks the SHIP-007 layer-0 oracle bisection workflow end-to-end with a synthetic example

---

## PMAT-075 — `apr publish` end-to-end (Tier 1, §1.3)

**Priority**: P1
**Estimate**: 1.5 days
**Depends on**: PMAT-072

### Scope

3 recipes:
- `cli/cli_publish_manifest_full` — full validate → manifest → upload (dry-run)
- `cli/cli_validate_manifest_live_safetensors_dtype` — FALSIFY-PM-007 SafeTensors dtype canary
- (third recipe folded into validate-manifest-live; keep at 2 if scope tight)

### Definition of Done

- All recipes use `--dry-run` or tempdir destinations (offline-only per IIUR)
- `publish-manifest-v1.yaml` v1.1.0 falsifier set referenced in docs
- The "30.46 GiB F32 fp16-manifest bug" scenario from SHIP-TWO-001 §12.7.2 is exercised in a `#[should_panic]` test

---

## PMAT-076 — `apr serve anthropic` + `apr serve plan hf://` (Tier 1, §1.4 + §1.6)

**Priority**: P1
**Estimate**: 1 day
**Depends on**: PMAT-072

### Scope

2 recipes:
- `serve/serve_anthropic_messages_api_drop_in` — Claude Messages API drop-in demo
- `serve/serve_plan_hf_dryrun_no_weights` — `apr serve plan hf://org/repo` config-only

### Definition of Done

- Anthropic recipe uses local mock or test fixture (not live API)
- HF dry-run recipe stubs the HF client (no real network) per IIUR offline rule
- Both ground against `apr-claude-proxy-v1.yaml` (DRAFT) for the Anthropic recipe

---

## PMAT-077 — MCP M5 transports + notifications (Tier 1, §1.5)

**Priority**: P1
**Estimate**: 1.5 days
**Depends on**: PMAT-072

### Scope

4 recipes:
- `mcp/mcp_sse_server_transport`
- `mcp/mcp_websocket_server_transport`
- `mcp/mcp_notification_progress_long_running`
- `mcp/mcp_byte_parity_pmcp_dispatcher` (FALSIFY-MCP-009)

### Definition of Done

- All 4 use in-memory transports for testability
- Byte-parity test asserts hand-rolled and pmcp produce identical output
- Progress notification test verifies the notification JSON shape

---

## PMAT-078 — `aprender-mcp` embedded recipes (Tier 2)

**Priority**: P2
**Estimate**: 1 day
**Depends on**: PMAT-072, PMAT-077 (shares MCP infrastructure)

### Scope

3 recipes per [subcrate-coverage.md](subcrate-coverage.md) §"aprender-mcp":
- `mcp_embedded_server_minimal`
- `mcp_embedded_register_custom_tool`
- `mcp_embedded_byte_parity_pmcp`

### Definition of Done

- Recipes embed `aprender-mcp` directly (not via `apr-cli`)
- All 3 use `Cursor`/`Vec<u8>` transports for IIUR isolation

---

## PMAT-079 — `aprender-tsp` recipes (Tier 2)

**Priority**: P2
**Estimate**: 1 day
**Depends on**: PMAT-072

### Scope

3 recipes:
- `tsp/tsp_personalized_route_apr`
- `tsp/tsp_local_2opt_optimization`
- `tsp/tsp_train_personalized_apr_model`

### Definition of Done

- Each demonstrates a different aprender-tsp API entry point
- Synthetic graphs/history; no external data files
- Round-trip determinism asserted (same seed → same route)

---

## PMAT-080 — `aprender-shell` recipes (Tier 2)

**Priority**: P2
**Estimate**: 1 day
**Depends on**: PMAT-072

### Scope

3 recipes:
- `shell/shell_history_to_apr_corpus`
- `shell/shell_completion_train_local`
- `shell/shell_completion_serve_inline`

### Definition of Done

- Synthetic history strings inline (no real `.zsh_history` access)
- Trained model size asserted < 10 MB
- Top-k completion contains expected tokens

---

## PMAT-081 — `aprender-monte-carlo` recipes (Tier 2)

**Priority**: P2
**Estimate**: 0.5 day (math-heavy but no external deps)
**Depends on**: PMAT-072

### Scope

3 recipes:
- `monte-carlo/mc_stock_price_simulation_gbm`
- `monte-carlo/mc_business_revenue_forecast`
- `monte-carlo/mc_value_at_risk_historical_vs_parametric`

### Definition of Done

- All deterministic via RecipeContext seed
- Each asserts a known property (GBM mean ≈ analytical, P50 ≤ P90, |hist_VaR - param_VaR| < ε)

---

## PMAT-082 — `aprender-cgp` recipes (Tier 2)

**Priority**: P2
**Estimate**: 1 day (cross-backend complexity)
**Depends on**: PMAT-072

### Scope

3 recipes:
- `cgp/cgp_unified_kernel_profile_scalar_simd_wgpu_cuda`
- `cgp/cgp_perf_regression_gate_ci`
- `cgp/cgp_cross_backend_comparison_report`

### Definition of Done

- Scalar backend always tested; SIMD on x86_64; wgpu/CUDA gated
- Regression gate test asserts threshold sensitivity (0% passes, 6% fails)

---

## PMAT-083 — `aprender-contracts-macros` recipes (Tier 2)

**Priority**: P2
**Estimate**: 0.5 day
**Depends on**: PMAT-072

### Scope

3 recipes:
- `contracts-macros/contracts_macros_attribute_basic`
- `contracts-macros/contracts_macros_compile_time_precondition`
- `contracts-macros/contracts_macros_yaml_codegen_roundtrip`

### Definition of Done

- All 3 use `#[contract]` attribute macro
- One uses `compile_fail` doc-test for compile-time enforcement demo
- Roundtrip recipe asserts YAML → Rust → YAML byte-identical

---

## PMAT-084 — Tier 3 perf bench recipes (lower priority)

**Priority**: P3
**Estimate**: 1 day
**Depends on**: PMAT-072

### Scope

4 recipes (Tier 3):
- `acceleration/acceleration_moe_rayon_dispatch_bench`
- `acceleration/acceleration_mmap_per_tensor_diff_bench`
- `bundling/bundle_streaming_q4k_large_model`
- `conversion/conversion_gguf_legacy_quant_fallback`
- `inference/inference_qwen3_moe_numerical_parity_smoke`

### Definition of Done

- Bench recipes use `criterion` or `std::time::Instant` for timing
- Synthetic inputs sized to demonstrate the speedup without requiring real ≥4 GiB models in CI
- Threshold assertions (e.g., MoE rayon dispatch > 1.5× speedup vs. serial baseline on multi-core)

---

## PMAT-085 — Tier 4 authoring patterns + extended distillation (lower priority)

**Priority**: P3
**Estimate**: 1 day
**Depends on**: PMAT-072

### Scope

4 recipes:
- `analysis/analysis_contract_algorithm_binding_pattern`
- `analysis/analysis_pv_check_parity_authoring`
- `analysis/analysis_json_schema_draft7_meta_validation`
- `distillation/distill_against_contract_v1`

### Definition of Done

- Each authoring-pattern recipe walks the contract-author workflow with a working example
- Distillation recipe explicitly cites and tests against `apr-cli-distill-train-v1.yaml` falsifier set
- `pv check-parity` recipe creates a synthetic parity-matrix contract and runs the gate

---

## PMAT-086 — Spec bump v6.0.0 → v6.1.0 + closeout

**Priority**: P3
**Estimate**: 0.5 day
**Depends on**: PMAT-072..085 all merged

### Scope

1. Bump `docs/specifications/apr-cookbook.md` Version 6.0.0 → 6.1.0
2. Update Tech Stack diagram with the 6 new categories + sister crates
3. Update README hero to reflect 34 categories / ~440 recipes
4. Regenerate recipe table via `scripts/generate-recipe-table.sh --update`
5. Tag `v6.1.0`

### Definition of Done

- README + spec + book reflect expanded scope
- v6.1.0 git tag pushed
- All 12 expand-cookbooks PMAT tickets show `completed` in roadmap

---

## Dependency Graph

```
PMAT-072 (scaffold) ─┬─→ PMAT-073 (apr code, P1)
                     ├─→ PMAT-074 (GPU/CPU bisection, P1)
                     ├─→ PMAT-075 (apr publish, P1)
                     ├─→ PMAT-076 (apr serve anthropic + plan hf://, P1)
                     ├─→ PMAT-077 (MCP M5, P1) ──→ PMAT-078 (aprender-mcp, P2)
                     ├─→ PMAT-079 (aprender-tsp, P2)
                     ├─→ PMAT-080 (aprender-shell, P2)
                     ├─→ PMAT-081 (aprender-monte-carlo, P2)
                     ├─→ PMAT-082 (aprender-cgp, P2)
                     ├─→ PMAT-083 (aprender-contracts-macros, P2)
                     ├─→ PMAT-084 (perf benches, P3)
                     └─→ PMAT-085 (authoring + distill, P3)
                                   ↓
                          PMAT-086 (spec bump v6.1.0, P3)
```

PMAT-073..083 can be parallelized after PMAT-072 lands. PMAT-084..085 in parallel with the P1/P2 work; PMAT-086 closes after all merge.

---

## Backout Plan

Each ticket merges as its own PR; revert any single PR with `git revert <merge-sha>`. The 6 new sister-crate dev-dependencies in PMAT-072 are additive — reverting just removes recipes that depend on them. No destructive operations until PMAT-086 spec bump.
