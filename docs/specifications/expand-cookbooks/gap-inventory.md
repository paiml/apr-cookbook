# Gap Inventory

Per-capability gap between aprender 0.31.0..0.31.2 (Unreleased) and apr-cookbook v6.0.0 coverage. Sourced from `~/src/aprender/CHANGELOG.md` Unreleased + 0.31.0/0.31.1 sections, cross-checked against `examples/cli/` (37 files) and `examples/{mcp,serve,analysis,...}` directories.

Ranked by user-facing impact within each tier.

---

## Tier 1 — High-value, completely uncovered

### 1.1 GPU/CPU oracle bisection (the silent-GPU-gibberish canary)

**aprender ships**:
- `apr trace --save-tensor <stages>` — per-stage forward-pass tensor dumps in APRT byte format (`apr-cli-trace-save-tensor-v1.yaml` v1.4.0 FUNCTIONAL, FALSIFY-009/010/011)
- `apr diff --values` — recognizes APRT stage tensors (closes trace→diff loop)
- `apr-cpu-vs-gpu-output-parity-v1.yaml` v1.5.0 ACTIVE — 5/5 falsifiers DISCHARGED
- CUDA fallback log prefix `[apr-cpu-vs-gpu-output-parity-v1] CUDA path rejected`
- wgpu fallback log prefix + inline cosine parity gate on embedding stage; below threshold = fail closed → CPU fallback
- HF FP16 oracle bisection script (`scripts/ship-007-layer0-oracle/`) pinpointing layer-0 attn_out divergence

**Cookbook covers**: nothing. The most important new safety feature in aprender 0.31.x has zero recipe coverage.

**Impact**: a user investigating "why does my GPU model output gibberish" finds nothing in the cookbook. They have to read aprender's source.

**Recipes needed**: 3 (see [recipe-catalog.md](recipe-catalog.md) §"cli/" for `cli_trace_save_tensor_layer0`, `cli_diff_values_aprt_stage`; §"analysis/" for `analysis_cpu_vs_gpu_parity_gate`).

---

### 1.2 `apr code` — Claude Code parity

**aprender ships** (`apr-code-parity-v1.yaml` v5.1, 21 rows: 14 SHIPPED / 3 PARTIAL / 4 NONE):

P0 (4): MCP client tool registration, SlashCommand 11→21 variants, hook surface + SessionStart wiring, Task-tool subagent spawn.

P1 (5): custom agents discovery (`.apr/agents/` + `.claude/agents/`), privacy-gated NetworkTool/BrowserTool, skills discovery (`.apr/skills/` + `.claude/skills/`), git worktree isolation, permission-mode lattice.

P2 (2 epic-closing): REPL status-line primitive, managed org policy loader (`/etc/apr-code/CLAUDE.md` with `/etc/claude-code/CLAUDE.md` fallback).

**Cookbook covers**: nothing.

**Impact**: an entire new agentic surface — the headline 0.31.0 feature — invisible from the cookbook.

**Recipes needed**: 7, one per P0/P1 SHIPPED row (P2 status-line + org-policy folded into 1 combined recipe).

---

### 1.3 `apr publish` end-to-end

**aprender ships**:
- `publish-manifest-v1.yaml` v1.1.0 (FALSIFY-PM-001..007, 8 unit tests)
- `apr validate-manifest` (Rust-native, replaces pyyaml helper)
- `apr validate-manifest --live` (ureq-based URL HEAD + content-length match + streaming GET + sha256 — discharges FALSIFY-PM-002-live + FALSIFY-PM-003)
- `apr validate-manifest --artifact model.safetensors` (FALSIFY-PM-007: parses safetensors header JSON, verifies per-tensor dtype matches `manifest.quantization`; weight tensors must match, norm/bias tensors may stay F32)
- `apr-cli-publish-extra-v1.yaml` v1.1.0 + FALSIFY-PUB-EXTRA-008 (Python dependency eliminated from ship path)

**Cookbook covers**: `validate_manifest_*` recipes exist (validate_batch.rs, validate_fix_suggestions.rs, validate_manifest_happy.rs, validate_manifest_live_check.rs) but only the validation half. **No end-to-end `apr publish` workflow** — composing validate → manifest → upload.

**Impact**: SHIP-TWO-001 is the operator-facing recipe (cookbook v5.1.0); the **publisher-facing** workflow is missing.

**Recipes needed**: 3 (publish_manifest_full, publish_safetensors_dtype_canary, publish_parent_chain_termination).

---

### 1.4 `apr serve anthropic` — Claude Messages API drop-in

**aprender ships**:
- `apr-claude-proxy-v1.yaml` (DRAFT, promotes to ENFORCED at M6-α)
- 6 FALSIFY-CLAUDE-PROXY gates: model fallback chain, SSE event sequence, Messages-API request/response, token counting, error mapping, header passthrough

**Cookbook covers**: `serve_grpc_stream.rs` and `serve_rate_limited.rs`. **Nothing** for the Anthropic-API-compatible serving mode.

**Impact**: the headline integration story (drop your `ANTHROPIC_API_KEY=foo` and your sovereign apr serve answers) is not demonstrated.

**Recipes needed**: 1 (serve_anthropic_messages_api). Could expand to 3 if we cover model-fallback, SSE events, and token-counting separately.

---

### 1.5 MCP M5 transports + notifications

**aprender ships**:
- M5 scaffold (`pmcp = "2.3"` behind `pmcp-dispatcher` feature flag, default off)
- `notifications/cancelled` (SIGTERM→SIGKILL on long jobs, FALSIFY-MCP-006)
- `notifications/progress` (for `apr.finetune`, FALSIFY-MCP-PROGRESS-001)
- JSON Schema Draft 7 meta-validation on every tool input schema (FALSIFY-MCP-002 strict)
- Tool schemas codegen from YAML (`crates/aprender-mcp/build.rs` emits `APR_<TOOL>_SCHEMA` constants; FALSIFY-MCP-008)

**Cookbook covers**: `mcp/mcp_stdio_server.rs`, `mcp_client_simulation.rs`, `mcp_tool_discovery.rs` (3 recipes, all stdio + manual tool registration).

**Impact**: SSE transport, WebSocket transport, byte-parity test, and the lifecycle notifications all uncovered.

**Recipes needed**: 4 (mcp_sse_server, mcp_websocket_server, mcp_notification_progress, mcp_byte_parity_pmcp).

---

### 1.6 `apr serve plan hf://` — config-only dry-run

**aprender ships**: `apr serve plan` accepts HuggingFace repo IDs (`hf://org/repo` or bare `org/repo`). Fetches **only ~2KB config.json** — no weight download.

**Cookbook covers**: nothing. `model_canary_deploy/` and `model_ab_testing/` exist but are about traffic management, not pre-flight planning.

**Impact**: enables CI dry-runs without GB downloads — a powerful DevOps feature with no demo.

**Recipes needed**: 1 (serve_plan_hf_dryrun).

---

### 1.7 `apr-cli-distill-train-v1` — distillation training contract

**aprender ships**: `apr-cli-distill-train-v1.yaml` with 9 falsifiers all algorithm-bound at PARTIAL_ALGORITHM_LEVEL. Sweep closes 9/9 with TRAIN-009 explicitly classified BLOCKER_FIXTURE_ABSENT. `hf_pipeline` DistillationLoss tests added for FALSIFY-TRAIN-003/004.

**Cookbook covers**: 5 distillation recipes (`distill_knowledge_transfer`, `distill_layer_matching`, `distill_quantization_aware`, `distill_attention_transfer`, `distill_self_distillation`). None ground against the contract.

**Impact**: the contract is the formal definition of correctness; recipes that don't reference it can drift.

**Recipes needed**: 1 (distill_against_contract_v1) that explicitly cites and tests against the falsifier set. Could expand to 3 if we cover one falsifier-cluster per recipe.

---

### 1.8 Streaming APR→Q4K for ≥4 GiB models (ALB-093 / GH-434)

**aprender ships**: streaming quantize path that doesn't OOM on ≥4 GiB models. Enables training/fine-tuning at model scales the single-pass path can't handle.

**Cookbook covers**: `bundle_apr_quantized_q4` (tiny-model demo). Nothing for ≥4 GiB streaming.

**Impact**: training on real-world model sizes (Qwen2.5-Coder-7B, Qwen3-Coder-30B) is silently failing without this path.

**Recipes needed**: 1 (bundle_streaming_q4k_large_model). Uses synthetic ≥4 GiB tensor for IIUR offline-only.

---

## Tier 2 — Sister crates (the hard ≥3 requirement)

See [subcrate-coverage.md](subcrate-coverage.md) for the per-crate recipe specs.

| Crate | Version | Cookbook recipes today | Required (≥3) |
|-------|---------|------------------------|---------------|
| `aprender-mcp` | 0.31.2 | 0 (cookbook uses apr-cli's MCP only) | 3 |
| `aprender-tsp` | 0.31.2 | 0 | 3 |
| `aprender-shell` | 0.31.2 | 0 | 3 |
| `aprender-monte-carlo` | 0.31.2 | 0 | 3 |
| `aprender-cgp` | 0.31.2 | 0 (overlaps with `acceleration/`, none use the unified profiler) | 3 |
| `aprender-contracts-macros` | 0.31.2 | 0 (cookbook uses runtime YAML validator only) | 3 |
| **Total** | — | **0 / 18 minimum** | **18** |

---

## Tier 3 — Performance work shipped without bench recipe

### 3.1 MoE rayon dispatch (2× speedup, qwen3-moe forward path)

**aprender ships**: `forward_qwen3_moe` parallelized with rayon. Discharges `qwen3-moe-forward-v1` v1.3.0 → v1.4.0 FUNCTIONAL.

**Cookbook covers**: `acceleration_kernel_fusion`, `acceleration_compression_benchmark` exist but no MoE-specific bench.

**Recipes needed**: 1 (acceleration_moe_rayon_dispatch_bench).

### 3.2 APR file mmap in `load_tensor_f32` (12+ min → 192s on 7B)

**aprender ships**: mmap-backed lazy tensor load enables `apr diff --values` on 7B-parameter models.

**Cookbook covers**: `acceleration_mmap_inference` (about loading whole model). No per-tensor diff demo.

**Recipes needed**: 1 (acceleration_mmap_per_tensor_diff_bench).

### 3.3 GGUF Q4_0/Q5_0/Q8_0 import fallback (dequant-requant)

**aprender ships**: `apr import` of GGUF with unsupported quants (Q4_0, Q5_0, Q8_0) falls back to dequant-requant via f32 intermediate. Raw import preserves Q4_K/Q6_K exactly.

**Cookbook covers**: `convert_gguf_to_apr` (assumes happy path). No legacy-format fallback demo.

**Recipes needed**: 1 (conversion_gguf_legacy_quant_fallback).

### 3.4 Qwen3-MoE numerical-parity bundle

**aprender ships**: 4 root-cause fixes for Qwen3-Coder-30B-A3B (Q/K RMSNorm rank-3 reshape, rope_theta default rank-4, chat template emission, traced sync). Multi-domain dogfood (math/geo/translate/code) now correct end-to-end.

**Cookbook covers**: nothing for Qwen3-MoE inference specifically.

**Recipes needed**: 1 (inference_qwen3_moe_numerical_parity_smoke). Uses tiny synthetic MoE tensors for IIUR offline-only.

---

## Tier 4 — Authoring patterns

### 4.1 Algorithm-binding sweep (150+ contracts flipped to PARTIAL_ALGORITHM_LEVEL)

**aprender ships**: a record-breaking contract algorithm-binding sweep. Each binding ties an existing falsifier to a concrete, executable algorithm reference.

**Cookbook covers**: nothing about authoring this kind of contract.

**Recipes needed**: 1 (analysis_contract_algorithm_binding_pattern).

### 4.2 `pv check-parity` — parity-matrix contracts

**aprender ships**: `pv check-parity` SEMANTIC gate for parity-matrix contracts (FALSIFY-CODE-PARITY-001..005). Runs each row's `cross_check_command` with `expected_min_hits` / `expected_max_hits`.

**Cookbook covers**: nothing.

**Recipes needed**: 1 (analysis_pv_check_parity_authoring).

### 4.3 JSON Schema Draft 7 meta-validation

**aprender ships**: meta-validation on every tool input schema in CI (FALSIFY-MCP-002 strict).

**Cookbook covers**: nothing.

**Recipes needed**: 1 (analysis_json_schema_draft7_meta_validation).

---

## Recipe count summary

| Tier | Recipes |
|------|---------|
| Tier 1 (High-value uncovered) | 7 + 7 + 3 + 1 + 4 + 1 + 1 + 1 = **25** |
| Tier 2 (Sister crates ≥3 each) | 18 |
| Tier 3 (Perf work bench recipes) | 4 |
| Tier 4 (Authoring patterns) | 3 |
| **Total** | **50** |
