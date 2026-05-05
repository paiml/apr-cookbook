# Scope & Charter

## Decision

Expand apr-cookbook from 28 categories (post-centralize-cookbooks v6.0.0) to **34 categories**, adding ~50 new recipes that close the gap between aprender's actual surface and the cookbook's coverage. No repository rename, no spec version bump (this is additive, not a re-foundation).

## Before / After Scope

### Before (apr-cookbook v6.0.0, 2026-05-05)

> The APR Cookbook is the umbrella technical manual for the PAIML sovereign AI stack: model bundling and deployment (`.apr`), data loading (`alimentar`/`aprender-data`), declarative visualization (`presentar`), and infrastructure-as-recipes (`sovereign`/`forjar`).

28 categories. ~388 recipes. Covers the **inherited** surface from sovereign/alimentar/presentar plus the **2026-04-22-and-earlier** apr-cli surface. Missing: everything aprender added in 0.31.0..0.31.2 (Unreleased).

### After (apr-cookbook v6.1.0 post-expand-cookbooks)

> Same charter, expanded to cover aprender 0.31.2's full surface — including agentic patterns (`apr code`), GPU/CPU oracle bisection, MCP M5 transports, Anthropic-API-compatible serving, end-to-end model publishing, and the 6 sister crates (mcp/tsp/shell/monte-carlo/cgp/contracts-macros).

34 categories. ~440 recipes.

## What Migrates From Where

Nothing migrates. This is **net-new** content. No source repository to consolidate; no archive checklist needed.

## New Categories (6)

| Category | Path | Purpose | Recipe count target |
|----------|------|---------|---------------------|
| `code/` | `examples/code/` | `apr code` agentic surface (custom agents, skills, hooks, web tools, worktrees, permissions) | 7 |
| `tsp/` | `examples/tsp/` | `aprender-tsp` local TSP optimization | 3 |
| `shell/` | `examples/shell/` | `aprender-shell` AI-powered shell completion | 3 |
| `monte-carlo/` | `examples/monte-carlo/` | `aprender-monte-carlo` finance/business simulation | 3 |
| `cgp/` | `examples/cgp/` | `aprender-cgp` Compute-GPU-Profile cross-backend perf | 3 |
| `contracts-macros/` | `examples/contracts-macros/` | `aprender-contracts-macros` `#[contract]` proc-macros | 3 |

## Categories Extended (8)

| Category | Existing recipes | Adding |
|----------|------------------|--------|
| `cli/` | 37 | `apr trace --save-tensor`, `apr diff --values` (APRT), `apr publish` end-to-end, `apr validate-manifest --live`, `apr finetune --progress`, `apr qa --require-golden-output` |
| `serve/` | 7 | `apr serve anthropic` (Claude Messages API drop-in), `apr serve plan hf://` (dry-run, no weights) |
| `mcp/` | 3 | SSE transport, WebSocket transport, notifications/cancelled, notifications/progress, byte-parity gate (FALSIFY-MCP-009) |
| `analysis/` | 64 | CPU-vs-GPU output parity gate (`apr-cpu-vs-gpu-output-parity-v1`), wgpu fallback log assertion |
| `acceleration/` | 7 | MoE rayon dispatch bench, APR file mmap per-tensor diff bench |
| `bundling/` | 9 | Streaming APR→Q4K for ≥4 GiB models (ALB-093) |
| `conversion/` | 5 | GGUF Q4_0/Q5_0/Q8_0 import fallback (dequant-requant) |
| `distillation/` | 5 | Distillation training against `apr-cli-distill-train-v1` falsifier set |

## Naming Conventions

- New Rust example file names use snake_case prefix matching the category (e.g., `code_custom_agent.rs`, `tsp_personalized_route.rs`, `mcp_sse_server.rs`).
- Each recipe ships with a `Contract: contracts/recipe-iiur-v1.yaml` (Rust binary) header and an arXiv/DOI/spec citation.
- Subcrate-specific recipes do NOT have a "feature flag" gate — the sister-crate dep is always available because the cookbook is the integration test.

## Charter Boundaries

The expand-cookbooks initiative covers:
- ✅ Every aprender 0.31.x subcommand and feature with a runnable recipe
- ✅ Every published sister crate with ≥3 recipes
- ✅ Provable-contract authoring patterns (algorithm-binding sweep)

The expand-cookbooks initiative does **not** cover:
- ❌ Re-implementing sister-crate logic (recipes call published APIs)
- ❌ Unstable/experimental aprender APIs (`#[doc(hidden)]`, `#[cfg(feature = "experimental")]`)
- ❌ Live-network examples (everything offline-only per cookbook policy)
- ❌ GPU-required CI gates (CPU smoke is the floor)

## Versioning

apr-cookbook spec bumps from v6.0.0 → **v6.1.0** after expand-cookbooks lands (minor: additive, no breaking re-categorization). The expand-cookbooks spec itself stays at v1.0.0 — one-shot.
