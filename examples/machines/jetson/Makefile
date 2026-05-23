# machines/jetson/Makefile — Canary pipeline for Jetson Orin Nano
#
# Build runs on INTEL (self-hosted runner), not on jetson.
# intel cross-compiles apr-cli for aarch64, deploys via SSH,
# jetson only executes pre-built binaries.
#
# Usage:
#   make canary          # Phase 1: apr + model tests
#   make canary-e2e      # Phase 2: compiled binary + probador
#   make canary-all      # Full pipeline (both phases)
#   make verify          # Quick Tier 1 only (post-provision)

SHELL          := /bin/bash
.DEFAULT_GOAL  := help

# ── Machines ──────────────────────────────────────────────────
INTEL          := intel
JETSON         := jetson
JETSON_URL     := http://192.168.50.53:8080

# ── Paths ─────────────────────────────────────────────────────
APRENDER_SRC   := /home/noah/src/aprender
CROSS_DIR      := /tmp/cross-jetson
TARGET         := aarch64-unknown-linux-gnu
LINKER         := aarch64-linux-gnu-gcc
MODEL_DIR      := /home/noah/data/models/canary
MODEL          := $(MODEL_DIR)/qwen-1.5b-q4k.apr
CANARY_OUT     := $(CURDIR)/canary
CORRECTNESS    := $(CANARY_OUT)/correctness.yaml

# ── Feature flags for cross-compile ──────────────────────────
APR_FEATURES   := hf-hub,safetensors-compare,inference,zram

# ── Model preparation (one-time, on intel) ───────────────────
.PHONY: canary-import
canary-import: ## Import from HuggingFace -> .apr -> quantize
	ssh $(INTEL) 'apr import hf://Qwen/Qwen2.5-Coder-1.5B-Instruct \
		-o $(MODEL_DIR)/qwen-1.5b.apr'
	ssh $(INTEL) 'apr quantize $(MODEL_DIR)/qwen-1.5b.apr \
		--scheme q4k -o $(MODEL)'

# ── Cross-compile apr-cli (on intel) ─────────────────────────
.PHONY: apr-build
apr-build: ## Cross-compile apr for aarch64
	ssh $(INTEL) 'cd $(APRENDER_SRC) && \
		CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=$(LINKER) \
		CC_aarch64_unknown_linux_gnu=$(LINKER) \
		cargo +nightly build --release \
			--target $(TARGET) \
			--target-dir $(CROSS_DIR) \
			-p apr-cli \
			--no-default-features --features "$(APR_FEATURES)"'

.PHONY: apr-build-cuda
apr-build-cuda: ## Cross-compile with CUDA (after trueno-gpu fix)
	ssh $(INTEL) 'cd $(APRENDER_SRC) && \
		CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=$(LINKER) \
		CC_aarch64_unknown_linux_gnu=$(LINKER) \
		cargo +nightly build --release \
			--target $(TARGET) \
			--target-dir $(CROSS_DIR) \
			-p apr-cli \
			--no-default-features --features "$(APR_FEATURES),cuda"'

# ── Deploy (intel -> jetson) ─────────────────────────────────
.PHONY: apr-deploy
apr-deploy: apr-build ## Deploy apr binary to Jetson
	ssh $(INTEL) 'scp $(CROSS_DIR)/$(TARGET)/release/apr $(JETSON):~/.cargo/bin/'

.PHONY: canary-models
canary-models: ## Sync .apr model to Jetson (copia delta)
	ssh $(INTEL) 'copia sync $(MODEL_DIR)/ $(JETSON):$(MODEL_DIR)/'

# ── Canary tests (on jetson) ─────────────────────────────────
.PHONY: canary-check
canary-check: ## Tier 1-2: hardware probe + kernel correctness
	ssh $(JETSON) 'apr check --json' | tee $(CANARY_OUT)/check.json

.PHONY: canary-qa
canary-qa: ## Tier 3: 10-gate falsifiable QA
	ssh $(JETSON) 'apr qa $(MODEL) --json' | tee $(CANARY_OUT)/qa.json

.PHONY: canary-parity
canary-parity: ## Tier 3: GPU vs CPU equivalence
	ssh $(JETSON) 'apr parity $(MODEL) \
		--prompt "Write a Python fibonacci function" \
		--assert --json' | tee $(CANARY_OUT)/parity.json

.PHONY: canary-bench
canary-bench: ## Tier 5: throughput profiling
	ssh $(JETSON) 'apr bench $(MODEL) \
		--warmup 3 --iterations 10 --json' \
		| tee $(CANARY_OUT)/bench.json

.PHONY: canary-chat
canary-chat: ## Tier 4: end-to-end chat
	ssh $(JETSON) 'apr run $(MODEL) \
		--prompt "Write a Python fibonacci function" \
		--max-tokens 128'

# ── End game: compiled binary + probador verification ────────
.PHONY: canary-compile
canary-compile: ## Compile self-contained binary (model baked in)
	ssh $(INTEL) 'apr compile $(MODEL) \
		--target $(TARGET) --release --strip --lto \
		-o $(CROSS_DIR)/qwen-coder-jetson'
	ssh $(INTEL) 'scp $(CROSS_DIR)/qwen-coder-jetson $(JETSON):~/'

.PHONY: canary-serve
canary-serve: ## Start compiled binary as chat server on Jetson
	ssh $(JETSON) 'pkill -f qwen-coder-jetson || true'
	ssh -f $(JETSON) './qwen-coder-jetson --serve --port 8080'
	@echo "Waiting for server..."
	@until curl -sf $(JETSON_URL)/health > /dev/null 2>&1; do sleep 1; done
	@echo "[ok] Server ready at $(JETSON_URL)"

.PHONY: canary-test
canary-test: ## probador correctness tests against Jetson chat server
	probador llm test \
		--config $(CORRECTNESS) \
		--url $(JETSON_URL) \
		--runtime-name qwen-coder-jetson-aarch64 \
		--output $(CANARY_OUT)/probar-test.json

.PHONY: canary-load
canary-load: ## probador load test (concurrency=2, 30s)
	probador llm load \
		--url $(JETSON_URL) \
		--concurrency 2 \
		--duration 30s \
		--warmup 5s \
		--runtime-name qwen-coder-jetson-aarch64 \
		--output $(CANARY_OUT)/probar-load.json

.PHONY: canary-bench-full
canary-bench-full: ## probador multi-run benchmark with regression gate
	probador llm bench \
		--url $(JETSON_URL) \
		--runs 3 --duration 30s --warmup 5s \
		--concurrency 2 \
		--baseline $(CANARY_OUT)/baselines/latest.json \
		--fail-on-regression 50.0 \
		--runtime-name qwen-coder-jetson-aarch64 \
		--output $(CANARY_OUT)/probar-bench.json

.PHONY: canary-stop
canary-stop: ## Stop Jetson chat server
	ssh $(JETSON) 'pkill -f qwen-coder-jetson || true'

# ── Full pipeline ────────────────────────────────────────────
.PHONY: canary
canary: apr-deploy canary-models canary-check canary-qa canary-parity canary-bench canary-chat ## Phase 1: apr + model tests
	@echo "[ok] Canary phase 1 complete (apr + model file)."
	@jq '.status' $(CANARY_OUT)/check.json
	@jq '.gates_passed' $(CANARY_OUT)/qa.json

.PHONY: canary-e2e
canary-e2e: canary-compile canary-serve canary-test canary-load canary-stop ## Phase 2: compiled binary + probador
	@echo "[ok] End-to-end canary complete."
	@echo "Correctness:"; jq '.summary' $(CANARY_OUT)/probar-test.json
	@echo "Load:"; jq '{rps: .throughput_rps, p50: .latency_p50_ms, tok_s: .tokens_per_sec}' \
		$(CANARY_OUT)/probar-load.json

.PHONY: canary-all
canary-all: canary canary-e2e ## Full pipeline (both phases)

# ── Quick verification (post-provision) ──────────────────────
.PHONY: verify
verify: ## Tier 1 only: hardware probe (post-provision check)
	ssh $(JETSON) 'nvidia-smi && vulkaninfo --summary && apr check --json' \
		| tee $(CANARY_OUT)/verify.json
	@echo "[ok] Jetson hardware verified."

# ── Help ─────────────────────────────────────────────────────
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
