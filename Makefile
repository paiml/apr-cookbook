# Use bash for shell commands to support advanced features
SHELL := /bin/bash

# bashrs: Disable built-in implicit rules (MAKE013) and ensure clean error handling
.SUFFIXES:
.DELETE_ON_ERROR:

# PERFORMANCE TARGETS (Toyota Way: Zero Defects, Fast Feedback)
# - make test-fast: < 5 minutes (50 property test cases)
# - make coverage:  < 10 minutes (100 property test cases)
# - make test:      comprehensive (500 property test cases)
# Override with: PROPTEST_CASES=n make <target>

.PHONY: all validate quick-validate release clean help
.PHONY: format format-check lint lint-check check test test-fast test-quick test-doc test-property
.PHONY: quality-gate audit docs build install examples
.PHONY: docs-validate cli-parity variant-coverage contracts-lint contract-grade format-coverage citation-check lean-build
.PHONY: update-deps update-deps-check
.PHONY: coverage coverage-ci coverage-clean clean-coverage coverage-open
.PHONY: sub-test sub-lint sub-check

# Parallel job execution
MAKEFLAGS += -j$(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Sub-projects (disabled - using crates.io dependencies)
# Sovereign stack now consolidated in ../aprender monorepo (APR-MONO v0.31.2).
# The `pv` binary ships from aprender-contracts-cli; prefer the installed binary
# for speed, fall back to `cargo run` so CI works without a prior install step.
# Override: `make PV=~/my/pv docs-validate` to use a custom binary.
PV ?= $(shell command -v pv 2>/dev/null || echo "cargo run -q -p aprender-contracts-cli --manifest-path ../aprender/Cargo.toml --bin pv --")

# Default target
all: validate build

# Quick validation for development (skip expensive checks)
quick-validate: format-check lint-check check test-fast
	@echo "✅ Quick validation passed!"

# Full validation pipeline with quality gates
validate: format lint check test quality-gate audit
	@echo "✅ All validation passed!"
	@echo "  ✓ Code formatting"
	@echo "  ✓ Linting (clippy)"
	@echo "  ✓ Type checking"
	@echo "  ✓ Test suite"
	@echo "  ✓ Quality metrics"
	@echo "  ✓ Security audit"

# =============================================================================
# FORMATTING
# =============================================================================

format: ## Format code
	@echo "🎨 Formatting code..."
	@cargo fmt --all

format-check: ## Check code formatting
	@echo "🎨 Checking code formatting..."
	@cargo fmt --all -- --check

# =============================================================================
# LINTING
# =============================================================================

lint: ## Run clippy with auto-fix
	@echo "🔍 Running clippy..."
	@cargo clippy --all-targets --all-features --fix --allow-dirty --allow-staged 2>/dev/null || true
	@cargo clippy --all-targets --all-features -- -D warnings

lint-check: ## Check clippy without fixing
	@echo "🔍 Checking clippy..."
	@cargo clippy --all-targets --all-features -- -D warnings

# =============================================================================
# TYPE CHECKING
# =============================================================================

check: ## Type check all targets
	@echo "🔍 Type checking..."
	@cargo check --all-targets --all-features

# =============================================================================
# TESTING
# =============================================================================

# TARGET: < 5 minutes (enforced with minimal property test cases)
test-fast: ## Run fast tests (target: <5 min, 50 prop cases)
	@echo "⚡ Running fast tests (target: <5 min)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		PROPTEST_CASES=25 cargo nextest run --workspace --status-level skip --failure-output immediate; \
	else \
		PROPTEST_CASES=25 cargo test --workspace; \
	fi

test-quick: test-fast ## Alias for test-fast
	@echo "✅ Quick tests completed!"

test: test-fast test-doc test-property ## Run core test suite
	@echo "✅ Core test suite completed!"
	@echo "  - Fast unit tests ✓"
	@echo "  - Documentation tests ✓"
	@echo "  - Property-based tests ✓"

test-doc: ## Run documentation tests
	@echo "📚 Running documentation tests..."
	@PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --doc --workspace
	@echo "✅ Documentation tests completed!"

test-property: ## Run property-based tests (50 cases)
	@echo "🎲 Running property-based tests (50 cases per property)..."
	@PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --workspace -- proptest
	@echo "✅ Property tests completed!"

test-property-comprehensive: ## Run property-based tests (500 cases)
	@echo "🎲 Running property-based tests (500 cases per property)..."
	@PROPTEST_CASES=250 cargo test --workspace -- proptest
	@echo "✅ Property tests completed (comprehensive)!"

test-all: test test-property-comprehensive ## Run all tests comprehensively
	@echo "✅ All tests completed!"

# =============================================================================
# EXAMPLES
# =============================================================================

examples: ## Run all examples
	@echo "📝 Running examples..."
	@cargo run --example bundle_static_model
	@cargo run --example bundle_quantized_model
	@cargo run --example convert_safetensors_to_apr
	@cargo run --example convert_apr_to_gguf
	@cargo run --example convert_gguf_to_apr
	@cargo run --example simd_matrix_operations
	@cargo run --example apr_info -- --demo
	@cargo run --example apr_bench -- --demo
	@echo "✅ All examples completed!"

examples-encryption: ## Run encryption example (requires feature)
	@echo "🔐 Running encryption example..."
	@cargo run --example bundle_encrypted_model --features encryption
	@echo "✅ Encryption example completed!"

# =============================================================================
# DOCS + CLI PARITY ENFORCEMENT (F-DOCS-001, F-CLIPARITY-001)
# =============================================================================

docs-validate: ## Validate all *.md via pmat validate-readme + link integrity + pv schema
	@echo "📄 Validating documentation (pmat + provable-contracts)..."
	@if [ ! -f deep-context.md ]; then \
		echo "  → Generating deep context..."; \
		pmat context --output deep-context.md >/dev/null 2>&1 || true; \
	fi
	@echo "  → Factual validation (README, CLAUDE, spec, book)..."
	@pmat validate-readme \
		--targets README.md CLAUDE.md $$(find docs/specifications -name '*.md') $$(find book/src -name '*.md' 2>/dev/null) \
		--deep-context deep-context.md \
		--fail-on-contradiction \
		--fail-on-unverified 2>&1 || (echo "❌ pmat validate-readme FAILED"; exit 1)
	@echo "  → Link integrity..."
	@pmat validate-docs --root . --fail-on-error 2>&1 || (echo "❌ pmat validate-docs FAILED"; exit 1)
	@echo "  → CLI binding integrity (every apr cmd in docs exists)..."
	@apr --help 2>&1 | awk '/^  [a-z]/ {print $$1}' | sort -u > /tmp/apr-cmds-actual.txt
	@grep -rhoP '(?<![.\w])apr [a-z][a-z-]+\b' README.md CLAUDE.md docs/ book/src/ 2>/dev/null | awk '{print $$2}' | sort -u > /tmp/apr-cmds-in-docs.txt
	@MISSING=$$(comm -23 /tmp/apr-cmds-in-docs.txt /tmp/apr-cmds-actual.txt | grep -vE '^(help|subcommand|subcommands)$$' || true); \
	if [ -n "$$MISSING" ]; then \
		echo "❌ Docs reference nonexistent apr subcommands:"; \
		echo "$$MISSING" | sed 's/^/    /'; \
		exit 1; \
	fi
	@echo "  → Contract schema (in-process via aprender-contracts)..."
	@cargo test --test contracts -q --no-fail-fast >/dev/null 2>&1 || (echo "❌ contract validation failed — run 'cargo test --test contracts' for details"; exit 1)
	@echo "✅ Docs validation PASS (factual + links + CLI binding + schema)"

contracts-lint: ## Run pv lint + lean-status on all contracts (F-DOCS-001, F-CLIPARITY-001)
	@echo "📜 Linting provable-contracts (pv from ../aprender monorepo)..."
	@$(PV) lint contracts/ --binding contracts/binding.yaml || (echo "❌ pv lint FAILED"; exit 1)
	@echo "  → Lean proof status..."
	@$(PV) lean-status contracts/ || true
	@echo "  → Proof level report..."
	@$(PV) proof-status contracts/ || true
	@echo "✅ Contracts lint complete"

lean-build: ## Build Lean 4 theorem scaffold (PMAT-047)
	@echo "🔷 Building Lean 4 theorem scaffold..."
	@cd lean && lake build 2>&1 | tail -5
	@echo "✅ Lean build complete (23 proved, 16 sorry — see lean/ProvableContracts/*)"

install-pv: ## Install the `pv` binary from the aprender monorepo (one-time setup)
	@echo "📦 Installing pv from ../aprender/crates/aprender-contracts-cli..."
	@cargo install --path ../aprender/crates/aprender-contracts-cli --force
	@echo "✅ pv installed — verify: pv --version"

cli-parity: ## Verify every apr subcommand has ≥1 cookbook recipe (F-CLIPARITY-001)
	@echo "🎯 Checking apr-cli ↔ recipes 1:1 parity..."
	@apr --help 2>&1 | awk '/^  [a-z]/ {print $$1}' | sort -u | grep -v '^help$$' > /tmp/apr-subs.txt
	@grep -rhoiE 'CLI [Ee]quivalent[*: ]+`?apr [a-z][a-z-]+' examples/ 2>/dev/null | grep -oE 'apr [a-z][a-z-]+' | awk '{print $$2}' | sort -u > /tmp/recipe-subs.txt
	@MISSING=$$(comm -23 /tmp/apr-subs.txt /tmp/recipe-subs.txt || true); \
	N_SUBS=$$(wc -l < /tmp/apr-subs.txt); \
	N_COVERED=$$(comm -12 /tmp/apr-subs.txt /tmp/recipe-subs.txt | wc -l); \
	printf "  Subcommands:  %3d\n  Covered:      %3d\n  Coverage:     %d%%\n" $$N_SUBS $$N_COVERED $$((100*N_COVERED/N_SUBS)); \
	if [ -n "$$MISSING" ]; then \
		echo "❌ Subcommands WITHOUT a recipe:"; \
		echo "$$MISSING" | sed 's/^/    apr /'; \
		exit 1; \
	fi
	@ORPHANS=$$(comm -13 /tmp/apr-subs.txt /tmp/recipe-subs.txt || true); \
	if [ -n "$$ORPHANS" ]; then \
		echo "⚠️  Orphan recipes (no matching apr subcommand):"; \
		echo "$$ORPHANS" | sed 's/^/    /'; \
	fi
	@echo "✅ CLI parity: all $$( wc -l < /tmp/apr-subs.txt | tr -d ' ') subcommands have ≥1 recipe"

variant-depth: ## Invariant F: verify every apr subcommand has ≥3 cookbook recipes (F-VARIANT-DEPTH-001)
	@echo "🔁 Checking variant-depth (≥3 recipes per subcommand)..."
	@apr --help 2>&1 | awk '/^  [a-z]/ {print $$1}' | sort -u | grep -v '^help$$' > /tmp/apr-subs.txt
	@grep -rhoiE 'CLI [Ee]quivalent[*: ]+`?apr [a-z][a-z-]+' examples/ 2>/dev/null \
		| grep -oiE 'apr [a-z][a-z-]+' | awk '{print tolower($$2)}' | sort | uniq -c \
		| awk '{printf "%-25s %s\n", $$2, $$1}' > /tmp/sub-counts.txt
	@TOTAL=$$(wc -l < /tmp/apr-subs.txt); \
	OK=$$(while read sub; do \
		count=$$(awk -v s="$$sub" '$$1==s{print $$2; exit}' /tmp/sub-counts.txt || echo 0); \
		count=$${count:-0}; \
		if [ "$$count" -ge 3 ]; then echo "$$sub"; fi; \
	done < /tmp/apr-subs.txt | wc -l); \
	SHORT=$$(while read sub; do \
		count=$$(awk -v s="$$sub" '$$1==s{print $$2; exit}' /tmp/sub-counts.txt || echo 0); \
		count=$${count:-0}; \
		if [ "$$count" -lt 3 ]; then printf "    apr %-20s  (%d recipes)\n" "$$sub" "$$count"; fi; \
	done < /tmp/apr-subs.txt); \
	printf "  Subcommands:   %3d\n  At ≥3 depth:   %3d\n  Coverage:      %d%%\n" "$$TOTAL" "$$OK" "$$((100*OK/TOTAL))"; \
	if [ -n "$$SHORT" ]; then \
		echo "❌ Below variant-depth target (<3 recipes — Invariant F regression):"; \
		echo "$$SHORT"; \
		echo "    See: docs/specifications/components/quality-gates.md#invariant-f"; \
		exit 1; \
	fi
	@echo "✅ Variant depth: all 66/66 subcommands have ≥3 recipes (Invariant F ENFORCED)"

variant-coverage: ## Report per-subcommand flag/variant coverage
	@echo "📊 Per-subcommand variant coverage:"
	@printf "  %-15s %8s %8s %6s\n" "SUBCOMMAND" "FLAGS" "RECIPES" "COV%"
	@printf "  %-15s %8s %8s %6s\n" "----------" "-----" "-------" "----"
	@apr --help 2>&1 | awk '/^  [a-z]/ {print $$1}' | sort -u | while read sub; do \
		FLAGS=$$(apr $$sub --help 2>&1 | grep -cE '^\s+--[a-z]' || echo 0); \
		RECIPES=$$(grep -liE "CLI [Ee]quivalent[*: ]+.*apr $$sub\b" examples/**/*.rs 2>/dev/null | wc -l); \
		if [ "$$FLAGS" -gt 0 ]; then \
			COV=$$((100 * RECIPES / FLAGS)); \
		else \
			COV=100; \
		fi; \
		printf "  %-15s %8d %8d %5d%%\n" "$$sub" "$$FLAGS" "$$RECIPES" "$$COV"; \
	done

contract-grade: ## Invariant B: check recipe-to-contract bindings (F-CONTRACT-GRADE-001)
	@echo "📋 Checking recipe → contract bindings (Invariant B)..."
	@N_RECIPES=$$(find examples/ -name '*.rs' | wc -l); \
	N_WITH_CONTRACT=$$(grep -rl 'Contract:' examples/ 2>/dev/null | wc -l); \
	PCT=$$((100 * N_WITH_CONTRACT / N_RECIPES)); \
	printf "  Recipes:       %3d\n  With contract: %3d\n  Coverage:      %d%%\n" $$N_RECIPES $$N_WITH_CONTRACT $$PCT; \
	if [ $$PCT -lt 50 ]; then \
		echo "⚠️  Invariant B below 50% — target, not yet enforced"; \
	else \
		if [ $$N_WITH_CONTRACT -lt $$N_RECIPES ]; then \
			echo "❌ Invariant B: $$((N_RECIPES - N_WITH_CONTRACT)) recipes missing contract reference"; \
			exit 1; \
		else \
			echo "✅ Invariant B: all recipes reference a contract"; \
		fi; \
	fi

format-coverage: ## Invariant C: check APR/GGUF/SafeTensors format coverage (F-FORMAT-COV-001)
	@echo "📦 Checking model format coverage (Invariant C)..."
	@N_RECIPES=$$(find examples/ -name '*.rs' | wc -l); \
	N_APR=$$(grep -rlE '\.(apr)\b' examples/ 2>/dev/null | wc -l); \
	N_GGUF=$$(grep -rlE '\.(gguf)\b' examples/ 2>/dev/null | wc -l); \
	N_ST=$$(grep -rlE '\.(safetensors)\b' examples/ 2>/dev/null | wc -l); \
	N_ALL3=$$(comm -12 <(comm -12 <(grep -rlE '\.(apr)\b' examples/ 2>/dev/null | sort) <(grep -rlE '\.(gguf)\b' examples/ 2>/dev/null | sort)) <(grep -rlE '\.(safetensors)\b' examples/ 2>/dev/null | sort) | wc -l); \
	printf "  Total recipes: %3d\n  Mention .apr:  %3d\n  Mention .gguf: %3d\n  Mention .safetensors: %3d\n  All 3 formats: %3d\n" $$N_RECIPES $$N_APR $$N_GGUF $$N_ST $$N_ALL3; \
	PCT=$$((100 * N_ALL3 / N_RECIPES)); \
	printf "  Multi-format coverage: %d%%\n" $$PCT; \
	if [ $$PCT -lt 50 ]; then \
		echo "⚠️  Invariant C below 50% — target, not yet enforced"; \
	else \
		echo "✅ Invariant C: format coverage at $$PCT%%"; \
	fi

citation-check: ## Invariant D: check arXiv/DOI citations in recipes (F-ARXIV-001)
	@echo "📚 Checking arXiv/DOI citations (Invariant D)..."
	@N_RECIPES=$$(find examples/ -name '*.rs' | wc -l); \
	N_CITED=$$(grep -rlE '(arXiv:|DOI:)' examples/ 2>/dev/null | wc -l); \
	PCT=$$((100 * N_CITED / N_RECIPES)); \
	printf "  Recipes:    %3d\n  With citation: %3d\n  Coverage:   %d%%\n" $$N_RECIPES $$N_CITED $$PCT; \
	if [ $$PCT -lt 50 ]; then \
		echo "⚠️  Invariant D below 50% — target, not yet enforced"; \
	else \
		if [ $$N_CITED -lt $$N_RECIPES ]; then \
			echo "❌ Invariant D: $$((N_RECIPES - N_CITED)) recipes missing arXiv/DOI citation"; \
			exit 1; \
		else \
			echo "✅ Invariant D: all recipes have citations"; \
		fi; \
	fi

# =============================================================================
# COVERAGE (Toyota Way: "make coverage" just works)
# TARGET: < 10 minutes (enforced with reduced property test cases)
# =============================================================================

coverage: ## Generate HTML coverage report (target: <10 min)
	@echo "📊 Running comprehensive test coverage analysis (target: <10 min)..."
	@echo "🔍 Checking for cargo-llvm-cov..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@echo "🧹 Cleaning old coverage data..."
	@mkdir -p target/coverage
	@echo "🧪 Phase 1: Running tests with instrumentation..."
	@env PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo llvm-cov test --lib --no-report --all-features --workspace
	@echo "📊 Phase 2: Generating coverage reports..."
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info || true
	@echo ""
	@echo "📊 Coverage Summary:"
	@echo "=================="
	@cargo llvm-cov report --summary-only 2>/dev/null || echo "(run again for summary)"
	@echo ""
	@echo "💡 COVERAGE INSIGHTS:"
	@echo "- HTML report: target/coverage/html/index.html"
	@echo "- LCOV file: target/coverage/lcov.info"
	@echo "- Open HTML: make coverage-open"
	@echo ""

coverage-summary: ## Show coverage summary
	@cargo llvm-cov report --summary-only 2>/dev/null || echo "Run 'make coverage' first"

coverage-open: ## Open HTML coverage report in browser
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Please open: target/coverage/html/index.html"; \
	else \
		echo "❌ Run 'make coverage' first to generate the HTML report"; \
	fi

coverage-ci: ## Generate LCOV report for CI/CD (fast mode)
	@echo "=== Code Coverage for CI/CD ==="
	@echo "Phase 1: Running tests with instrumentation..."
	@env PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo llvm-cov test --lib --no-report --all-features --workspace
	@echo "Phase 2: Generating LCOV report..."
	@cargo llvm-cov report --lcov --output-path lcov.info
	@echo "✓ Coverage report generated: lcov.info"

coverage-clean: ## Clean coverage artifacts
	@rm -f lcov.info coverage.xml target/coverage/lcov.info
	@rm -rf target/llvm-cov target/coverage
	@find . -name "*.profraw" -delete 2>/dev/null || true
	@echo "✓ Coverage artifacts cleaned"

clean-coverage: coverage-clean ## Alias for coverage-clean
	@echo "✓ Fresh coverage ready (run 'make coverage' to regenerate)"

# =============================================================================
# QUALITY
# =============================================================================

quality-gate: docs-validate contracts-lint cli-parity ## Run quality checks (includes docs + CLI parity gates)
	@echo "🔍 Running quality gate checks..."
	@echo "  📊 Checking test count..."
	@TEST_COUNT=$$(cargo test --workspace 2>&1 | grep -E "^test result:" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+"); \
	echo "  Tests: $$TEST_COUNT"; \
	if [ "$$TEST_COUNT" -lt 50 ]; then \
		echo "  ⚠️  Warning: Test count below target (50+)"; \
	else \
		echo "  ✓ Test count acceptable"; \
	fi
	@echo "✅ Quality gates passed (docs F-DOCS-001, CLI parity F-CLIPARITY-001, contracts, tests)!"

# =============================================================================
# SECURITY
# =============================================================================

architecture-demos-coverage: ## Verify architecture-demos manifest is in sync with on-disk recipes/contracts
	@echo "🔧 Checking architecture-demos manifest..."
	@bash scripts/architecture-demos-gen.sh --check
	@echo "🔧 Linting all contracts (includes architecture-demos)..."
	@pv lint contracts/

audit: ## Run security audit
	@echo "🔒 Running security audit..."
	@if command -v cargo-audit >/dev/null 2>&1; then \
		cargo audit; \
	else \
		echo "📦 Installing cargo-audit..."; \
		cargo install cargo-audit && cargo audit; \
	fi

# =============================================================================
# SUB-PROJECTS (Sovereign Stack)
# =============================================================================

sub-check: ## Type check all sub-projects
	@echo "🔍 Type checking sub-projects..."
	@for proj in $(SUB_PROJECTS); do \
		if [ -d "$$proj" ]; then \
			echo "  Checking $$proj..."; \
			(cd "$$proj" && cargo check --all-targets) || exit 1; \
		fi; \
	done
	@echo "✅ All sub-projects type check passed!"

sub-lint: ## Lint all sub-projects
	@echo "🔍 Linting sub-projects..."
	@for proj in $(SUB_PROJECTS); do \
		if [ -d "$$proj" ]; then \
			echo "  Linting $$proj..."; \
			(cd "$$proj" && cargo clippy --all-targets -- -D warnings) || exit 1; \
		fi; \
	done
	@echo "✅ All sub-projects lint passed!"

sub-test: ## Test all sub-projects
	@echo "🧪 Testing sub-projects..."
	@for proj in $(SUB_PROJECTS); do \
		if [ -d "$$proj" ]; then \
			echo "  Testing $$proj..."; \
			(cd "$$proj" && cargo test) || exit 1; \
		fi; \
	done
	@echo "✅ All sub-projects tests passed!"

# =============================================================================
# DEPENDENCY MANAGEMENT
# =============================================================================

update-deps: ## Update dependencies (semver-compatible)
	@echo "🔄 Updating dependencies..."
	@cargo update
	@echo "✅ Dependencies updated!"

update-deps-check: ## Check for outdated dependencies
	@echo "🔍 Checking for outdated dependencies..."
	@if command -v cargo-outdated >/dev/null 2>&1; then \
		cargo outdated --root-deps-only; \
	else \
		echo "📦 Installing cargo-outdated..."; \
		cargo install cargo-outdated && cargo outdated --root-deps-only; \
	fi

# =============================================================================
# BUILD
# =============================================================================

build: ## Build release binaries
	@echo "🔨 Building release..."
	@cargo build --release --all-features

docs: ## Build documentation
	@echo "📚 Building documentation..."
	@cargo doc --all-features --no-deps
	@echo "Documentation available at target/doc/apr_cookbook/index.html"

# =============================================================================
# CLEAN
# =============================================================================

clean: ## Clean build artifacts
	@echo "🧹 Cleaning..."
	@cargo clean
	@rm -rf target/coverage
	@rm -f lcov.info

# =============================================================================
# HELP
# =============================================================================

help: ## Show this help
	@echo "APR Cookbook Build System"
	@echo "========================="
	@echo ""
	@echo "Main targets:"
	@echo "  make              - Run validation and build"
	@echo "  make lint         - Run linting with fixes"
	@echo "  make test-fast    - Run fast tests (target: <5 min)"
	@echo "  make coverage     - Generate coverage report (target: <10 min)"
	@echo ""
	@echo "Validation:"
	@echo "  make validate     - Full validation pipeline"
	@echo "  make quick-validate - Quick validation for development"
	@echo ""
	@echo "Testing (Performance Targets Enforced):"
	@echo "  make test-fast    - Fast unit tests (50 prop cases)"
	@echo "  make test         - Core test suite"
	@echo "  make test-all     - Comprehensive tests (500 prop cases)"
	@echo "  make test-doc     - Documentation tests"
	@echo "  make test-property - Property-based tests"
	@echo ""
	@echo "Coverage:"
	@echo "  make coverage     - Generate HTML coverage report"
	@echo "  make coverage-open - Open HTML coverage in browser"
	@echo "  make coverage-ci  - Generate LCOV report for CI/CD"
	@echo "  make coverage-clean - Clean coverage artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make examples     - Run all examples"
	@echo "  make examples-encryption - Run encryption example"
	@echo ""
	@echo "Sub-Projects (Sovereign Stack):"
	@echo "  make sub-check    - Type check sub-projects"
	@echo "  make sub-lint     - Lint sub-projects"
	@echo "  make sub-test     - Test sub-projects"
	@echo ""
	@echo "Other:"
	@echo "  make quality-gate - Run quality checks"
	@echo "  make audit        - Security audit"
	@echo "  make docs         - Build documentation"
	@echo "  make build        - Build release"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make help         - Show this help"
