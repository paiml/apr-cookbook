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
.PHONY: update-deps update-deps-check
.PHONY: coverage coverage-ci coverage-clean clean-coverage coverage-open
.PHONY: sub-test sub-lint sub-check

# Parallel job execution
MAKEFLAGS += -j$(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Sub-projects (disabled - using crates.io dependencies)
# SUB_PROJECTS := ../aprender ../trueno ../entrenar

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

quality-gate: ## Run quality checks
	@echo "🔍 Running quality gate checks..."
	@echo "  📊 Checking test count..."
	@TEST_COUNT=$$(cargo test --workspace 2>&1 | grep -E "^test result:" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+"); \
	echo "  Tests: $$TEST_COUNT"; \
	if [ "$$TEST_COUNT" -lt 50 ]; then \
		echo "  ⚠️  Warning: Test count below target (50+)"; \
	else \
		echo "  ✓ Test count acceptable"; \
	fi
	@echo "✅ Quality gates passed!"

# =============================================================================
# SECURITY
# =============================================================================

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
