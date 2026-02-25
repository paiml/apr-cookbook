# APR Cookbook — just task runner (mirrors Makefile)

default: validate build

# Quick validation for development
quick-validate: format-check lint-check check test-fast
    @echo "Quick validation passed"

# Full validation pipeline
validate: format lint check test-fast audit
    @echo "All validation passed"

# Format code
format:
    cargo fmt --all

# Check code formatting
format-check:
    cargo fmt --all -- --check

# Run clippy with auto-fix
lint:
    cargo clippy --all-targets --all-features --fix --allow-dirty --allow-staged 2>/dev/null || true
    cargo clippy --all-targets --all-features -- -D warnings

# Check clippy without fixing
lint-check:
    cargo clippy --all-targets --all-features -- -D warnings

# Type check all targets
check:
    cargo check --all-targets --all-features

# Fast tests (< 5 min)
test-fast:
    PROPTEST_CASES=25 cargo test --workspace

# Core test suite
test: test-fast test-doc
    @echo "Core test suite completed"

# Documentation tests
test-doc:
    PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --doc --workspace

# Build all examples
examples:
    cargo build --examples

# Build release
build:
    cargo build --release --all-features

# Build documentation
docs:
    cargo doc --all-features --no-deps

# Security audit
audit:
    cargo deny check
    cargo audit

# Generate coverage report
coverage:
    PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo llvm-cov test --lib --no-report --all-features --workspace
    cargo llvm-cov report --html --output-dir target/coverage/html
    cargo llvm-cov report --summary-only

# Update dependencies
update-deps:
    cargo update

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/coverage
    rm -f lcov.info
