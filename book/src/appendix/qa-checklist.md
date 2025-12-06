# Recipe QA Checklist

Every recipe in this cookbook is verified against this checklist.

## Status Block

Each recipe page displays a status block:

```
> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+
```

- **Verified**: Recipe compiles and runs successfully
- **Idempotent**: Running twice produces identical output
- **Coverage**: Percentage of code covered by tests

## Verification Steps

### 1. Build Verification
```bash
cargo build --example recipe_name
```
Must exit with code 0.

### 2. Run Verification
```bash
cargo run --example recipe_name
```
Must produce expected output without errors.

### 3. Test Coverage
```bash
cargo test --example recipe_name
```
All unit tests pass.

### 4. Determinism Check
```bash
# Run twice, compare output
cargo run --example recipe_name > out1.txt
cargo run --example recipe_name > out2.txt
diff out1.txt out2.txt
```
No differences for deterministic recipes.

### 5. Memory Check
```bash
# Verify no leaks
valgrind cargo run --example recipe_name
```
No memory leaks reported.

### 6. Lint Verification
```bash
cargo clippy --example recipe_name -- -D warnings
```
No warnings.

## Property Tests

Each recipe includes property-based tests using `proptest`:

```rust
proptest! {
    #[test]
    fn prop_invariant_holds(input in strategy()) {
        // Verify invariant for all generated inputs
        prop_assert!(check_invariant(input));
    }
}
```

Minimum 100 test cases per property.
