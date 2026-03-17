# IIUR Principles & Recipe Architecture

Every recipe in the APR Cookbook follows four principles: **Isolated**, **Idempotent**, **Useful**, and **Reproducible**.

---

## 1. Isolated

Each recipe MUST:

- **No shared mutable state**: No global variables, no shared filesystems, no persistent databases between runs
- **Self-contained dependencies**: All required assets created inline or embedded via `include_bytes!()`
- **Temp directory isolation**: Any file I/O uses `tempfile::tempdir()` with automatic cleanup
- **Feature flag independence**: Recipes work with their declared features only
- **Thread safety**: Concurrent execution of any two recipes produces identical results

```rust
// CORRECT: Isolated recipe
fn main() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let model_path = temp.path().join("model.apr");
    // ... work within temp directory
    Ok(())  // temp directory automatically cleaned up
}

// INCORRECT: Shares state
static mut GLOBAL_MODEL: Option<Model> = None;  // Violates isolation
```

## 2. Idempotent

Each recipe MUST:

- **f(f(x)) = f(x)**: Running a recipe twice produces identical output
- **No accumulation**: Repeated runs do not accumulate files, state, or side effects
- **Deterministic seeds**: Any randomness uses fixed seeds for reproducibility
- **Atomic operations**: Either fully succeeds or fully fails with no partial state

```rust
// CORRECT: Idempotent with deterministic seed
let rng = StdRng::seed_from_u64(42);
let model = train_with_rng(&data, rng)?;

// INCORRECT: Non-deterministic
let model = train(&data)?;  // Uses thread_rng internally
```

## 3. Useful

Each recipe MUST:

- **Solve a real problem**: Addresses a concrete use case from production ML workflows
- **Executable demonstration**: `cargo run --example <name>` produces meaningful output
- **Clear learning objective**: Single concept per recipe with explicit takeaway
- **Copy-paste ready**: Code can be directly adapted for production use

## 4. Reproducible

Each recipe MUST:

- **Pinned dependencies**: Uses exact versions from workspace `Cargo.lock`
- **Cross-platform**: Works on x86_64 Linux, aarch64 Linux, aarch64 macOS, WASM
- **CI-verified**: All recipes run in CI on every commit
- **Documented environment**: Clearly states any system requirements

---

## Recipe Structure

Every recipe follows this canonical structure:

```
examples/
└── category/
    └── recipe_name.rs
```

### File Template

```rust
//! # Recipe: [Descriptive Title]
//!
//! **Category**: [Category Name]
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: [List feature flags required]
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (if applicable)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! [One sentence describing what this recipe teaches]
//!
//! ## Run Command
//! ```bash
//! cargo run --example recipe_name [--features feature1,feature2]
//! ```

use apr_cookbook::prelude::*;

fn main() -> apr_cookbook::Result<()> {
    let ctx = RecipeContext::new("recipe_name")?;
    let result = execute_recipe(&ctx)?;
    ctx.report(&result)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recipe_idempotent() {
        let result1 = main();
        let result2 = main();
        assert_eq!(result1.is_ok(), result2.is_ok());
    }

    #[test]
    fn test_recipe_isolated() {
        // Verify no side effects persist
    }
}
```

### CLI Demo Template

```rust
//! # Recipe: [Title]
//!
//! **Category**: [optimize|chat|analysis|format]
//! **CLI Equivalent**: `apr [command] [flags]`
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Clippy clean
//! 6. [x] No `unwrap()` in logic
//!
//! ## Learning Objective
//! [What this teaches]

use apr_cookbook::prelude::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("example_name")?;
    // ... sections
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // 8-15 unit tests
}
```

---

## RecipeContext Utility

Provides standardized isolation primitives:

```rust
pub struct RecipeContext {
    pub temp_dir: TempDir,
    pub rng: StdRng,
    pub metadata: RecipeMetadata,
}

impl RecipeContext {
    pub fn new(name: &str) -> Result<Self> {
        let seed = hash_name_to_seed(name);
        Ok(Self {
            temp_dir: tempfile::tempdir()?,
            rng: StdRng::seed_from_u64(seed),
            metadata: RecipeMetadata::from_name(name),
        })
    }

    pub fn path(&self, filename: &str) -> PathBuf {
        self.temp_dir.path().join(filename)
    }
}
```

---

## Test Harness Requirements

| Test Type | Requirement | Coverage |
|-----------|-------------|----------|
| Unit Tests | Core logic verification | 95% minimum |
| Idempotency Test | `main(); main();` produces same result | Required |
| Isolation Test | No filesystem leaks after run | Required |
| Property Tests | Proptest for input variations | 3+ properties |
| Doc Tests | All code examples compile | Required |

### Property Test Pattern

```rust
proptest! {
    #[test]
    fn prop_deterministic_output(seed in 0u64..1000) {
        let r1 = run_with_seed(seed);
        let r2 = run_with_seed(seed);
        prop_assert_eq!(r1, r2);
    }

    #[test]
    fn prop_valid_output_format(input in valid_inputs()) {
        let output = process(input);
        prop_assert!(is_valid_format(&output));
    }

    #[test]
    fn prop_no_panics(input in any_input()) {
        let _ = process(input);  // Should not panic
    }
}
```
