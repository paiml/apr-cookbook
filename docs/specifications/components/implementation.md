# Implementation Guidelines

---

## Toyota Way Compliance

Each recipe MUST embody:

| Principle | Implementation |
|-----------|----------------|
| **Jidoka** (Built-in Quality) | Type-safe errors, compile-time validation, property tests |
| **Muda** (Waste Elimination) | No unnecessary dependencies, minimal allocations, zero-copy where possible |
| **Heijunka** (Level Loading) | Consistent recipe structure, predictable resource usage |
| **Kaizen** (Continuous Improvement) | Benchmarks for every recipe, performance regression tests |
| **Genchi Genbutsu** (Go and See) | Observable metrics, clear output, no hidden side effects |
| **Poka-Yoke** (Error-Proofing) | Impossible states unrepresentable, exhaustive pattern matching |

---

## Code Style

```rust
// GOOD: Self-documenting, minimal
fn process_model(path: &Path) -> Result<Model> {
    let bytes = std::fs::read(path)?;
    let model = BundledModel::from_bytes(&bytes)?;
    Ok(model)
}

// BAD: Over-engineered, unnecessary abstraction
trait ModelProcessor {
    type Output;
    fn process(&self) -> Result<Self::Output>;
}

struct FileModelProcessor {
    path: PathBuf,
    config: ProcessorConfig,
    logger: Box<dyn Logger>,
}
```

---

## Error Handling

```rust
// Use thiserror for domain errors
#[derive(Debug, thiserror::Error)]
pub enum RecipeError {
    #[error("Model not found: {0}")]
    ModelNotFound(PathBuf),

    #[error("Invalid format: expected {expected}, got {actual}")]
    InvalidFormat { expected: String, actual: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, RecipeError>;
```

---

## Documentation Requirements

Each recipe file MUST include:

1. **Module doc comment** with objective, run command, and dependencies
2. **Inline comments** only for non-obvious logic
3. **Example output** in doc comment
4. **Error scenarios** documented

---

## Verification After Each Phase

1. `cargo build --examples` — all compile
2. `cargo test --all-features` — all pass
3. `cargo clippy --all-targets -- -D warnings` — zero warnings

---

## Reproducibility Checklist

Each example must satisfy:

- [ ] Compiles with `cargo build --example <name>`
- [ ] Runs with `cargo run --example <name>`
- [ ] Includes `--help` documentation via clap (CLI examples)
- [ ] Has corresponding falsification tests (where applicable)
- [ ] Passes `cargo clippy -- -D warnings`
- [ ] Achieves >= 95% test coverage
- [ ] Documents all falsifiable claims with F-codes
- [ ] Works on Linux, macOS, and Windows
- [ ] WASM examples compile to `wasm32-unknown-unknown`

---

## Documentation Integration (mdbook)

### Direct Inclusion

Do NOT copy-paste code into markdown. Use mdbook's include feature:

```markdown
# Bundle a Static Model

This recipe demonstrates how to embed a model directly into your binary.

{{#include ../../examples/bundling/bundle_apr_static_binary.rs}}
```

### Structure Alignment

`SUMMARY.md` must mirror the 12 Categories (A-L):
- Required: `recipes/category-b-bundling/bundle-static-binary.md`
- Not: `recipes/bundle-static.md`

### Status Badges

Each recipe page starts with:

```markdown
> **Recipe Status**: Verified | **Idempotent**: Yes | **Coverage**: 100%
```

### CI Validation

CI verifies every file in `examples/` is referenced at least once in `book/src/` to prevent orphaned recipes.
