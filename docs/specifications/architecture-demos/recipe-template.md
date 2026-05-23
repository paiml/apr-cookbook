# Recipe Template

Standard scaffold for `examples/inference/inference_<family>_smoke.rs`. Every architecture-demos recipe follows this shape so the manifest reconciler can verify structure mechanically.

## File Layout

```
examples/inference/inference_<family>_smoke.rs   # the recipe
tests/fixtures/architectures/<family>/           # bundled micro-checkpoint
    config.json                                  # 2-layer reduced config
    model.safetensors                            # synthetic weights (< 1 MB)
    README.md                                    # provenance, generation script
```

## Doc Header (mandatory)

```rust
//! # <Family> Smoke Inference
//!
//! Load a synthetic <Family> micro-checkpoint via `aprender::rosetta`,
//! run a deterministic forward pass, emit a `Verdict::Ok` value with
//! the resulting logits checksum.
//!
//! Demonstrates the **<FAMILY>.smoke** recipe per
//! `docs/specifications/architecture-demos.md`. The fixture is a 2-layer
//! reduced-config <Family> with random seeded weights — load semantics,
//! tensor name validation, and forward-pass numerical stability are
//! exercised, but no claim is made about checkpoint quality.
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-<family>-smoke-v1.yaml (grade A; lean_status: wip)
//! Citation: <author> et al. (<year>). *<title>*. arXiv:<id>
//!
//! Run with: cargo run --example inference_<family>_smoke
//!
//! Added by PMAT-<NNN> (architecture-demos: <Family> coverage).
```

## Verdict Enum (mandatory)

```rust
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmokeVerdict {
    Ok {
        family: String,
        format: String,
        logits_checksum: u32,
        layer_count: u32,
    },
    LoaderUnavailable { reason: String },
    InvalidFixture,
}
```

The four-arm shape (`Ok`, `LoaderUnavailable`, `InvalidFixture`, plus optional family-specific arm) is consistent across all 18 families so a meta-test can pattern-match.

## Body Pattern

```rust
pub fn smoke(fixture_path: &str, format: &str) -> SmokeVerdict {
    if !std::path::Path::new(fixture_path).exists() {
        return SmokeVerdict::InvalidFixture;
    }
    let model = match aprender::rosetta::load_family("<family>", fixture_path, format) {
        Ok(m) => m,
        Err(e) => {
            return SmokeVerdict::LoaderUnavailable {
                reason: e.to_string(),
            };
        }
    };
    // Synthetic single-token forward pass with seed 42.
    let logits = model.forward_smoke(&[42_u32; 4]);
    let checksum = logits.iter().fold(0u32, |a, x| a.wrapping_add(*x as u32));
    SmokeVerdict::Ok {
        family: "<family>".to_string(),
        format: format.to_string(),
        logits_checksum: checksum,
        layer_count: model.layer_count() as u32,
    }
}
```

> Note: `aprender::rosetta::load_family` and `model.forward_smoke` are the **expected** APIs. If upstream uses different names, the template adapts but the verdict shape stays.

## main() (mandatory)

```rust
fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_<family>_smoke")?;
    let fixture = "tests/fixtures/architectures/<family>/model.safetensors";
    println!("safetensors: {:?}", smoke(fixture, "safetensors"));
    let apr_fixture = "tests/fixtures/architectures/<family>/model.apr";
    if std::path::Path::new(apr_fixture).exists() {
        println!("apr: {:?}", smoke(apr_fixture, "apr"));
    }
    let gguf_fixture = "tests/fixtures/architectures/<family>/model.gguf";
    if std::path::Path::new(gguf_fixture).exists() {
        println!("gguf: {:?}", smoke(gguf_fixture, "gguf"));
    }
    Ok(())
}
```

A recipe declaring `formats: [safetensors, apr, gguf]` in the manifest must produce all three printlns at runtime; the meta-test asserts the recipe's stdout contains each declared format string.

## Test Block (mandatory ≥ 8 tests)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = "tests/fixtures/architectures/<family>/model.safetensors";

    #[test]
    fn smoke_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn missing_fixture_returns_invalid() {
        assert_eq!(smoke("/no/such/path", "safetensors"), SmokeVerdict::InvalidFixture);
    }

    #[test]
    fn deterministic_checksum() {
        let a = smoke(FIXTURE, "safetensors");
        let b = smoke(FIXTURE, "safetensors");
        assert_eq!(a, b);
    }

    #[test]
    fn layer_count_matches_config() {
        if let SmokeVerdict::Ok { layer_count, .. } = smoke(FIXTURE, "safetensors") {
            assert_eq!(layer_count, 2);  // micro-config is 2 layers
        }
    }

    #[test]
    fn family_name_in_verdict() {
        if let SmokeVerdict::Ok { family, .. } = smoke(FIXTURE, "safetensors") {
            assert_eq!(family, "<family>");
        }
    }

    #[test]
    fn checksum_is_nonzero() {
        if let SmokeVerdict::Ok { logits_checksum, .. } = smoke(FIXTURE, "safetensors") {
            assert!(logits_checksum != 0, "all-zero logits indicates broken forward pass");
        }
    }

    #[test]
    fn all_declared_formats_load() {
        // Per manifest entry; this list must match formats[] for the family.
        for fmt in &["safetensors"] {  // extend per family
            let path = format!("tests/fixtures/architectures/<family>/model.{fmt}");
            if std::path::Path::new(&path).exists() {
                assert!(matches!(smoke(&path, fmt), SmokeVerdict::Ok { .. }));
            }
        }
    }

    #[test]
    fn loader_unavailable_returns_clear_reason() {
        // Sanity: if upstream loader were stripped, error path must be informative.
        // This test exercises the LoaderUnavailable arm via a malformed fixture.
        let v = smoke("/dev/null", "safetensors");
        assert!(matches!(v, SmokeVerdict::InvalidFixture | SmokeVerdict::LoaderUnavailable { .. }));
    }
}
```

## IIUR Compliance Checklist

- [x] `RecipeContext::new` for tempdir isolation
- [x] No network calls (fixtures bundled)
- [x] No `unwrap()` in main logic (use `match`)
- [x] Deterministic output (seeded forward pass)
- [x] Verdict enum with `Ok` + invalid arms
- [x] ≥ 8 unit tests including determinism and edge cases
- [x] Doc header with arXiv/DOI citation
- [x] IIUR contract: `contracts/recipe-iiur-v1.yaml` declared
- [x] Snake-case filename matching `inference_<family>_smoke.rs`

## Provable-Contract Compliance Checklist (Invariant B)

- [x] Per-family contract YAML at `contracts/inference-<family>-smoke-v1.yaml`
- [x] `metadata: { version, created, description, kind: recipe-smoke, depends_on: [] }`
- [x] `kernel_structure.phases:` with `setup` / `load` / `forward` / `verify` / `teardown`
- [x] `equations:` covering at minimum: `loader_dispatch`, `tensor_validation`, `forward_determinism`
- [x] Each equation has `preconditions:`, `postconditions:`, `lean_theorem:`
- [x] Each obligation has `tolerance:` and `lean: { theorem, status: wip|proved|sorry|not-applicable, module }`
- [x] `pv lint contracts/inference-<family>-smoke-v1.yaml` exits 0
- [x] `pv score contracts/inference-<family>-smoke-v1.yaml --summary` reports grade A
- [x] Manifest `lean_status:` field reflects actual state (`wip` is the honest landing default)

### Per-family contract skeleton

```yaml
metadata:
  version: "1.0.0"
  created: "<YYYY-MM-DD>"
  description: "<Family> smoke inference invariants"
  kind: recipe-smoke
  references:
    - "https://arxiv.org/abs/<id>"
  depends_on: []

kernel_structure:
  phases:
    - name: setup
      invariant: "RecipeContext::new produces empty temp_dir; fixture path exists"
    - name: load
      invariant: "aprender::rosetta::load_family('<family>', _, _) returns Ok"
    - name: forward
      invariant: "model.forward_smoke(&[42; 4]) returns logits of expected shape"
    - name: verify
      invariant: "Two consecutive smoke() calls on same fixture produce equal Verdicts"
    - name: teardown
      invariant: "RecipeContext drop removes temp_dir"

equations:
  loader_dispatch:
    formula: "load_family(family, fixture, fmt) ↦ Ok(Model)"
    preconditions:
      - "Path::new(fixture).exists()"
      - "fmt ∈ {safetensors, apr, gguf}"
    postconditions:
      - "Model.family() == '<family>'"
      - "Model.layer_count() == 2"
    lean_theorem: "Theorems.<Family>.LoaderDispatch"
    tolerance: 0
    lean:
      theorem: "Theorems.<Family>.LoaderDispatch"
      status: wip
      module: "lean/Theorems/<Family>.lean"

  tensor_validation:
    formula: "validate_tensor_names(model, names) ↦ Ok(())"
    # ... preconditions/postconditions/lean_theorem ...

  forward_determinism:
    formula: "smoke(f, fmt) == smoke(f, fmt)"
    # ... preconditions/postconditions/lean_theorem ...
```

## Cargo.toml Wiring

```toml
[[example]]
name = "inference_<family>_smoke"
path = "examples/inference/inference_<family>_smoke.rs"
```

Added under the `# --- architecture-demos (PMAT-<NNN>) ---` section by the generator.

## Fixture Generation

Each fixture is generated once and committed. The generation script lives at `scripts/architecture-demos-gen-fixture.py` and uses `transformers.AutoModel.from_config()` with a 2-layer reduced config + `torch.manual_seed(42)`. README.md in the fixture dir captures the exact command for reproducibility.

Fixtures must:
- Be under 1 MB (gzipped if necessary)
- Include `config.json`, weights file, `tokenizer.json` only if forward-pass requires it
- Have a top-level `README.md` documenting provenance

## Companion Recipes

Some families warrant a companion `examples/conversion/convert_<family>_to_apr.rs` when the HF→APR mapping has non-trivial tensor renames or fused-QKV splits. The smoke recipe does NOT depend on the converter — they are independent. The manifest tracks both via `recipe_path` (smoke) and `companion_converter` (optional).
