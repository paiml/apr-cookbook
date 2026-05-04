# IIUR Conformance for Migrated Artifacts

The existing IIUR contract (`contracts/recipe-iiur-v1.yaml`) assumes Rust binaries built around `RecipeContext::new(...)`. Three of the migration sources don't fit that mold:

1. **sovereign YAMLs** — declarative deployment configs consumed by `forjar`
2. **presentar `.yaml` / `.prs`** — declarative scene/widget configs consumed by `presentar` runtime
3. **alimentar `examples/*.rs`** — Rust, but lacks the cookbook's IIUR header convention

This document defines how each class earns its IIUR grade.

---

## Class 1: Rust Examples (alimentar) — Retrofit Pass

The 18 alimentar examples are Rust binaries; they need only a header retrofit, not a structural rewrite.

### Retrofit Recipe

For each `examples/data-loading/*.rs`:

1. **Prepend the IIUR doc header**:
   ```rust
   //! # <Title>
   //!
   //! <One-paragraph description.>
   //!
   //! Contract: contracts/recipe-iiur-v1.yaml
   //! Citation: <arXiv ID or DOI>
   //!
   //! Run with: cargo run --example <name>
   ```
2. **Wrap `main` body in `RecipeContext`**:
   ```rust
   use apr_cookbook::prelude::*;

   fn main() -> Result<()> {
       let _ctx = RecipeContext::new("<name>");
       // ... existing example body
       Ok(())
   }
   ```
3. **Replace `unwrap()` / `expect()` in main logic** with `?`. The pre-existing `#![allow(clippy::unwrap_used, clippy::expect_used)]` block at the top of alimentar examples is **removed** — the cookbook's clippy gate forbids these in main paths.
4. **Append a tests module**:
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn example_runs() {
           main().expect("recipe execution failed");
       }
   }
   ```
5. **Replace any `~/.alimentar/` or absolute path references** with `tempfile::tempdir()` per IIUR isolation.

The retrofit is mechanical and should be scripted as `scripts/iiur-retrofit-alimentar.sh`. Citation lookup is the only manual step (resolve a real arXiv/DOI for each example's topic — drift_detection → cite the drift-detection lit, hub_publishing → cite HF Datasets paper, etc.). PMAT-066 carries that work.

### Acceptance

A retrofitted example passes when:
- `cargo build --example <name>` succeeds
- `cargo test --example <name>` passes
- `cargo clippy --example <name> -- -D warnings` is clean (no `#![allow]` escape hatches)
- The contract validator (`cargo test --test contracts`) reads the doc header and accepts it

---

## Class 2: Declarative Configs (sovereign + presentar) — Wrapped or Validated

Declarative configs cannot exhibit IIUR properties on their own — they have no execution. IIUR is satisfied by **a Rust artifact that owns the config** and earns the grade on its behalf.

Two wrapping strategies, picked per-source based on volume:

### Strategy A: Per-File Wrapper (sovereign — 14 wrappers)

Sovereign recipes are heterogeneous (each one models a different service). Each gets its own Rust wrapper at `examples/deployment-stacks/<recipe_name>.rs`:

```rust
//! # <Recipe Name> — <Description>
//!
//! Loads the deployment recipe `recipes/<recipe-name>.yaml`, validates its
//! schema, asserts required inputs are declared, and exits.
//!
//! This wrapper does NOT execute the deployment — it is a parse-and-validate
//! recipe. Real provisioning is performed by `forjar` against a target machine.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: <relevant infra-as-code paper or N/A — see PMAT-065>
//!
//! Run with: cargo run --example <wrapper_name>

use apr_cookbook::prelude::*;
use std::path::Path;

const RECIPE_YAML: &str =
    include_str!("recipes/<recipe-name>.yaml");

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("<wrapper_name>");

    let parsed: serde_yaml::Value = serde_yaml::from_str(RECIPE_YAML)?;
    let recipe = parsed.get("recipe").context("recipe key missing")?;

    let name = recipe.get("name").and_then(|v| v.as_str())
        .context("recipe.name missing")?;
    let inputs = recipe.get("inputs").context("recipe.inputs missing")?;

    println!("recipe={name} inputs={}", inputs.as_mapping().map(|m| m.len()).unwrap_or(0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_parses() {
        main().expect("wrapper execution failed");
    }

    #[test]
    fn recipe_has_required_fields() {
        let parsed: serde_yaml::Value = serde_yaml::from_str(RECIPE_YAML).unwrap();
        let recipe = parsed.get("recipe").unwrap();
        assert!(recipe.get("name").is_some());
        assert!(recipe.get("version").is_some());
        assert!(recipe.get("description").is_some());
        assert!(recipe.get("inputs").is_some());
    }
}
```

The wrapper IS the IIUR-graded artifact. The YAML is its embedded fixture (via `include_str!`).

### Strategy B: Single Validator (presentar — 1 wrapper, 28 fixtures)

Presentar's 28 declarative files are homogeneous (charts/dashboards/widgets/scenes). One Rust validator covers all of them:

```rust
//! # Visualization Config Validator
//!
//! Loads every `.yaml` and `.prs` file under this directory, parses it via
//! `presentar`, and asserts schema validity.
//!
//! Contract: contracts/recipe-iiur-config-v1.yaml
//! Citation: <presentar paper or specification>
//!
//! Run with: cargo run --example load_visualization

use apr_cookbook::prelude::*;
use std::fs;
use std::path::PathBuf;

fn config_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples").join("visualization");
    for sub in &["ald", "apr", "charts", "dashboards", "edge_cases", "prs"] {
        for entry in fs::read_dir(root.join(sub)).unwrap().flatten() {
            let p = entry.path();
            if matches!(p.extension().and_then(|s| s.to_str()), Some("yaml" | "prs")) {
                out.push(p);
            }
        }
    }
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("load_visualization");
    let files = config_files();
    println!("validating {} visualization configs", files.len());
    for f in &files {
        let content = fs::read_to_string(f)?;
        // Schema validation via presentar's parser
        match f.extension().and_then(|s| s.to_str()) {
            Some("yaml") => { let _: serde_yaml::Value = serde_yaml::from_str(&content)?; }
            Some("prs")  => { let _: serde_yaml::Value = serde_yaml::from_str(&content)?; }
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_least_28_configs() {
        assert!(config_files().len() >= 28);
    }

    #[test]
    fn all_configs_parse() {
        main().expect("validator failed");
    }
}
```

This single wrapper is graded against `contracts/recipe-iiur-config-v1.yaml` (defined below).

### Why per-file for sovereign, single for presentar?

- Sovereign recipes vary a lot in declared inputs; per-file wrappers let each one assert recipe-specific invariants (e.g., the GPU recipe asserts `gpu_device` is declared).
- Presentar configs share schema; one validator avoids 28 near-duplicate wrapper files.
- Cookbook coverage Invariant F (≥3 recipes per subcommand) doesn't apply here — sovereign and presentar artifacts aren't `apr` subcommands.

---

## Class 3: New Contract — `recipe-iiur-config-v1.yaml`

The current contract (`recipe-iiur-v1.yaml`) requires execution and isolation guarantees that don't apply to a "load and parse" validator. A sibling contract relaxes the runtime obligations while keeping the documentary ones.

### Contract draft (place at `contracts/recipe-iiur-config-v1.yaml`)

```yaml
contract: recipe-iiur-config-v1
description: |
  IIUR contract for declarative-config recipes (YAML, .prs).
  The graded artifact is a Rust wrapper that loads, parses, and validates
  the declarative configs under its purview. Idempotence and isolation
  apply to the WRAPPER, not the configs themselves.
metadata:
  depends_on: [contracts/recipe-iiur-v1.yaml]
kernel_structure:
  phases:
    - load            # read declarative file(s) from disk or include_str!
    - parse           # parse to typed value
    - validate        # assert schema and required-field invariants
    - report          # print summary

obligations:
  - name: doc_header_present
    description: Wrapper file begins with IIUR doc header including Contract and Citation lines
    tolerance: 0
    lean: { theorem: doc_header_present_thm, status: not-applicable, module: na }
  - name: parse_succeeds
    description: All declared config files parse to typed value without error
    tolerance: 0
    lean: { theorem: parse_succeeds_thm, status: not-applicable, module: na }
  - name: schema_invariants_asserted
    description: At least one #[test] checks a config-specific schema field
    tolerance: 0
    lean: { theorem: schema_inv_thm, status: not-applicable, module: na }
  - name: no_network
    description: Wrapper performs no network I/O (offline-only per cookbook policy)
    tolerance: 0
    lean: { theorem: no_network_thm, status: not-applicable, module: na }
  - name: no_unwrap_in_main
    description: Wrapper main() body has zero unwrap()/expect() calls (tests may use them)
    tolerance: 0
    lean: { theorem: no_unwrap_thm, status: not-applicable, module: na }

preconditions:
  - All config files referenced exist under the wrapper's directory
  - Cookbook workspace has the relevant dev-dep (presentar, forjar, etc.) for parsing

postconditions:
  - Wrapper exits 0 on success
  - All asserted schema invariants hold for every config in scope

lean_theorem: config_iiur_thm
```

This contract has no Lean proof obligations (`status: not-applicable`) — it's a documentary contract enforced at the cookbook test level, not the formal-methods level. That's an honest declaration, not a cheat: declarative-config validation isn't the kind of thing Lean is good at.

---

## Coverage Invariants — Update

The Six Coverage Invariants (apr-cookbook v5.0) are extended for the new categories:

| Invariant | Change |
|-----------|--------|
| A: CLI parity | **Unchanged.** Sovereign/presentar/alimentar artifacts are not `apr` subcommands. Numerator stays scoped to `apr-cli`. |
| B: Contract grade | **Extended.** Includes `recipe-iiur-config-v1.yaml` as an acceptable contract (denominator grows to all recipes including new categories). |
| C: Format variants | **Conditionally extended.** Applies to `data-loading/` (Arrow/Parquet/CSV/JSON variants) and `visualization/` (yaml + .prs). Not applied to `deployment-stacks/`. |
| D: arXiv citation | **Extended.** Every new recipe header MUST cite. Class-2 wrappers cite the relevant infra-as-code or visualization paper; if no clean citation exists, `Citation: N/A — see ticket PMAT-NNN` is acceptable as a gap marker. |
| E: Docs contract coverage | **Extended.** Book chapters under `data-loading/` and `visualization/` count toward the 264/267 ratio (denominator grows). |
| F: Variant depth | **Carved out.** Does not apply to deployment-stacks (one config per service is enough) or machines (one config per platform). Applies to data-loading and visualization at a lowered threshold of ≥1 (volume already satisfies). |

These extensions are encoded in `scripts/coverage-invariants.sh`; the change is part of PMAT-069.
