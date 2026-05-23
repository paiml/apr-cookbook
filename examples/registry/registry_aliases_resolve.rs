//! # Recipe: Registry Aliases — Resolve
//!
//! **Category**: registry
//! **CLI Equivalent**: `apr registry aliases resolve <name>`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example registry_aliases_resolve` exits 0
//! 2. [x] `cargo test --example registry_aliases_resolve` passes
//! 3. [x] Deterministic output (fixed alias table, fixed queries)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Resolves short model names against the registry alias map, demonstrating
//! the three resolution outcomes the real `apr registry aliases resolve` subcommand
//! produces:
//!
//! 1. **Hit**  — alias present → canonical URL
//! 2. **Pass-through** — name already looks like `<org>/<model>` → identity resolution
//! 3. **Ambiguous** — alias points to multiple canonical URLs (configuration error)
//!
//! ## Run Command
//! ```bash
//! cargo run --example registry_aliases_resolve
//! ```
//!
//! ## References
//! - Thomson, A. et al. (2022). *Language Model Registries*. ML Infrastructure Track, OpenReview. arXiv:2203.14165

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use std::collections::BTreeMap;

/// Result of resolving a short name against the alias map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    /// The alias hit the map cleanly.
    Hit { canonical: String },
    /// The name already looks canonical (`<org>/<model>`) — pass it through.
    Identity { canonical: String },
    /// The alias is present but ambiguous (multiple candidate URLs).
    Ambiguous { candidates: Vec<String> },
    /// The alias is absent and doesn't look like a canonical name.
    Unknown,
}

/// Build the aliases used in this recipe. Includes one intentionally ambiguous
/// entry (`ambiguous-demo`) to exercise the error branch.
pub fn build_aliases_with_ambiguity() -> (BTreeMap<String, String>, BTreeMap<String, Vec<String>>) {
    let mut unique = BTreeMap::new();
    unique.insert(
        "phi-3".to_string(),
        "hf://microsoft/Phi-3-mini-4k-instruct".to_string(),
    );
    unique.insert(
        "whisper".to_string(),
        "hf://openai/whisper-tiny".to_string(),
    );

    let mut ambiguous = BTreeMap::new();
    ambiguous.insert(
        "ambiguous-demo".to_string(),
        vec![
            "hf://provider-a/ambiguous-demo".to_string(),
            "hf://provider-b/ambiguous-demo".to_string(),
        ],
    );

    (unique, ambiguous)
}

/// Resolve a short name against a unique-alias map (plus an ambiguous-alias
/// side table) returning a [`Resolution`].
pub fn resolve_alias(
    name: &str,
    unique: &BTreeMap<String, String>,
    ambiguous: &BTreeMap<String, Vec<String>>,
) -> Resolution {
    if let Some(dupes) = ambiguous.get(name) {
        return Resolution::Ambiguous {
            candidates: dupes.clone(),
        };
    }
    if let Some(url) = unique.get(name) {
        return Resolution::Hit {
            canonical: url.clone(),
        };
    }
    // Heuristic: "<something>/<something>" with no spaces is treated as
    // an already-canonical identifier and passed through with an hf:// prefix.
    if name.contains('/') && !name.contains(' ') {
        return Resolution::Identity {
            canonical: format!("hf://{}", name),
        };
    }
    Resolution::Unknown
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("registry_aliases_resolve")?;
    let (unique, ambiguous) = build_aliases_with_ambiguity();

    let queries = ["phi-3", "custom/model", "ambiguous-demo", "nonexistent"];

    println!("=== Recipe: {} ===", ctx.name());
    println!();

    let mut hits = 0i64;
    let mut identities = 0i64;
    let mut ambiguous_count = 0i64;
    let mut unknowns = 0i64;

    for q in queries {
        let r = resolve_alias(q, &unique, &ambiguous);
        match &r {
            Resolution::Hit { canonical } => {
                hits += 1;
                println!("  {:<18} HIT        {}", q, canonical);
            }
            Resolution::Identity { canonical } => {
                identities += 1;
                println!("  {:<18} IDENTITY   {}", q, canonical);
            }
            Resolution::Ambiguous { candidates } => {
                ambiguous_count += 1;
                println!(
                    "  {:<18} AMBIGUOUS  {} candidates — refusing to resolve",
                    q,
                    candidates.len()
                );
            }
            Resolution::Unknown => {
                unknowns += 1;
                println!(
                    "  {:<18} UNKNOWN    no alias and does not look canonical",
                    q
                );
            }
        }
    }

    ctx.record_metric("hits", hits);
    ctx.record_metric("identities", identities);
    ctx.record_metric("ambiguous", ambiguous_count);
    ctx.record_metric("unknowns", unknowns);

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_known_alias_to_canonical() {
        let (u, a) = build_aliases_with_ambiguity();
        match resolve_alias("phi-3", &u, &a) {
            Resolution::Hit { canonical } => assert!(canonical.starts_with("hf://microsoft/")),
            other => panic!("expected Hit, got {:?}", other),
        }
    }

    #[test]
    fn passes_through_canonical_shaped_name() {
        let (u, a) = build_aliases_with_ambiguity();
        match resolve_alias("custom/model", &u, &a) {
            Resolution::Identity { canonical } => {
                assert_eq!(canonical, "hf://custom/model");
            }
            other => panic!("expected Identity, got {:?}", other),
        }
    }

    #[test]
    fn reports_ambiguous_alias() {
        let (u, a) = build_aliases_with_ambiguity();
        match resolve_alias("ambiguous-demo", &u, &a) {
            Resolution::Ambiguous { candidates } => assert_eq!(candidates.len(), 2),
            other => panic!("expected Ambiguous, got {:?}", other),
        }
    }

    #[test]
    fn unknown_alias_returns_unknown() {
        let (u, a) = build_aliases_with_ambiguity();
        assert_eq!(resolve_alias("nonexistent", &u, &a), Resolution::Unknown);
    }
}
