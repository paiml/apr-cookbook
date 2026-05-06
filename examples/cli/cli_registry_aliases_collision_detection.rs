//! # apr registry aliases — Collision Detection
//!
//! Multiple alias entries pointing to the same canonical URL is allowed
//! (e.g., `qwen-coder-7b` and `qwen2.5-coder` both → same model). But
//! the SAME alias declared TWICE in the YAML is a configuration error.
//! This recipe builds the duplicate-key detector since serde_yaml's
//! default Mapping silently keeps the last entry — operator never sees
//! the override.
//!
//! Demonstrates the **REGISTRY-ALIASES.6** recipe for PMAT-103 (apr registry aliases coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CRUX-A-01 + YAML 1.2 spec (duplicate-key behavior)
//!
//! Run with: cargo run --example cli_registry_aliases_collision_detection
//!
//! Added by PMAT-103 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq)]
pub struct CollisionReport {
    pub duplicate_aliases: Vec<String>,
    pub shared_canonicals: BTreeMap<String, Vec<String>>, // canonical → [aliases]
}

/// Manually scan raw YAML lines for repeated keys (since serde_yaml swallows them).
pub fn detect_duplicate_keys(yaml: &str) -> Vec<String> {
    let mut seen: BTreeMap<String, u32> = BTreeMap::new();
    for line in yaml.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(colon) = trimmed.find(':') {
            let key = trimmed[..colon].trim().to_string();
            *seen.entry(key).or_insert(0) += 1;
        }
    }
    seen.into_iter()
        .filter(|(_, n)| *n > 1)
        .map(|(k, _)| k)
        .collect()
}

/// Group alias map by canonical URL — multiple aliases per canonical is allowed
/// but worth surfacing as a report so the operator can choose to consolidate.
pub fn group_by_canonical(aliases: &BTreeMap<String, String>) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (alias, canonical) in aliases {
        out.entry(canonical.clone())
            .or_default()
            .push(alias.clone());
    }
    out.retain(|_, v| v.len() > 1);
    out
}

pub fn full_report(yaml: &str, parsed: &BTreeMap<String, String>) -> CollisionReport {
    CollisionReport {
        duplicate_aliases: detect_duplicate_keys(yaml),
        shared_canonicals: group_by_canonical(parsed),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_registry_aliases_collision_detection")?;

    let yaml = "qwen-coder-7b: hf://Qwen/Qwen2.5-Coder-7B-Instruct\nqwen2.5-coder: hf://Qwen/Qwen2.5-Coder-7B-Instruct\nqwen-coder-7b: hf://Qwen/typo-here\n";
    let mut parsed: BTreeMap<String, String> = BTreeMap::new();
    parsed.insert(
        "qwen-coder-7b".into(),
        "hf://Qwen/Qwen2.5-Coder-7B-Instruct".into(),
    );
    parsed.insert(
        "qwen2.5-coder".into(),
        "hf://Qwen/Qwen2.5-Coder-7B-Instruct".into(),
    );

    let report = full_report(yaml, &parsed);
    println!("{report:#?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collision_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn duplicate_keys_detected() {
        let yaml = "a: 1\nb: 2\na: 3\n";
        let dup = detect_duplicate_keys(yaml);
        assert_eq!(dup, vec!["a".to_string()]);
    }

    #[test]
    fn no_duplicates_yields_empty() {
        let yaml = "a: 1\nb: 2\nc: 3\n";
        assert!(detect_duplicate_keys(yaml).is_empty());
    }

    #[test]
    fn comments_and_blank_lines_skipped() {
        let yaml = "# comment\na: 1\n\n# another\nb: 2\n";
        assert!(detect_duplicate_keys(yaml).is_empty());
    }

    #[test]
    fn shared_canonical_grouped() {
        let mut m = BTreeMap::new();
        m.insert("alias-a".into(), "hf://X".into());
        m.insert("alias-b".into(), "hf://X".into());
        m.insert("alias-c".into(), "hf://Y".into());
        let g = group_by_canonical(&m);
        // hf://X has 2 aliases; hf://Y has 1 (filtered out).
        assert!(g.contains_key("hf://X"));
        assert_eq!(g["hf://X"].len(), 2);
        assert!(!g.contains_key("hf://Y"));
    }

    #[test]
    fn unique_canonicals_yield_empty_group_report() {
        let mut m = BTreeMap::new();
        m.insert("a".into(), "hf://A".into());
        m.insert("b".into(), "hf://B".into());
        assert!(group_by_canonical(&m).is_empty());
    }

    #[test]
    fn full_report_combines_both_signals() {
        let yaml = "a: hf://X\nb: hf://X\na: hf://typo\n";
        let mut parsed = BTreeMap::new();
        parsed.insert("a".into(), "hf://X".into());
        parsed.insert("b".into(), "hf://X".into());
        let report = full_report(yaml, &parsed);
        assert_eq!(report.duplicate_aliases, vec!["a".to_string()]);
        assert!(report.shared_canonicals.contains_key("hf://X"));
    }
}
