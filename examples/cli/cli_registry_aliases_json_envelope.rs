//! # apr registry aliases — `--json` Envelope
//!
//! `apr registry aliases --json` emits a single JSON object
//! `{"aliases": {<name>: <canonical_url>, ...}}`. This recipe builds the
//! envelope generator and asserts the contract: top-level "aliases" key
//! always present, the inner map preserves insertion order via BTreeMap,
//! empty alias set still emits `{"aliases": {}}` rather than `{}`.
//!
//! Demonstrates the **REGISTRY-ALIASES.5** recipe for PMAT-103 (apr registry aliases coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CRUX-A-01
//!
//! Run with: cargo run --example cli_registry_aliases_json_envelope
//!
//! Added by PMAT-103 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub fn render_envelope(aliases: &BTreeMap<String, String>) -> Value {
    json!({ "aliases": aliases })
}

pub fn validate_envelope(v: &Value) -> Result<usize> {
    let aliases = v.get("aliases").and_then(Value::as_object).ok_or_else(|| {
        apr_cookbook::CookbookError::Validation("envelope missing 'aliases' key".into())
    })?;
    Ok(aliases.len())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_registry_aliases_json_envelope")?;

    let mut m: BTreeMap<String, String> = BTreeMap::new();
    m.insert(
        "qwen-coder-7b".into(),
        "hf://Qwen/Qwen2.5-Coder-7B-Instruct".into(),
    );
    m.insert("whisper-tiny".into(), "hf://openai/whisper-tiny".into());

    let env = render_envelope(&m);
    println!("{}", serde_json::to_string_pretty(&env).unwrap());
    println!("\nempty: {}", render_envelope(&BTreeMap::new()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn envelope_has_aliases_key() {
        let mut m = BTreeMap::new();
        m.insert("a".into(), "hf://a".into());
        let env = render_envelope(&m);
        assert!(env.get("aliases").is_some());
    }

    #[test]
    fn empty_alias_set_still_emits_aliases_key() {
        let env = render_envelope(&BTreeMap::new());
        assert_eq!(env, json!({ "aliases": {} }));
    }

    #[test]
    fn validate_returns_count() {
        let mut m = BTreeMap::new();
        m.insert("a".into(), "hf://a".into());
        m.insert("b".into(), "hf://b".into());
        let env = render_envelope(&m);
        assert_eq!(validate_envelope(&env).unwrap(), 2);
    }

    #[test]
    fn validate_rejects_missing_aliases_key() {
        let bad = json!({ "wrong_key": {} });
        assert!(validate_envelope(&bad).is_err());
    }

    #[test]
    fn validate_rejects_non_object_aliases() {
        let bad = json!({ "aliases": ["a", "b"] });
        assert!(validate_envelope(&bad).is_err());
    }

    #[test]
    fn output_keys_sorted_via_btreemap() {
        // BTreeMap ordering preserved when serialized.
        let mut m = BTreeMap::new();
        m.insert("z".into(), "z".into());
        m.insert("a".into(), "a".into());
        m.insert("m".into(), "m".into());
        let env = render_envelope(&m);
        let serialized = serde_json::to_string(&env).unwrap();
        // Keys appear in alphabetical order.
        let a_pos = serialized.find("\"a\":\"a\"").unwrap();
        let m_pos = serialized.find("\"m\":\"m\"").unwrap();
        let z_pos = serialized.find("\"z\":\"z\"").unwrap();
        assert!(a_pos < m_pos);
        assert!(m_pos < z_pos);
    }
}
