//! # apr registry aliases — YAML Loader Validator
//!
//! `apr registry aliases` reads `configs/aliases.yaml`. This recipe
//! builds the YAML loader contract: the file must be a top-level map of
//! string→string entries; arrays/numbers/nested objects reject; missing
//! file produces an empty map (not an error — operator may have a fresh
//! install).
//!
//! Demonstrates the **REGISTRY-ALIASES.4** recipe for PMAT-103 (apr registry aliases coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CRUX-A-01 + serde_yaml type system
//!
//! Run with: cargo run --example cli_registry_aliases_yaml_loader
//!
//! Added by PMAT-103 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum LoadVerdict {
    Ok(BTreeMap<String, String>),
    NotMap,
    NonStringValue { key: String },
    YamlParseError(String),
    EmptyFile,
}

pub fn load_aliases(yaml: &str) -> LoadVerdict {
    if yaml.trim().is_empty() {
        return LoadVerdict::EmptyFile;
    }
    let value: serde_yaml::Value = match serde_yaml::from_str(yaml) {
        Ok(v) => v,
        Err(e) => return LoadVerdict::YamlParseError(e.to_string()),
    };
    let serde_yaml::Value::Mapping(map) = value else {
        return LoadVerdict::NotMap;
    };
    let mut out = BTreeMap::new();
    for (k, v) in map {
        let Some(key) = k.as_str().map(str::to_string) else {
            continue;
        };
        let serde_yaml::Value::String(val) = v else {
            return LoadVerdict::NonStringValue { key };
        };
        out.insert(key, val);
    }
    LoadVerdict::Ok(out)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_registry_aliases_yaml_loader")?;

    let happy = "qwen-coder-7b: hf://Qwen/Qwen2.5-Coder-7B-Instruct\nwhisper-tiny: hf://openai/whisper-tiny\n";
    let bad_array = "- foo\n- bar\n";
    let bad_value = "qwen: 42\n";

    println!("happy:     {:?}", load_aliases(happy));
    println!("bad-array: {:?}", load_aliases(bad_array));
    println!("bad-value: {:?}", load_aliases(bad_value));
    println!("empty:     {:?}", load_aliases(""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loader_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_yaml_loads_to_map() {
        let yaml = "qwen: hf://Qwen/foo\nwhisper: hf://openai/bar\n";
        let v = load_aliases(yaml);
        if let LoadVerdict::Ok(m) = v {
            assert_eq!(m.len(), 2);
            assert_eq!(m.get("qwen").map(String::as_str), Some("hf://Qwen/foo"));
        } else {
            panic!("expected Ok, got {v:?}");
        }
    }

    #[test]
    fn empty_file_yields_empty_verdict() {
        assert_eq!(load_aliases(""), LoadVerdict::EmptyFile);
        assert_eq!(load_aliases("   \n  \n"), LoadVerdict::EmptyFile);
    }

    #[test]
    fn array_at_top_level_rejected() {
        let yaml = "- foo\n- bar\n";
        let v = load_aliases(yaml);
        assert_eq!(v, LoadVerdict::NotMap);
    }

    #[test]
    fn non_string_value_rejected() {
        let yaml = "qwen: 42\n";
        let v = load_aliases(yaml);
        assert!(matches!(v, LoadVerdict::NonStringValue { .. }));
    }

    #[test]
    fn parse_error_surfaces() {
        let yaml = "this: is: not: valid: yaml";
        let v = load_aliases(yaml);
        assert!(matches!(v, LoadVerdict::YamlParseError(_)));
    }

    #[test]
    fn output_keys_sorted_via_btreemap() {
        let yaml = "z: hf://z\na: hf://a\nm: hf://m\n";
        if let LoadVerdict::Ok(m) = load_aliases(yaml) {
            let keys: Vec<&String> = m.keys().collect();
            let mut sorted = keys.clone();
            sorted.sort();
            assert_eq!(keys, sorted);
        }
    }
}
