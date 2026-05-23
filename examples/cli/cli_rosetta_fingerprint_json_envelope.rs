//! # apr rosetta fingerprint — `--output <FILE>` JSON Envelope
//!
//! `apr rosetta fingerprint <FILE> -o fingerprints.json` writes a stable
//! JSON envelope with a top-level `version`, `model`, and `tensors` map
//! keyed by tensor name. The version field is critical: validate-stats
//! must refuse to load fingerprints from a future schema, so the version
//! must be present and parseable.
//!
//! Demonstrates the **ROSETTA-FINGERPRINT.4** recipe for PMAT-097 (fingerprint coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-201 + JAX-STAT-001 + serde_json 1.x
//!
//! Run with: cargo run --example cli_rosetta_fingerprint_json_envelope
//!
//! Added by PMAT-097 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct StatTuple {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub l2_norm: f64,
}

pub const SCHEMA_VERSION: u32 = 1;

pub fn build_envelope(model: &str, tensors: &BTreeMap<String, StatTuple>) -> Value {
    json!({
        "version": SCHEMA_VERSION,
        "model": model,
        "tensors": tensors
            .iter()
            .map(|(k, t)| (k.clone(), json!({
                "mean": t.mean,
                "std": t.std,
                "min": t.min,
                "max": t.max,
                "l2_norm": t.l2_norm
            })))
            .collect::<serde_json::Map<_, _>>()
    })
}

pub fn validate_envelope(v: &Value) -> Result<u32> {
    let version = v.get("version").and_then(Value::as_u64).ok_or_else(|| {
        apr_cookbook::CookbookError::Validation("envelope missing version field".into())
    })?;
    if version > u64::from(SCHEMA_VERSION) {
        return Err(apr_cookbook::CookbookError::Validation(format!(
            "envelope version {version} > supported {SCHEMA_VERSION}"
        )));
    }
    if v.get("model").and_then(Value::as_str).is_none() {
        return Err(apr_cookbook::CookbookError::Validation(
            "envelope missing model field".into(),
        ));
    }
    if v.get("tensors").and_then(Value::as_object).is_none() {
        return Err(apr_cookbook::CookbookError::Validation(
            "envelope missing tensors object".into(),
        ));
    }
    Ok(version as u32)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_fingerprint_json_envelope")?;

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "embed_tokens".into(),
        StatTuple {
            mean: 0.0,
            std: 0.02,
            min: -0.1,
            max: 0.1,
            l2_norm: 12.5,
        },
    );
    let env = build_envelope("model.apr", &tensors);
    println!("{}", serde_json::to_string_pretty(&env).unwrap());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_envelope() -> Value {
        let mut t = BTreeMap::new();
        t.insert(
            "x".into(),
            StatTuple {
                mean: 0.0,
                std: 1.0,
                min: -3.0,
                max: 3.0,
                l2_norm: 5.0,
            },
        );
        build_envelope("m", &t)
    }

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn version_field_present() {
        let env = sample_envelope();
        assert_eq!(env["version"], json!(SCHEMA_VERSION));
    }

    #[test]
    fn validate_accepts_current_schema() {
        let env = sample_envelope();
        let v = validate_envelope(&env).unwrap();
        assert_eq!(v, SCHEMA_VERSION);
    }

    #[test]
    fn validate_rejects_future_schema() {
        // Refuse to load a fingerprint from a newer version — caller should
        // upgrade the binary, not silently skip unknown fields.
        let mut env = sample_envelope();
        env["version"] = json!(SCHEMA_VERSION + 99);
        assert!(validate_envelope(&env).is_err());
    }

    #[test]
    fn validate_rejects_missing_version() {
        let mut env = sample_envelope();
        env.as_object_mut().unwrap().remove("version");
        assert!(validate_envelope(&env).is_err());
    }

    #[test]
    fn validate_rejects_missing_model() {
        let mut env = sample_envelope();
        env.as_object_mut().unwrap().remove("model");
        assert!(validate_envelope(&env).is_err());
    }

    #[test]
    fn tensors_object_keyed_by_name() {
        let env = sample_envelope();
        let tensors = env["tensors"].as_object().unwrap();
        assert!(tensors.contains_key("x"));
        let entry = &tensors["x"];
        for key in ["mean", "std", "min", "max", "l2_norm"] {
            assert!(entry.get(key).is_some(), "missing key {key}");
        }
    }
}
