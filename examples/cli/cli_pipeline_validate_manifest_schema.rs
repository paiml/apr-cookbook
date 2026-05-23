//! # apr pipeline validate — Manifest Schema Validator
//!
//! `apr pipeline validate <MANIFEST>` checks the manifest YAML without
//! connecting to any resources. This recipe encodes the manifest schema
//! as a pure validator: top-level `apiVersion`, `kind`, `name`, `spec`
//! all required; spec must contain `resources` array; each resource must
//! have `id`, `type`, `depends_on` (possibly empty).
//!
//! Demonstrates the **PIPELINE.15** recipe for PMAT-107 (apr pipeline coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PIPELINE-003 + Kubernetes-style manifest convention
//!
//! Run with: cargo run --example cli_pipeline_validate_manifest_schema
//!
//! Added by PMAT-107 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::Value;

#[derive(Debug, PartialEq)]
pub enum ValidateVerdict {
    Ok,
    MissingTopLevelKey(&'static str),
    SpecNotObject,
    ResourcesNotArray,
    ResourceMissingKey { index: usize, key: &'static str },
    InvalidApiVersion(String),
}

const SUPPORTED_API_VERSIONS: &[&str] = &["apr.paiml.com/v1", "apr.paiml.com/v1beta1"];

pub fn validate_manifest(manifest: &Value) -> ValidateVerdict {
    for key in ["apiVersion", "kind", "name", "spec"] {
        if manifest.get(key).is_none() {
            return ValidateVerdict::MissingTopLevelKey(match key {
                "apiVersion" => "apiVersion",
                "kind" => "kind",
                "name" => "name",
                "spec" => "spec",
                _ => "unknown",
            });
        }
    }

    let api_version = manifest
        .get("apiVersion")
        .and_then(Value::as_str)
        .unwrap_or("");
    if !SUPPORTED_API_VERSIONS.contains(&api_version) {
        return ValidateVerdict::InvalidApiVersion(api_version.into());
    }

    let Some(spec) = manifest.get("spec").and_then(Value::as_object) else {
        return ValidateVerdict::SpecNotObject;
    };

    let Some(resources) = spec.get("resources").and_then(Value::as_array) else {
        return ValidateVerdict::ResourcesNotArray;
    };

    for (i, r) in resources.iter().enumerate() {
        for key in ["id", "type", "depends_on"] {
            if r.get(key).is_none() {
                return ValidateVerdict::ResourceMissingKey {
                    index: i,
                    key: match key {
                        "id" => "id",
                        "type" => "type",
                        "depends_on" => "depends_on",
                        _ => "unknown",
                    },
                };
            }
        }
    }

    ValidateVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pipeline_validate_manifest_schema")?;

    let happy = serde_json::json!({
        "apiVersion": "apr.paiml.com/v1",
        "kind": "Pipeline",
        "name": "training-pipeline",
        "spec": {
            "resources": [
                { "id": "download", "type": "data.download", "depends_on": [] },
                { "id": "train", "type": "model.train", "depends_on": ["download"] }
            ]
        }
    });
    println!("happy:           {:?}", validate_manifest(&happy));

    let missing_kind = serde_json::json!({
        "apiVersion": "apr.paiml.com/v1",
        "name": "p",
        "spec": { "resources": [] }
    });
    println!("missing kind:    {:?}", validate_manifest(&missing_kind));

    let bad_api = serde_json::json!({
        "apiVersion": "v999",
        "kind": "Pipeline",
        "name": "p",
        "spec": { "resources": [] }
    });
    println!("bad api version: {:?}", validate_manifest(&bad_api));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn happy_manifest() -> Value {
        serde_json::json!({
            "apiVersion": "apr.paiml.com/v1",
            "kind": "Pipeline",
            "name": "p",
            "spec": {
                "resources": [
                    { "id": "r1", "type": "t1", "depends_on": [] }
                ]
            }
        })
    }

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_manifest_passes() {
        assert_eq!(validate_manifest(&happy_manifest()), ValidateVerdict::Ok);
    }

    #[test]
    fn missing_top_level_key_rejected() {
        let mut m = happy_manifest();
        m.as_object_mut().unwrap().remove("kind");
        assert!(matches!(
            validate_manifest(&m),
            ValidateVerdict::MissingTopLevelKey("kind")
        ));
    }

    #[test]
    fn beta_api_version_accepted() {
        let mut m = happy_manifest();
        m["apiVersion"] = serde_json::json!("apr.paiml.com/v1beta1");
        assert_eq!(validate_manifest(&m), ValidateVerdict::Ok);
    }

    #[test]
    fn unsupported_api_version_rejected() {
        let mut m = happy_manifest();
        m["apiVersion"] = serde_json::json!("apr.paiml.com/v999");
        let v = validate_manifest(&m);
        assert!(matches!(v, ValidateVerdict::InvalidApiVersion(_)));
    }

    #[test]
    fn spec_not_object_rejected() {
        let mut m = happy_manifest();
        m["spec"] = serde_json::json!("not-an-object");
        let v = validate_manifest(&m);
        assert!(matches!(v, ValidateVerdict::SpecNotObject));
    }

    #[test]
    fn resources_not_array_rejected() {
        let mut m = happy_manifest();
        m["spec"]["resources"] = serde_json::json!("not-an-array");
        assert!(matches!(
            validate_manifest(&m),
            ValidateVerdict::ResourcesNotArray
        ));
    }

    #[test]
    fn resource_missing_key_rejected() {
        let mut m = happy_manifest();
        let resources = m["spec"]["resources"].as_array_mut().unwrap();
        resources[0].as_object_mut().unwrap().remove("type");
        let v = validate_manifest(&m);
        assert!(matches!(
            v,
            ValidateVerdict::ResourceMissingKey {
                index: 0,
                key: "type"
            }
        ));
    }

    #[test]
    fn empty_resources_array_passes() {
        let mut m = happy_manifest();
        m["spec"]["resources"] = serde_json::json!([]);
        assert_eq!(validate_manifest(&m), ValidateVerdict::Ok);
    }
}
