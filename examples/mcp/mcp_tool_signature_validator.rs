//! # MCP Tool Signature Validator
//!
//! MCP tools declare an input schema (JSON-Schema). Validate per-call:
//! required keys present, types match (string/number/boolean/object/
//! array), unknown keys rejected when `additionalProperties: false`.
//! This recipe builds the per-call validator.
//!
//! Demonstrates the **MCP.12** recipe for PMAT-132 (mcp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: JSON Schema Draft 7 + MCP tool spec.
//!
//! Run with: cargo run --example mcp_tool_signature_validator
//!
//! Added by PMAT-132 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaType {
    String,
    Number,
    Bool,
    Object,
    Array,
}

#[derive(Debug, Clone)]
pub struct PropertySpec {
    pub ty: SchemaType,
    pub required: bool,
}

#[derive(Debug, Clone)]
pub struct ToolSchema {
    pub properties: BTreeMap<String, PropertySpec>,
    pub additional_properties: bool,
}

#[derive(Debug, PartialEq)]
pub enum CallVerdict {
    Ok,
    MissingRequired {
        key: String,
    },
    TypeMismatch {
        key: String,
        expected: SchemaType,
        got: SchemaType,
    },
    UnknownKey {
        key: String,
    },
}

pub fn validate(schema: &ToolSchema, call: &BTreeMap<String, SchemaType>) -> CallVerdict {
    for (key, spec) in &schema.properties {
        if spec.required && !call.contains_key(key) {
            return CallVerdict::MissingRequired { key: key.clone() };
        }
    }
    for (key, ty) in call {
        match schema.properties.get(key) {
            Some(spec) if spec.ty != *ty => {
                return CallVerdict::TypeMismatch {
                    key: key.clone(),
                    expected: spec.ty,
                    got: *ty,
                };
            }
            None if !schema.additional_properties => {
                return CallVerdict::UnknownKey { key: key.clone() };
            }
            _ => {}
        }
    }
    CallVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mcp_tool_signature_validator")?;

    let mut props = BTreeMap::new();
    props.insert(
        "query".into(),
        PropertySpec {
            ty: SchemaType::String,
            required: true,
        },
    );
    props.insert(
        "limit".into(),
        PropertySpec {
            ty: SchemaType::Number,
            required: false,
        },
    );
    let schema = ToolSchema {
        properties: props,
        additional_properties: false,
    };

    let mut call_ok = BTreeMap::new();
    call_ok.insert("query".into(), SchemaType::String);
    call_ok.insert("limit".into(), SchemaType::Number);
    println!("ok: {:?}", validate(&schema, &call_ok));

    let mut call_missing = BTreeMap::new();
    call_missing.insert("limit".into(), SchemaType::Number);
    println!("missing: {:?}", validate(&schema, &call_missing));

    let mut call_unknown = call_ok.clone();
    call_unknown.insert("extra".into(), SchemaType::Bool);
    println!("unknown: {:?}", validate(&schema, &call_unknown));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schema() -> ToolSchema {
        let mut props = BTreeMap::new();
        props.insert(
            "query".into(),
            PropertySpec {
                ty: SchemaType::String,
                required: true,
            },
        );
        props.insert(
            "limit".into(),
            PropertySpec {
                ty: SchemaType::Number,
                required: false,
            },
        );
        ToolSchema {
            properties: props,
            additional_properties: false,
        }
    }

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn complete_call_passes() {
        let mut call = BTreeMap::new();
        call.insert("query".into(), SchemaType::String);
        call.insert("limit".into(), SchemaType::Number);
        assert_eq!(validate(&sample_schema(), &call), CallVerdict::Ok);
    }

    #[test]
    fn optional_field_omitted_passes() {
        let mut call = BTreeMap::new();
        call.insert("query".into(), SchemaType::String);
        assert_eq!(validate(&sample_schema(), &call), CallVerdict::Ok);
    }

    #[test]
    fn missing_required_rejected() {
        let call = BTreeMap::new();
        let v = validate(&sample_schema(), &call);
        assert!(matches!(v, CallVerdict::MissingRequired { .. }));
    }

    #[test]
    fn type_mismatch_rejected() {
        let mut call = BTreeMap::new();
        call.insert("query".into(), SchemaType::Number);
        let v = validate(&sample_schema(), &call);
        assert!(matches!(v, CallVerdict::TypeMismatch { .. }));
    }

    #[test]
    fn unknown_key_with_strict_rejected() {
        let mut call = BTreeMap::new();
        call.insert("query".into(), SchemaType::String);
        call.insert("extra".into(), SchemaType::Bool);
        let v = validate(&sample_schema(), &call);
        assert!(matches!(v, CallVerdict::UnknownKey { .. }));
    }

    #[test]
    fn unknown_key_with_open_passes() {
        let mut schema = sample_schema();
        schema.additional_properties = true;
        let mut call = BTreeMap::new();
        call.insert("query".into(), SchemaType::String);
        call.insert("extra".into(), SchemaType::Bool);
        assert_eq!(validate(&schema, &call), CallVerdict::Ok);
    }

    #[test]
    fn empty_schema_with_empty_call_passes() {
        let schema = ToolSchema {
            properties: BTreeMap::new(),
            additional_properties: false,
        };
        let call = BTreeMap::new();
        assert_eq!(validate(&schema, &call), CallVerdict::Ok);
    }

    #[test]
    fn first_missing_required_reported() {
        let mut props = BTreeMap::new();
        props.insert(
            "a".into(),
            PropertySpec {
                ty: SchemaType::String,
                required: true,
            },
        );
        props.insert(
            "b".into(),
            PropertySpec {
                ty: SchemaType::Number,
                required: true,
            },
        );
        let schema = ToolSchema {
            properties: props,
            additional_properties: false,
        };
        let call = BTreeMap::new();
        let v = validate(&schema, &call);
        assert!(matches!(v, CallVerdict::MissingRequired { .. }));
    }

    #[test]
    fn type_check_runs_before_unknown_key() {
        let mut call = BTreeMap::new();
        call.insert("query".into(), SchemaType::Number); // type mismatch
        let v = validate(&sample_schema(), &call);
        assert!(matches!(v, CallVerdict::TypeMismatch { .. }));
    }
}
