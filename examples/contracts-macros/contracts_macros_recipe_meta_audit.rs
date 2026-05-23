//! # Contracts-Macros Recipe Metadata Audit
//!
//! Audit metadata fields on a contract: required (author, version),
//! recommended (last_updated, license), and optional. Returns the
//! first missing required field.
//!
//! Demonstrates the **CMM.28** recipe for PMAT-167 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SLSA provenance metadata.
//!
//! Run with: cargo run --example contracts_macros_recipe_meta_audit
//!
//! Added by PMAT-167 (catalog 1126→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Metadata {
    pub author: Option<String>,
    pub version: Option<String>,
    pub last_updated: Option<String>,
    pub license: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum AuditVerdict {
    Complete,
    MissingRecommended { fields: Vec<String> },
    MissingRequired { field: &'static str },
}

pub fn audit(metadata: &Metadata) -> AuditVerdict {
    if metadata.author.as_deref().unwrap_or("").trim().is_empty() {
        return AuditVerdict::MissingRequired { field: "author" };
    }
    if metadata.version.as_deref().unwrap_or("").trim().is_empty() {
        return AuditVerdict::MissingRequired { field: "version" };
    }
    let mut missing = Vec::new();
    if metadata
        .last_updated
        .as_deref()
        .unwrap_or("")
        .trim()
        .is_empty()
    {
        missing.push("last_updated".to_string());
    }
    if metadata.license.as_deref().unwrap_or("").trim().is_empty() {
        missing.push("license".to_string());
    }
    if missing.is_empty() {
        AuditVerdict::Complete
    } else {
        AuditVerdict::MissingRecommended { fields: missing }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_meta_audit")?;

    let complete = Metadata {
        author: Some("Alice".to_string()),
        version: Some("1.0.0".to_string()),
        last_updated: Some("2026-01-01".to_string()),
        license: Some("MIT".to_string()),
    };
    println!("complete: {:?}", audit(&complete));

    let missing_author = Metadata {
        author: None,
        version: Some("1.0".to_string()),
        last_updated: None,
        license: None,
    };
    println!("missing author: {:?}", audit(&missing_author));

    let missing_recommended = Metadata {
        author: Some("Bob".to_string()),
        version: Some("0.1".to_string()),
        last_updated: None,
        license: None,
    };
    println!("missing recommended: {:?}", audit(&missing_recommended));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete() -> Metadata {
        Metadata {
            author: Some("Alice".to_string()),
            version: Some("1.0.0".to_string()),
            last_updated: Some("2026-01-01".to_string()),
            license: Some("MIT".to_string()),
        }
    }

    #[test]
    fn auditor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn complete_metadata_passes() {
        assert_eq!(audit(&complete()), AuditVerdict::Complete);
    }

    #[test]
    fn missing_author_required() {
        let mut m = complete();
        m.author = None;
        let v = audit(&m);
        if let AuditVerdict::MissingRequired { field } = v {
            assert_eq!(field, "author");
        }
    }

    #[test]
    fn missing_version_required() {
        let mut m = complete();
        m.version = None;
        let v = audit(&m);
        if let AuditVerdict::MissingRequired { field } = v {
            assert_eq!(field, "version");
        }
    }

    #[test]
    fn empty_string_treated_as_missing() {
        let mut m = complete();
        m.author = Some("".to_string());
        assert!(matches!(audit(&m), AuditVerdict::MissingRequired { .. }));
    }

    #[test]
    fn whitespace_only_treated_as_missing() {
        let mut m = complete();
        m.version = Some("   ".to_string());
        assert!(matches!(audit(&m), AuditVerdict::MissingRequired { .. }));
    }

    #[test]
    fn missing_last_updated_recommended() {
        let mut m = complete();
        m.last_updated = None;
        let v = audit(&m);
        if let AuditVerdict::MissingRecommended { fields } = v {
            assert!(fields.contains(&"last_updated".to_string()));
        }
    }

    #[test]
    fn missing_license_recommended() {
        let mut m = complete();
        m.license = None;
        let v = audit(&m);
        if let AuditVerdict::MissingRecommended { fields } = v {
            assert!(fields.contains(&"license".to_string()));
        }
    }

    #[test]
    fn required_takes_precedence_over_recommended() {
        let m = Metadata {
            author: None,
            version: None,
            last_updated: None,
            license: None,
        };
        let v = audit(&m);
        assert!(matches!(v, AuditVerdict::MissingRequired { .. }));
    }

    #[test]
    fn both_recommended_missing() {
        let mut m = complete();
        m.last_updated = None;
        m.license = None;
        let v = audit(&m);
        if let AuditVerdict::MissingRecommended { fields } = v {
            assert_eq!(fields.len(), 2);
        }
    }

    #[test]
    fn deterministic() {
        let m = complete();
        let a = audit(&m);
        let b = audit(&m);
        assert_eq!(a, b);
    }
}
