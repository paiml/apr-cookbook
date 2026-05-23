//! # Lint JSON Schema Drift Detector
//!
//! Compare two flat schemas (field name → type) and classify changes:
//! AddedField (additive — usually safe), RemovedField (breaking),
//! TypeChange (breaking), Compatible (no change). Used in API contract
//! tests to fail-fast on accidental breaking changes.
//!
//! Demonstrates the **LINT.57** recipe for PMAT-131 (lint coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Open API Specification §schema-evolution best practices.
//!
//! Run with: cargo run --example lint_schema_drift_detector
//!
//! Added by PMAT-131 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    String,
    Number,
    Bool,
    Array,
    Object,
    Null,
}

#[derive(Debug, PartialEq, Eq)]
pub enum DriftChange {
    AddedField {
        name: String,
        ty: FieldType,
    },
    RemovedField {
        name: String,
        ty: FieldType,
    },
    TypeChange {
        name: String,
        from: FieldType,
        to: FieldType,
    },
}

#[derive(Debug, PartialEq)]
pub enum DriftSeverity {
    Compatible,
    Additive,
    Breaking,
}

#[derive(Debug, PartialEq)]
pub struct DriftReport {
    pub changes: Vec<DriftChange>,
    pub severity: DriftSeverity,
}

pub fn diff(prev: &BTreeMap<String, FieldType>, curr: &BTreeMap<String, FieldType>) -> DriftReport {
    let mut changes = Vec::new();
    let mut has_breaking = false;
    let mut has_additive = false;
    for (name, ty) in curr {
        match prev.get(name) {
            None => {
                changes.push(DriftChange::AddedField {
                    name: name.clone(),
                    ty: *ty,
                });
                has_additive = true;
            }
            Some(prev_ty) if prev_ty != ty => {
                changes.push(DriftChange::TypeChange {
                    name: name.clone(),
                    from: *prev_ty,
                    to: *ty,
                });
                has_breaking = true;
            }
            _ => {}
        }
    }
    for (name, ty) in prev {
        if !curr.contains_key(name) {
            changes.push(DriftChange::RemovedField {
                name: name.clone(),
                ty: *ty,
            });
            has_breaking = true;
        }
    }
    let severity = if has_breaking {
        DriftSeverity::Breaking
    } else if has_additive {
        DriftSeverity::Additive
    } else {
        DriftSeverity::Compatible
    };
    DriftReport { changes, severity }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("lint_schema_drift_detector")?;

    let mut prev = BTreeMap::new();
    prev.insert("id".into(), FieldType::Number);
    prev.insert("name".into(), FieldType::String);

    let mut curr_added = prev.clone();
    curr_added.insert("email".into(), FieldType::String);
    println!("added: {:?}", diff(&prev, &curr_added));

    let mut curr_removed = prev.clone();
    curr_removed.remove("name");
    println!("removed: {:?}", diff(&prev, &curr_removed));

    let mut curr_type = prev.clone();
    curr_type.insert("id".into(), FieldType::String);
    println!("type: {:?}", diff(&prev, &curr_type));

    println!("identical: {:?}", diff(&prev, &prev));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_prev() -> BTreeMap<String, FieldType> {
        let mut m = BTreeMap::new();
        m.insert("id".into(), FieldType::Number);
        m.insert("name".into(), FieldType::String);
        m
    }

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_schemas_compatible() {
        let r = diff(&sample_prev(), &sample_prev());
        assert_eq!(r.severity, DriftSeverity::Compatible);
        assert!(r.changes.is_empty());
    }

    #[test]
    fn added_field_additive() {
        let prev = sample_prev();
        let mut curr = prev.clone();
        curr.insert("email".into(), FieldType::String);
        let r = diff(&prev, &curr);
        assert_eq!(r.severity, DriftSeverity::Additive);
        assert_eq!(r.changes.len(), 1);
    }

    #[test]
    fn removed_field_breaking() {
        let prev = sample_prev();
        let mut curr = prev.clone();
        curr.remove("name");
        let r = diff(&prev, &curr);
        assert_eq!(r.severity, DriftSeverity::Breaking);
        assert!(r
            .changes
            .iter()
            .any(|c| matches!(c, DriftChange::RemovedField { .. })));
    }

    #[test]
    fn type_change_breaking() {
        let prev = sample_prev();
        let mut curr = prev.clone();
        curr.insert("id".into(), FieldType::String);
        let r = diff(&prev, &curr);
        assert_eq!(r.severity, DriftSeverity::Breaking);
    }

    #[test]
    fn mixed_changes_breaking_overrides_additive() {
        let prev = sample_prev();
        let mut curr = prev.clone();
        curr.insert("email".into(), FieldType::String);
        curr.remove("name");
        let r = diff(&prev, &curr);
        // Additive + Breaking → Breaking wins.
        assert_eq!(r.severity, DriftSeverity::Breaking);
    }

    #[test]
    fn empty_to_populated_additive() {
        let empty = BTreeMap::new();
        let r = diff(&empty, &sample_prev());
        assert_eq!(r.severity, DriftSeverity::Additive);
    }

    #[test]
    fn populated_to_empty_breaking() {
        let empty = BTreeMap::new();
        let r = diff(&sample_prev(), &empty);
        assert_eq!(r.severity, DriftSeverity::Breaking);
    }

    #[test]
    fn both_empty_compatible() {
        let empty = BTreeMap::new();
        let r = diff(&empty, &empty);
        assert_eq!(r.severity, DriftSeverity::Compatible);
    }

    #[test]
    fn change_count_matches_field_diffs() {
        let prev = sample_prev();
        let mut curr = prev.clone();
        curr.insert("a".into(), FieldType::Number);
        curr.insert("b".into(), FieldType::String);
        curr.remove("name");
        let r = diff(&prev, &curr);
        assert_eq!(r.changes.len(), 3);
    }
}
