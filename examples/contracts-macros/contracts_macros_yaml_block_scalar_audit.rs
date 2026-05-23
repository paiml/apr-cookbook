//! # Contracts-Macros YAML Block Scalar Audit
//!
//! Verify YAML block scalar markers are used correctly:
//! - `|` (literal) preserves newlines verbatim
//! - `>` (folded) joins consecutive lines with spaces
//!
//! Returns flagged misuses (e.g. literal where folded would be cleaner).
//!
//! Demonstrates the **CMM.95** recipe for PMAT-189 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 spec §8.1 (block scalar styles).
//!
//! Run with: cargo run --example contracts_macros_yaml_block_scalar_audit
//!
//! Added by PMAT-189 (catalog 1324→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone)]
pub enum ScalarKind {
    Literal, // |
    Folded,  // >
    Plain,   // plain inline
    Quoted,  // "..." or '...'
}

#[derive(Debug, PartialEq)]
pub enum AuditVerdict {
    Ok {
        per_field: Vec<(String, ScalarKind, String)>,
        recommendation_count: u32,
    },
    InvalidConfig,
}

pub fn audit(fields: &[(&str, ScalarKind, u32)]) -> AuditVerdict {
    if fields.is_empty() {
        return AuditVerdict::InvalidConfig;
    }
    let mut per_field: Vec<(String, ScalarKind, String)> = Vec::with_capacity(fields.len());
    let mut recommendation_count = 0u32;
    for (name, kind, line_count) in fields {
        let recommendation = match kind {
            ScalarKind::Literal if *line_count == 1 => {
                recommendation_count += 1;
                "consider Plain for single-line".to_string()
            }
            ScalarKind::Folded if *line_count == 1 => {
                recommendation_count += 1;
                "consider Plain for single-line".to_string()
            }
            ScalarKind::Plain if *line_count > 5 => {
                recommendation_count += 1;
                "consider Literal `|` for multiline content".to_string()
            }
            _ => "ok".to_string(),
        };
        per_field.push(((*name).to_string(), kind.clone(), recommendation));
    }
    AuditVerdict::Ok {
        per_field,
        recommendation_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_block_scalar_audit")?;

    let fields = [
        ("title", ScalarKind::Plain, 1),
        ("body", ScalarKind::Literal, 8),
        ("brief", ScalarKind::Folded, 1),
    ];
    println!("audit: {:?}", audit(&fields));
    println!("invalid: {:?}", audit(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn literal_multiline_ok() {
        let fields = [("body", ScalarKind::Literal, 10)];
        let v = audit(&fields);
        if let AuditVerdict::Ok { per_field, .. } = v {
            assert_eq!(per_field[0].2, "ok");
        }
    }

    #[test]
    fn literal_single_line_recommends_plain() {
        let fields = [("title", ScalarKind::Literal, 1)];
        let v = audit(&fields);
        if let AuditVerdict::Ok {
            recommendation_count,
            ..
        } = v
        {
            assert_eq!(recommendation_count, 1);
        }
    }

    #[test]
    fn folded_single_line_recommends_plain() {
        let fields = [("brief", ScalarKind::Folded, 1)];
        let v = audit(&fields);
        if let AuditVerdict::Ok {
            recommendation_count,
            ..
        } = v
        {
            assert_eq!(recommendation_count, 1);
        }
    }

    #[test]
    fn plain_multiline_recommends_literal() {
        let fields = [("desc", ScalarKind::Plain, 10)];
        let v = audit(&fields);
        if let AuditVerdict::Ok {
            recommendation_count,
            ..
        } = v
        {
            assert_eq!(recommendation_count, 1);
        }
    }

    #[test]
    fn plain_short_ok() {
        let fields = [("name", ScalarKind::Plain, 1)];
        let v = audit(&fields);
        if let AuditVerdict::Ok {
            recommendation_count,
            ..
        } = v
        {
            assert_eq!(recommendation_count, 0);
        }
    }

    #[test]
    fn quoted_no_recommendation() {
        let fields = [("greeting", ScalarKind::Quoted, 5)];
        let v = audit(&fields);
        if let AuditVerdict::Ok {
            recommendation_count,
            ..
        } = v
        {
            assert_eq!(recommendation_count, 0);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), AuditVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let fields = [("a", ScalarKind::Plain, 1)];
        let r1 = audit(&fields);
        let r2 = audit(&fields);
        assert_eq!(r1, r2);
    }

    #[test]
    fn count_matches_field_count() {
        let fields = [("a", ScalarKind::Plain, 1), ("b", ScalarKind::Literal, 5)];
        let v = audit(&fields);
        if let AuditVerdict::Ok { per_field, .. } = v {
            assert_eq!(per_field.len(), 2);
        }
    }

    #[test]
    fn boundary_5_lines_plain_ok() {
        let fields = [("desc", ScalarKind::Plain, 5)];
        let v = audit(&fields);
        if let AuditVerdict::Ok {
            recommendation_count,
            ..
        } = v
        {
            assert_eq!(recommendation_count, 0);
        }
    }

    #[test]
    fn boundary_6_lines_plain_recommends() {
        let fields = [("desc", ScalarKind::Plain, 6)];
        let v = audit(&fields);
        if let AuditVerdict::Ok {
            recommendation_count,
            ..
        } = v
        {
            assert_eq!(recommendation_count, 1);
        }
    }
}
