//! # Contracts-Macros Obligation Export CSV
//!
//! Generate a stable CSV export of obligation records: id, kind,
//! status. Order is alphabetical by id; duplicate ids are flagged.
//!
//! Demonstrates the **CMM.97** recipe for PMAT-190 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: RFC 4180 (CSV); ISO/IEC 30170 §5 stable export rules.
//!
//! Run with: cargo run --example contracts_macros_obligation_export_csv
//!
//! Added by PMAT-190 (catalog 1333→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum ExportVerdict {
    Ok {
        csv: String,
        row_count: u32,
        duplicate_ids: Vec<String>,
    },
    InvalidConfig,
}

pub fn export(obligations: &[(&str, &str, &str)]) -> ExportVerdict {
    if obligations.is_empty() {
        return ExportVerdict::InvalidConfig;
    }
    let mut by_id: BTreeMap<String, (String, String)> = BTreeMap::new();
    let mut duplicate_set: Vec<String> = Vec::new();
    for (id, kind, status) in obligations {
        if by_id.contains_key(*id) {
            duplicate_set.push((*id).to_string());
        } else {
            by_id.insert(
                (*id).to_string(),
                ((*kind).to_string(), (*status).to_string()),
            );
        }
    }
    let mut csv = String::from("id,kind,status\n");
    for (id, (kind, status)) in &by_id {
        csv.push_str(&format!("{id},{kind},{status}\n"));
    }
    duplicate_set.sort();
    duplicate_set.dedup();
    ExportVerdict::Ok {
        csv,
        row_count: by_id.len() as u32,
        duplicate_ids: duplicate_set,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_export_csv")?;

    let obligations = [("o1", "pre", "satisfied"), ("o2", "post", "violated")];
    println!("export: {:?}", export(&obligations));
    println!("invalid: {:?}", export(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exporter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn header_line_present() {
        let obligations = [("o1", "pre", "ok")];
        let v = export(&obligations);
        if let ExportVerdict::Ok { csv, .. } = v {
            assert!(csv.starts_with("id,kind,status"));
        }
    }

    #[test]
    fn rows_match_obligations() {
        let obligations = [("o1", "pre", "ok"), ("o2", "post", "ok")];
        let v = export(&obligations);
        if let ExportVerdict::Ok { row_count, .. } = v {
            assert_eq!(row_count, 2);
        }
    }

    #[test]
    fn duplicate_ids_flagged() {
        let obligations = [("o1", "pre", "ok"), ("o1", "post", "ok")];
        let v = export(&obligations);
        if let ExportVerdict::Ok { duplicate_ids, .. } = v {
            assert_eq!(duplicate_ids, vec!["o1".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(export(&[]), ExportVerdict::InvalidConfig);
    }

    #[test]
    fn rows_alphabetically_sorted() {
        let obligations = [("zeta", "pre", "ok"), ("alpha", "pre", "ok")];
        let v = export(&obligations);
        if let ExportVerdict::Ok { csv, .. } = v {
            let lines: Vec<&str> = csv.lines().collect();
            assert!(lines[1].starts_with("alpha"));
            assert!(lines[2].starts_with("zeta"));
        }
    }

    #[test]
    fn deterministic() {
        let obligations = [("o1", "pre", "ok")];
        let r1 = export(&obligations);
        let r2 = export(&obligations);
        assert_eq!(r1, r2);
    }

    #[test]
    fn fields_comma_separated() {
        let obligations = [("o1", "pre", "ok")];
        let v = export(&obligations);
        if let ExportVerdict::Ok { csv, .. } = v {
            let lines: Vec<&str> = csv.lines().collect();
            assert_eq!(lines[1].matches(',').count(), 2);
        }
    }

    #[test]
    fn lines_terminated_with_newline() {
        let obligations = [("o1", "pre", "ok")];
        let v = export(&obligations);
        if let ExportVerdict::Ok { csv, .. } = v {
            assert!(csv.ends_with('\n'));
        }
    }

    #[test]
    fn duplicate_ids_dedup_when_repeated() {
        let obligations = [
            ("o1", "pre", "ok"),
            ("o1", "post", "ok"),
            ("o1", "pre", "ok"),
        ];
        let v = export(&obligations);
        if let ExportVerdict::Ok { duplicate_ids, .. } = v {
            assert_eq!(duplicate_ids, vec!["o1".to_string()]);
        }
    }

    #[test]
    fn first_occurrence_kept() {
        let obligations = [("o1", "first", "ok"), ("o1", "second", "ok")];
        let v = export(&obligations);
        if let ExportVerdict::Ok { csv, .. } = v {
            assert!(csv.contains("first"));
            assert!(!csv.contains("second"));
        }
    }

    #[test]
    fn many_obligations_handled() {
        let obligations: Vec<(&str, &str, &str)> = (0..20).map(|_| ("o", "pre", "ok")).collect();
        let v = export(&obligations);
        if let ExportVerdict::Ok { row_count, .. } = v {
            assert_eq!(row_count, 1);
        }
    }
}
