//! # TUI Color Scheme Validate
//!
//! Verify a color scheme defines all required named colors (fg, bg,
//! error, warn, info, success). Returns missing names + ok-count.
//!
//! Demonstrates the **TUI.115** recipe for PMAT-198 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: VS Code theme color contributions; tmux color theme
//!  conventions.
//!
//! Run with: cargo run --example tui_color_scheme_validate
//!
//! Added by PMAT-198 (catalog 1405→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum SchemeVerdict {
    Ok { missing: Vec<String>, ok_count: u32 },
    InvalidConfig,
}

pub fn validate(required: &[&str], defined: &[&str]) -> SchemeVerdict {
    if required.is_empty() {
        return SchemeVerdict::InvalidConfig;
    }
    let def_set: BTreeSet<&str> = defined.iter().copied().collect();
    let mut missing: Vec<String> = required
        .iter()
        .filter(|r| !def_set.contains(*r))
        .map(|r| (*r).to_string())
        .collect();
    missing.sort();
    missing.dedup();
    let req_set: BTreeSet<&str> = required.iter().copied().collect();
    let ok_count = (req_set.len() - missing.len()) as u32;
    SchemeVerdict::Ok { missing, ok_count }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_color_scheme_validate")?;

    let required = ["fg", "bg", "error", "warn", "info"];
    let defined = ["fg", "bg", "error", "info"];
    println!("missing warn: {:?}", validate(&required, &defined));
    println!("invalid: {:?}", validate(&[], &defined));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn complete_scheme_no_missing() {
        let v = validate(&["fg", "bg"], &["fg", "bg"]);
        if let SchemeVerdict::Ok { missing, .. } = v {
            assert!(missing.is_empty());
        }
    }

    #[test]
    fn missing_color_flagged() {
        let v = validate(&["fg", "bg", "error"], &["fg", "bg"]);
        if let SchemeVerdict::Ok { missing, .. } = v {
            assert_eq!(missing, vec!["error".to_string()]);
        }
    }

    #[test]
    fn empty_required_rejected() {
        assert_eq!(validate(&[], &["fg"]), SchemeVerdict::InvalidConfig);
    }

    #[test]
    fn extra_defined_ignored() {
        let v = validate(&["fg"], &["fg", "bg", "extra"]);
        if let SchemeVerdict::Ok { missing, .. } = v {
            assert!(missing.is_empty());
        }
    }

    #[test]
    fn ok_count_correct() {
        let v = validate(&["fg", "bg", "error"], &["fg", "bg"]);
        if let SchemeVerdict::Ok { ok_count, .. } = v {
            assert_eq!(ok_count, 2);
        }
    }

    #[test]
    fn missing_sorted() {
        let v = validate(&["zeta", "alpha"], &[]);
        if let SchemeVerdict::Ok { missing, .. } = v {
            assert_eq!(missing, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = validate(&["fg"], &["fg"]);
        let r2 = validate(&["fg"], &["fg"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn case_sensitive() {
        let v = validate(&["FG"], &["fg"]);
        if let SchemeVerdict::Ok { missing, .. } = v {
            assert_eq!(missing, vec!["FG".to_string()]);
        }
    }

    #[test]
    fn duplicate_required_dedup() {
        let v = validate(&["fg", "fg", "bg"], &["fg"]);
        if let SchemeVerdict::Ok { missing, .. } = v {
            assert_eq!(missing, vec!["bg".to_string()]);
        }
    }

    #[test]
    fn empty_defined_all_missing() {
        let v = validate(&["fg", "bg"], &[]);
        if let SchemeVerdict::Ok { missing, .. } = v {
            assert_eq!(missing.len(), 2);
        }
    }

    #[test]
    fn many_colors_handled() {
        let req: Vec<&str> = vec!["c"; 20];
        let v = validate(&req, &["c"]);
        if let SchemeVerdict::Ok { missing, .. } = v {
            assert!(missing.is_empty());
        }
    }
}
