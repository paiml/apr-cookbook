//! # WASM Custom Section Emit
//!
//! Verify that required custom sections are present in the module
//! (e.g., `producers`, `name`, `target_features`). Returns sorted
//! missing sections.
//!
//! Demonstrates the **WASM.X** recipe for PMAT-222 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly Core §5.5.3 custom sections; producers
//!  section spec (toolchain attribution).
//!
//! Run with: cargo run --example wasm_custom_section_emit
//!
//! Added by PMAT-222 (catalog 1621→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum CustomSectionVerdict {
    Ok {
        missing_sections: Vec<String>,
        present_count: u32,
    },
    InvalidConfig,
}

pub fn check(present: &[&str], required: &[&str]) -> CustomSectionVerdict {
    if required.is_empty() {
        return CustomSectionVerdict::InvalidConfig;
    }
    let present_set: BTreeSet<&str> = present.iter().copied().collect();
    let missing: BTreeSet<String> = required
        .iter()
        .filter(|s| !present_set.contains(*s))
        .map(|s| (*s).to_string())
        .collect();
    CustomSectionVerdict::Ok {
        missing_sections: missing.into_iter().collect(),
        present_count: present_set.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_custom_section_emit")?;

    let required = ["producers", "name", "target_features"];
    let present = ["name", "target_features"];
    println!("check: {:?}", check(&present, &required));
    println!("invalid: {:?}", check(&[], &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn all_present_no_missing() {
        let v = check(&["a", "b"], &["a", "b"]);
        if let CustomSectionVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert!(missing_sections.is_empty());
        }
    }

    #[test]
    fn missing_section_flagged() {
        let v = check(&["a"], &["a", "b"]);
        if let CustomSectionVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert_eq!(missing_sections, vec!["b".to_string()]);
        }
    }

    #[test]
    fn empty_required_rejected() {
        assert_eq!(check(&["a"], &[]), CustomSectionVerdict::InvalidConfig);
    }

    #[test]
    fn empty_present_all_missing() {
        let v = check(&[], &["a", "b"]);
        if let CustomSectionVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert_eq!(missing_sections.len(), 2);
        }
    }

    #[test]
    fn extra_sections_ok() {
        let v = check(&["a", "b", "extra"], &["a"]);
        if let CustomSectionVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert!(missing_sections.is_empty());
        }
    }

    #[test]
    fn present_count_correct() {
        let v = check(&["a", "b"], &["a"]);
        if let CustomSectionVerdict::Ok { present_count, .. } = v {
            assert_eq!(present_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&["a"], &["a"]);
        let r2 = check(&["a"], &["a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn missing_sorted() {
        let v = check(&[], &["zeta", "alpha"]);
        if let CustomSectionVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert_eq!(
                missing_sections,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn duplicate_present_dedup() {
        let v = check(&["a", "a"], &["a"]);
        if let CustomSectionVerdict::Ok { present_count, .. } = v {
            assert_eq!(present_count, 1);
        }
    }

    #[test]
    fn case_sensitive_section_name() {
        let v = check(&["Name"], &["name"]);
        if let CustomSectionVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert_eq!(missing_sections, vec!["name".to_string()]);
        }
    }

    #[test]
    fn many_required_handled() {
        let required: Vec<&str> = (0..20).map(|_| "req").collect();
        let v = check(&[], &required);
        if let CustomSectionVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert_eq!(missing_sections.len(), 1); // BTreeSet dedupes
        }
    }

    #[test]
    fn unicode_section_supported() {
        let v = check(&["café"], &["café"]);
        if let CustomSectionVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert!(missing_sections.is_empty());
        }
    }
}
