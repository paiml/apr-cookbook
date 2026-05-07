//! # Contracts-Macros YAML Field Alphabetize
//!
//! Verify YAML keys are alphabetized within each section. Returns
//! offending sections with their key list.
//!
//! Demonstrates the **CMM.122** recipe for PMAT-198 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: prettier YAML key sort plugin; jq sort_keys output mode.
//!
//! Run with: cargo run --example contracts_macros_yaml_field_alphabetize
//!
//! Added by PMAT-198 (catalog 1405→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AlphaVerdict {
    Ok {
        offending_sections: Vec<String>,
        sorted_count: u32,
    },
    InvalidConfig,
}

pub fn audit(sections: &[(&str, Vec<&str>)]) -> AlphaVerdict {
    if sections.is_empty() {
        return AlphaVerdict::InvalidConfig;
    }
    let mut offending: Vec<String> = Vec::new();
    let mut sorted_count = 0u32;
    for (name, keys) in sections {
        if keys.is_empty() {
            continue;
        }
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        if &sorted == keys {
            sorted_count += 1;
        } else {
            offending.push((*name).to_string());
        }
    }
    offending.sort();
    AlphaVerdict::Ok {
        offending_sections: offending,
        sorted_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_field_alphabetize")?;

    let sections = vec![
        ("section_a", vec!["alpha", "beta", "gamma"]),
        ("section_b", vec!["zeta", "alpha"]),
    ];
    println!("audit: {:?}", audit(&sections));
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
    fn sorted_section_no_offender() {
        let sections = vec![("s", vec!["a", "b", "c"])];
        let v = audit(&sections);
        if let AlphaVerdict::Ok {
            offending_sections, ..
        } = v
        {
            assert!(offending_sections.is_empty());
        }
    }

    #[test]
    fn unsorted_section_flagged() {
        let sections = vec![("s", vec!["b", "a"])];
        let v = audit(&sections);
        if let AlphaVerdict::Ok {
            offending_sections, ..
        } = v
        {
            assert_eq!(offending_sections, vec!["s".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), AlphaVerdict::InvalidConfig);
    }

    #[test]
    fn empty_section_skipped() {
        let sections: Vec<(&str, Vec<&str>)> = vec![("empty", vec![])];
        let v = audit(&sections);
        if let AlphaVerdict::Ok {
            offending_sections, ..
        } = v
        {
            assert!(offending_sections.is_empty());
        }
    }

    #[test]
    fn sorted_count_correct() {
        let sections = vec![
            ("good", vec!["a", "b"]),
            ("bad", vec!["b", "a"]),
            ("good2", vec!["x", "y"]),
        ];
        let v = audit(&sections);
        if let AlphaVerdict::Ok { sorted_count, .. } = v {
            assert_eq!(sorted_count, 2);
        }
    }

    #[test]
    fn offending_sorted() {
        let sections = vec![("zeta", vec!["b", "a"]), ("alpha", vec!["d", "c"])];
        let v = audit(&sections);
        if let AlphaVerdict::Ok {
            offending_sections, ..
        } = v
        {
            assert_eq!(
                offending_sections,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn deterministic() {
        let sections = vec![("s", vec!["a", "b"])];
        let r1 = audit(&sections);
        let r2 = audit(&sections);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_key_section_passes() {
        let sections = vec![("s", vec!["only"])];
        let v = audit(&sections);
        if let AlphaVerdict::Ok {
            offending_sections, ..
        } = v
        {
            assert!(offending_sections.is_empty());
        }
    }

    #[test]
    fn case_sensitive_sort() {
        // Capital letters sort before lowercase in ASCII.
        let sections = vec![("s", vec!["a", "B"])];
        let v = audit(&sections);
        if let AlphaVerdict::Ok {
            offending_sections, ..
        } = v
        {
            assert_eq!(offending_sections, vec!["s".to_string()]);
        }
    }

    #[test]
    fn sections_count_matches_input() {
        let sections = vec![("a", vec!["x"]), ("b", vec!["y"])];
        let v = audit(&sections);
        if let AlphaVerdict::Ok { sorted_count, .. } = v {
            assert_eq!(sorted_count, 2);
        }
    }

    #[test]
    fn many_sections_handled() {
        let sections: Vec<(&str, Vec<&str>)> = (0..20).map(|_| ("s", vec!["a", "b"])).collect();
        let v = audit(&sections);
        if let AlphaVerdict::Ok { sorted_count, .. } = v {
            assert_eq!(sorted_count, 20);
        }
    }
}
