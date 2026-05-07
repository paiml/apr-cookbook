//! # Contracts-Macros YAML Anchor Alphabetize
//!
//! Verify YAML anchor names within each file are declared in
//! alphabetical order. Returns sorted offending files (out of order)
//! and the count of correctly-ordered files.
//!
//! Demonstrates the **CMM.181** recipe for PMAT-218 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GitLab CI/CD anchor-style guide; jq sort_keys recursive
//!  output mode.
//!
//! Run with: cargo run --example contracts_macros_yaml_anchor_alphabetize
//!
//! Added by PMAT-218 (catalog 1585→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AlphabetizeVerdict {
    Ok {
        offending_files: Vec<String>,
        ordered_count: u32,
    },
    InvalidConfig,
}

pub fn check(files: &[(&str, Vec<&str>)]) -> AlphabetizeVerdict {
    if files.is_empty() {
        return AlphabetizeVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = Vec::new();
    let mut ordered = 0u32;
    for (name, anchors) in files {
        if anchors.is_empty() {
            ordered += 1;
            continue;
        }
        let mut sorted = anchors.clone();
        sorted.sort_unstable();
        if &sorted == anchors {
            ordered += 1;
        } else {
            offenders.push((*name).to_string());
        }
    }
    offenders.sort();
    AlphabetizeVerdict::Ok {
        offending_files: offenders,
        ordered_count: ordered,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_anchor_alphabetize")?;

    let files = vec![
        ("a.yml", vec!["alpha", "beta", "gamma"]),
        ("b.yml", vec!["zeta", "alpha"]),
    ];
    println!("check: {:?}", check(&files));
    println!("invalid: {:?}", check(&[]));
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
    fn ordered_anchors_no_offender() {
        let files = vec![("f", vec!["a", "b", "c"])];
        let v = check(&files);
        if let AlphabetizeVerdict::Ok {
            offending_files, ..
        } = v
        {
            assert!(offending_files.is_empty());
        }
    }

    #[test]
    fn unordered_anchors_offender() {
        let files = vec![("f", vec!["b", "a"])];
        let v = check(&files);
        if let AlphabetizeVerdict::Ok {
            offending_files, ..
        } = v
        {
            assert_eq!(offending_files, vec!["f".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[]), AlphabetizeVerdict::InvalidConfig);
    }

    #[test]
    fn empty_anchors_treated_as_ordered() {
        let files: Vec<(&str, Vec<&str>)> = vec![("empty", vec![])];
        let v = check(&files);
        if let AlphabetizeVerdict::Ok { ordered_count, .. } = v {
            assert_eq!(ordered_count, 1);
        }
    }

    #[test]
    fn ordered_count_correct() {
        let files = vec![
            ("good", vec!["a", "b"]),
            ("bad", vec!["b", "a"]),
            ("good2", vec!["x", "y"]),
        ];
        let v = check(&files);
        if let AlphabetizeVerdict::Ok { ordered_count, .. } = v {
            assert_eq!(ordered_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let files = vec![("f", vec!["a"])];
        let r1 = check(&files);
        let r2 = check(&files);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted() {
        let files = vec![("zeta", vec!["b", "a"]), ("alpha", vec!["d", "c"])];
        let v = check(&files);
        if let AlphabetizeVerdict::Ok {
            offending_files, ..
        } = v
        {
            assert_eq!(
                offending_files,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn single_anchor_in_order() {
        let files = vec![("f", vec!["only"])];
        let v = check(&files);
        if let AlphabetizeVerdict::Ok {
            offending_files, ..
        } = v
        {
            assert!(offending_files.is_empty());
        }
    }

    #[test]
    fn case_sensitive_ordering() {
        // Capital letters sort before lowercase in ASCII.
        let files = vec![("f", vec!["a", "B"])];
        let v = check(&files);
        if let AlphabetizeVerdict::Ok {
            offending_files, ..
        } = v
        {
            assert_eq!(offending_files, vec!["f".to_string()]);
        }
    }

    #[test]
    fn many_files_handled() {
        let files: Vec<(&str, Vec<&str>)> = (0..30).map(|_| ("f", vec!["a", "b"])).collect();
        let v = check(&files);
        if let AlphabetizeVerdict::Ok { ordered_count, .. } = v {
            assert_eq!(ordered_count, 30);
        }
    }

    #[test]
    fn unicode_anchor_supported() {
        let files = vec![("f", vec!["café", "résumé"])];
        let v = check(&files);
        if let AlphabetizeVerdict::Ok {
            offending_files, ..
        } = v
        {
            assert!(offending_files.is_empty());
        }
    }
}
