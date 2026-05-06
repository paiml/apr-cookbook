//! # Contracts-Macros YAML Max Nesting Depth
//!
//! Measure max indent-derived nesting depth across YAML lines.
//! Flag files exceeding `max_safe_depth` for refactoring.
//!
//! Demonstrates the **CMM.110** recipe for PMAT-194 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: deeply-nested-config anti-pattern (Hightower
//!  "Configuration Done Right", 2015).
//!
//! Run with: cargo run --example contracts_macros_yaml_max_nesting_depth
//!
//! Added by PMAT-194 (catalog 1369→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DepthVerdict {
    Ok { max_depth: u32, too_deep: bool },
    InvalidConfig,
}

pub fn measure(lines: &[&str], indent_size: u32, max_safe_depth: u32) -> DepthVerdict {
    if lines.is_empty() || indent_size == 0 || max_safe_depth == 0 {
        return DepthVerdict::InvalidConfig;
    }
    let mut max_depth = 0u32;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let leading: u32 = line.chars().take_while(|c| *c == ' ').count() as u32;
        let depth = leading / indent_size;
        if depth > max_depth {
            max_depth = depth;
        }
    }
    DepthVerdict::Ok {
        max_depth,
        too_deep: max_depth > max_safe_depth,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_max_nesting_depth")?;

    let lines = [
        "root:",
        "  level1:",
        "    level2:",
        "      level3:",
        "        level4: value",
    ];
    println!("audit: {:?}", measure(&lines, 2, 3));
    println!("invalid: {:?}", measure(&[], 2, 3));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measurer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn shallow_no_flag() {
        let lines = ["root:", "  child: value"];
        let v = measure(&lines, 2, 5);
        if let DepthVerdict::Ok { too_deep, .. } = v {
            assert!(!too_deep);
        }
    }

    #[test]
    fn deep_flagged() {
        let lines = ["a:", "  b:", "    c:", "      d:", "        e: v"];
        let v = measure(&lines, 2, 2);
        if let DepthVerdict::Ok { too_deep, .. } = v {
            assert!(too_deep);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(measure(&[], 2, 5), DepthVerdict::InvalidConfig);
    }

    #[test]
    fn zero_indent_rejected() {
        assert_eq!(measure(&["a"], 0, 5), DepthVerdict::InvalidConfig);
    }

    #[test]
    fn zero_max_safe_rejected() {
        assert_eq!(measure(&["a"], 2, 0), DepthVerdict::InvalidConfig);
    }

    #[test]
    fn max_depth_correct() {
        let lines = ["a:", "  b:", "    c:"];
        let v = measure(&lines, 2, 5);
        if let DepthVerdict::Ok { max_depth, .. } = v {
            assert_eq!(max_depth, 2);
        }
    }

    #[test]
    fn deterministic() {
        let lines = ["a:"];
        let r1 = measure(&lines, 2, 5);
        let r2 = measure(&lines, 2, 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn blank_lines_skipped() {
        let lines = ["root:", "", "    deep: v"];
        let v = measure(&lines, 2, 5);
        if let DepthVerdict::Ok { max_depth, .. } = v {
            assert_eq!(max_depth, 2);
        }
    }

    #[test]
    fn boundary_at_max_safe_no_flag() {
        let lines = ["a:", "  b:", "    c:"];
        let v = measure(&lines, 2, 2);
        if let DepthVerdict::Ok { too_deep, .. } = v {
            assert!(!too_deep);
        }
    }

    #[test]
    fn one_over_max_flagged() {
        let lines = ["a:", "  b:", "    c:", "      d:"];
        let v = measure(&lines, 2, 2);
        if let DepthVerdict::Ok { too_deep, .. } = v {
            assert!(too_deep);
        }
    }

    #[test]
    fn no_indent_zero_depth() {
        let lines = ["root: v"];
        let v = measure(&lines, 2, 5);
        if let DepthVerdict::Ok { max_depth, .. } = v {
            assert_eq!(max_depth, 0);
        }
    }

    #[test]
    fn four_space_indent_supported() {
        let lines = ["a:", "    b:", "        c:"];
        let v = measure(&lines, 4, 5);
        if let DepthVerdict::Ok { max_depth, .. } = v {
            assert_eq!(max_depth, 2);
        }
    }
}
