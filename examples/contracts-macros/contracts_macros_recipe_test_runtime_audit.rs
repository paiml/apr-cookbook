//! # Contracts-Macros Recipe Test Runtime Audit
//!
//! Flag recipes whose test suite runtime exceeds `max_ms_per_test`
//! threshold. Returns offenders + percentile statistics.
//!
//! Demonstrates the **CMM.117** recipe for PMAT-196 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: criterion benchmark slow-warning convention; `cargo
//!  test --report-time` output.
//!
//! Run with: cargo run --example contracts_macros_recipe_test_runtime_audit
//!
//! Added by PMAT-196 (catalog 1387→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RuntimeVerdict {
    Ok {
        offenders: Vec<String>,
        median_ms: u32,
        p99_ms: u32,
    },
    InvalidConfig,
}

pub fn audit(recipes: &[(&str, u32)], max_ms_per_test: u32) -> RuntimeVerdict {
    if recipes.is_empty() || max_ms_per_test == 0 {
        return RuntimeVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = Vec::new();
    let mut times: Vec<u32> = Vec::with_capacity(recipes.len());
    for (name, ms) in recipes {
        if *ms > max_ms_per_test {
            offenders.push((*name).to_string());
        }
        times.push(*ms);
    }
    offenders.sort();
    times.sort_unstable();
    let median_ms = times[times.len() / 2];
    let p99_idx = (times.len() as f64 * 0.99) as usize;
    let p99_ms = times[p99_idx.min(times.len() - 1)];
    RuntimeVerdict::Ok {
        offenders,
        median_ms,
        p99_ms,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_test_runtime_audit")?;

    let recipes = [("fast", 5), ("ok", 20), ("slow", 200), ("verysslow", 5000)];
    println!("audit max=100ms: {:?}", audit(&recipes, 100));
    println!("invalid: {:?}", audit(&[], 100));
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
    fn fast_no_offenders() {
        let v = audit(&[("a", 5)], 100);
        if let RuntimeVerdict::Ok { offenders, .. } = v {
            assert!(offenders.is_empty());
        }
    }

    #[test]
    fn slow_flagged() {
        let v = audit(&[("a", 200)], 100);
        if let RuntimeVerdict::Ok { offenders, .. } = v {
            assert_eq!(offenders, vec!["a".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[], 100), RuntimeVerdict::InvalidConfig);
    }

    #[test]
    fn zero_max_rejected() {
        assert_eq!(audit(&[("a", 5)], 0), RuntimeVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_at_max_no_flag() {
        let v = audit(&[("a", 100)], 100);
        if let RuntimeVerdict::Ok { offenders, .. } = v {
            assert!(offenders.is_empty());
        }
    }

    #[test]
    fn one_over_max_flagged() {
        let v = audit(&[("a", 101)], 100);
        if let RuntimeVerdict::Ok { offenders, .. } = v {
            assert_eq!(offenders, vec!["a".to_string()]);
        }
    }

    #[test]
    fn median_correct() {
        let v = audit(&[("a", 5), ("b", 10), ("c", 100)], 1000);
        if let RuntimeVerdict::Ok { median_ms, .. } = v {
            assert_eq!(median_ms, 10);
        }
    }

    #[test]
    fn p99_ge_median() {
        let v = audit(&[("a", 5), ("b", 10), ("c", 100)], 1000);
        if let RuntimeVerdict::Ok {
            median_ms, p99_ms, ..
        } = v
        {
            assert!(p99_ms >= median_ms);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&[("a", 5)], 100);
        let r2 = audit(&[("a", 5)], 100);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted() {
        let v = audit(&[("zeta", 200), ("alpha", 200)], 100);
        if let RuntimeVerdict::Ok { offenders, .. } = v {
            assert_eq!(offenders, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn many_recipes_handled() {
        let recipes: Vec<(&str, u32)> = (0..30).map(|_| ("r", 5)).collect();
        let v = audit(&recipes, 100);
        if let RuntimeVerdict::Ok { offenders, .. } = v {
            assert!(offenders.is_empty());
        }
    }
}
