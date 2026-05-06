//! # Contracts-Macros Recipe Status Summary
//!
//! Aggregate per-recipe pass/fail status into a portfolio-level
//! summary: total, passing, failing, pass-rate, and worst surface.
//!
//! Demonstrates the **CMM.55** recipe for PMAT-176 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: portfolio-quality dashboards (Grafana SLO panels).
//!
//! Run with: cargo run --example contracts_macros_recipe_status_summary
//!
//! Added by PMAT-176 (catalog 1207→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum SummaryVerdict {
    Ok {
        total: u32,
        passing: u32,
        failing: u32,
        pass_rate: f64,
        worst_surface: Option<String>,
    },
    EmptyInput,
}

pub fn summarize(records: &[(&str, &str, bool)]) -> SummaryVerdict {
    if records.is_empty() {
        return SummaryVerdict::EmptyInput;
    }
    let total = records.len() as u32;
    let passing = records.iter().filter(|(_, _, p)| *p).count() as u32;
    let failing = total - passing;
    let pass_rate = f64::from(passing) / f64::from(total);
    let mut by_surface: BTreeMap<&str, (u32, u32)> = BTreeMap::new();
    for (_, surface, ok) in records {
        let entry = by_surface.entry(*surface).or_insert((0, 0));
        if *ok {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }
    let worst_surface = by_surface
        .iter()
        .max_by_key(|(_, (_, fail))| *fail)
        .filter(|(_, (_, fail))| *fail > 0)
        .map(|(s, _)| (*s).to_string());
    SummaryVerdict::Ok {
        total,
        passing,
        failing,
        pass_rate,
        worst_surface,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_status_summary")?;

    let records = [
        ("recipe_a", "tui", true),
        ("recipe_b", "tui", true),
        ("recipe_c", "monte-carlo", false),
        ("recipe_d", "monte-carlo", false),
        ("recipe_e", "contracts-macros", true),
    ];
    println!("typical: {:?}", summarize(&records));

    let all_pass = [("a", "tui", true)];
    println!("all pass: {:?}", summarize(&all_pass));
    println!("empty: {:?}", summarize(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_breakdown() {
        let records = [("a", "tui", true), ("b", "tui", false), ("c", "mc", true)];
        if let SummaryVerdict::Ok {
            total,
            passing,
            failing,
            ..
        } = summarize(&records)
        {
            assert_eq!(total, 3);
            assert_eq!(passing, 2);
            assert_eq!(failing, 1);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(summarize(&[]), SummaryVerdict::EmptyInput);
    }

    #[test]
    fn pass_rate_correct() {
        let records = [
            ("a", "x", true),
            ("b", "x", true),
            ("c", "x", false),
            ("d", "x", false),
        ];
        if let SummaryVerdict::Ok { pass_rate, .. } = summarize(&records) {
            assert!((pass_rate - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn worst_surface_identified() {
        let records = [
            ("a", "tui", true),
            ("b", "mc", false),
            ("c", "mc", false),
            ("d", "contracts", false),
        ];
        if let SummaryVerdict::Ok { worst_surface, .. } = summarize(&records) {
            assert_eq!(worst_surface, Some("mc".to_string()));
        }
    }

    #[test]
    fn no_failures_no_worst() {
        let records = [("a", "tui", true)];
        if let SummaryVerdict::Ok { worst_surface, .. } = summarize(&records) {
            assert_eq!(worst_surface, None);
        }
    }

    #[test]
    fn all_passing() {
        let records = [("a", "tui", true), ("b", "tui", true)];
        if let SummaryVerdict::Ok {
            passing, failing, ..
        } = summarize(&records)
        {
            assert_eq!(passing, 2);
            assert_eq!(failing, 0);
        }
    }

    #[test]
    fn all_failing() {
        let records = [("a", "tui", false)];
        if let SummaryVerdict::Ok { pass_rate, .. } = summarize(&records) {
            assert!((pass_rate - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn single_record() {
        let records = [("only", "x", true)];
        if let SummaryVerdict::Ok { total, .. } = summarize(&records) {
            assert_eq!(total, 1);
        }
    }

    #[test]
    fn pass_rate_in_unit_range() {
        let records = [("a", "x", true), ("b", "x", false)];
        if let SummaryVerdict::Ok { pass_rate, .. } = summarize(&records) {
            assert!((0.0..=1.0).contains(&pass_rate));
        }
    }

    #[test]
    fn deterministic() {
        let records = [("a", "x", true), ("b", "x", false)];
        let a = summarize(&records);
        let b = summarize(&records);
        assert_eq!(a, b);
    }
}
