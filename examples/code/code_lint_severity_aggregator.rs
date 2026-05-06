//! # Code Lint Severity Aggregator
//!
//! Linters emit findings with severity levels (Error / Warning / Info /
//! Hint). Aggregation: any Error → fail; any Warning → warn (CI may
//! still pass); only Info/Hint → green. This recipe builds the
//! aggregator + per-file count tabulator.
//!
//! Demonstrates the **CODE.10** recipe for PMAT-125 (code coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ESLint severity levels; clippy lint groups.
//!
//! Run with: cargo run --example code_lint_severity_aggregator
//!
//! Added by PMAT-125 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Hint,
    Info,
    Warning,
    Error,
}

#[derive(Debug, PartialEq)]
pub enum AggregateVerdict {
    Pass,
    WarnOnly { warning_count: usize },
    Fail { error_count: usize },
    EmptyReport,
}

pub fn aggregate<'a, I>(findings: I) -> AggregateVerdict
where
    I: IntoIterator<Item = (&'a str, Severity)>,
{
    let mut errors = 0usize;
    let mut warnings = 0usize;
    let mut total = 0usize;
    for (_file, sev) in findings {
        total += 1;
        match sev {
            Severity::Error => errors += 1,
            Severity::Warning => warnings += 1,
            _ => {}
        }
    }
    if total == 0 {
        return AggregateVerdict::EmptyReport;
    }
    if errors > 0 {
        return AggregateVerdict::Fail {
            error_count: errors,
        };
    }
    if warnings > 0 {
        return AggregateVerdict::WarnOnly {
            warning_count: warnings,
        };
    }
    AggregateVerdict::Pass
}

#[derive(Debug, Default)]
pub struct PerFileTotals {
    pub errors: u32,
    pub warnings: u32,
    pub infos: u32,
    pub hints: u32,
}

pub fn per_file<'a, I>(findings: I) -> BTreeMap<String, PerFileTotals>
where
    I: IntoIterator<Item = (&'a str, Severity)>,
{
    let mut map: BTreeMap<String, PerFileTotals> = BTreeMap::new();
    for (file, sev) in findings {
        let entry = map.entry(file.to_string()).or_default();
        match sev {
            Severity::Error => entry.errors += 1,
            Severity::Warning => entry.warnings += 1,
            Severity::Info => entry.infos += 1,
            Severity::Hint => entry.hints += 1,
        }
    }
    map
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("code_lint_severity_aggregator")?;

    let findings = [
        ("src/lib.rs", Severity::Warning),
        ("src/lib.rs", Severity::Hint),
        ("src/main.rs", Severity::Error),
        ("src/main.rs", Severity::Warning),
        ("src/util.rs", Severity::Info),
    ];
    println!("aggregate: {:?}", aggregate(findings.iter().copied()));
    let totals = per_file(findings.iter().copied());
    for (f, t) in &totals {
        println!(
            "{f}: E={} W={} I={} H={}",
            t.errors, t.warnings, t.infos, t.hints
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_report_yields_empty() {
        let v = aggregate(std::iter::empty::<(&str, Severity)>());
        assert_eq!(v, AggregateVerdict::EmptyReport);
    }

    #[test]
    fn only_hints_passes() {
        let v = aggregate([("a", Severity::Hint), ("b", Severity::Hint)]);
        assert_eq!(v, AggregateVerdict::Pass);
    }

    #[test]
    fn only_info_passes() {
        let v = aggregate([("a", Severity::Info)]);
        assert_eq!(v, AggregateVerdict::Pass);
    }

    #[test]
    fn warnings_yield_warn_only() {
        let v = aggregate([("a", Severity::Warning), ("b", Severity::Hint)]);
        assert_eq!(v, AggregateVerdict::WarnOnly { warning_count: 1 });
    }

    #[test]
    fn any_error_fails() {
        let v = aggregate([("a", Severity::Warning), ("b", Severity::Error)]);
        assert_eq!(v, AggregateVerdict::Fail { error_count: 1 });
    }

    #[test]
    fn multiple_errors_counted() {
        let v = aggregate([
            ("a", Severity::Error),
            ("b", Severity::Error),
            ("c", Severity::Error),
        ]);
        assert_eq!(v, AggregateVerdict::Fail { error_count: 3 });
    }

    #[test]
    fn per_file_tabulates_separately() {
        let totals = per_file([
            ("a", Severity::Warning),
            ("a", Severity::Error),
            ("b", Severity::Hint),
        ]);
        let a = &totals["a"];
        assert_eq!(a.errors, 1);
        assert_eq!(a.warnings, 1);
        let b = &totals["b"];
        assert_eq!(b.hints, 1);
        assert_eq!(b.errors, 0);
    }

    #[test]
    fn severity_ordering_reflects_priority() {
        // Error > Warning > Info > Hint.
        assert!(Severity::Error > Severity::Warning);
        assert!(Severity::Warning > Severity::Info);
        assert!(Severity::Info > Severity::Hint);
    }

    #[test]
    fn per_file_empty_input_empty_map() {
        let totals = per_file(std::iter::empty::<(&str, Severity)>());
        assert!(totals.is_empty());
    }
}
