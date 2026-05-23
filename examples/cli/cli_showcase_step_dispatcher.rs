//! # apr showcase — Step Dispatcher
//!
//! `apr showcase --step <NAME>` runs a single step of the demo pipeline
//! instead of the full sequence (download → quantize → benchmark →
//! compare → report). This recipe documents the step ordering and the
//! dependency graph: each step has prerequisites that must run first or
//! dispatch must refuse to start.
//!
//! Demonstrates the **SHOWCASE.4** recipe for PMAT-096 (apr showcase coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHOWCASE-002 + DAG dispatch convention
//!
//! Run with: cargo run --example cli_showcase_step_dispatcher
//!
//! Added by PMAT-096 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Download,
    Quantize,
    Benchmark,
    Compare,
    Report,
}

impl Step {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "download" => Some(Step::Download),
            "quantize" => Some(Step::Quantize),
            "benchmark" => Some(Step::Benchmark),
            "compare" => Some(Step::Compare),
            "report" => Some(Step::Report),
            _ => None,
        }
    }

    /// Steps that must complete before this step can run.
    pub fn prerequisites(self) -> Vec<Step> {
        match self {
            Step::Download => vec![],
            Step::Quantize => vec![Step::Download],
            Step::Benchmark => vec![Step::Download, Step::Quantize],
            Step::Compare => vec![Step::Download, Step::Benchmark],
            Step::Report => vec![Step::Benchmark, Step::Compare],
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok,
    UnknownStep(String),
    MissingPrerequisites(Vec<Step>),
}

pub fn dispatch_step(name: &str, completed: &[Step]) -> DispatchVerdict {
    let Some(step) = Step::from_str_strict(name) else {
        return DispatchVerdict::UnknownStep(name.into());
    };
    let missing: Vec<Step> = step
        .prerequisites()
        .into_iter()
        .filter(|p| !completed.contains(p))
        .collect();
    if missing.is_empty() {
        DispatchVerdict::Ok
    } else {
        DispatchVerdict::MissingPrerequisites(missing)
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_showcase_step_dispatcher")?;

    // Fresh state: only download can run.
    println!("fresh state:");
    for s in [
        "download",
        "quantize",
        "benchmark",
        "compare",
        "report",
        "what",
    ] {
        println!("  --step {s:>10}  →  {:?}", dispatch_step(s, &[]));
    }

    // After download + quantize: benchmark unblocked.
    let done = [Step::Download, Step::Quantize];
    println!("\nafter download+quantize:");
    for s in ["benchmark", "compare", "report"] {
        println!("  --step {s:>10}  →  {:?}", dispatch_step(s, &done));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn download_has_no_prerequisites() {
        assert!(Step::Download.prerequisites().is_empty());
        assert_eq!(dispatch_step("download", &[]), DispatchVerdict::Ok);
    }

    #[test]
    fn quantize_requires_download() {
        match dispatch_step("quantize", &[]) {
            DispatchVerdict::MissingPrerequisites(missing) => {
                assert_eq!(missing, vec![Step::Download]);
            }
            v => panic!("expected MissingPrerequisites, got {v:?}"),
        }
    }

    #[test]
    fn benchmark_requires_download_and_quantize() {
        let verdict = dispatch_step("benchmark", &[Step::Download]);
        if let DispatchVerdict::MissingPrerequisites(missing) = verdict {
            assert_eq!(missing, vec![Step::Quantize]);
        } else {
            panic!("expected MissingPrerequisites");
        }
    }

    #[test]
    fn report_dispatches_after_benchmark_and_compare() {
        // Report needs both benchmark + compare; download/quantize done implicitly.
        assert_eq!(
            dispatch_step("report", &[Step::Benchmark, Step::Compare]),
            DispatchVerdict::Ok
        );
    }

    #[test]
    fn unknown_step_returns_unknown() {
        let v = dispatch_step("benchmrk", &[]);
        assert!(matches!(v, DispatchVerdict::UnknownStep(s) if s == "benchmrk"));
    }

    #[test]
    fn full_state_unblocks_everything() {
        let all = [
            Step::Download,
            Step::Quantize,
            Step::Benchmark,
            Step::Compare,
            Step::Report,
        ];
        for s in ["download", "quantize", "benchmark", "compare", "report"] {
            assert_eq!(dispatch_step(s, &all), DispatchVerdict::Ok);
        }
    }
}
