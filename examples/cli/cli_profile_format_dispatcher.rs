//! # apr profile — `--format` Output Dispatcher
//!
//! `apr profile <FILE> --format {human,json,flamegraph}` switches output
//! mode. Flamegraph requires `--callgraph` in the same invocation
//! (without callgraph data, the flamegraph would be empty). This recipe
//! builds the dispatcher and asserts the mutual-dependency contract.
//!
//! Demonstrates the **PROFILE.5** recipe for PMAT-102 (apr profile coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PROFILE-002 + Brendan Gregg flamegraph format
//!
//! Run with: cargo run --example cli_profile_format_dispatcher
//!
//! Added by PMAT-102 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileFormat {
    Human,
    Json,
    Flamegraph,
}

impl ProfileFormat {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "human" => Some(ProfileFormat::Human),
            "json" => Some(ProfileFormat::Json),
            "flamegraph" => Some(ProfileFormat::Flamegraph),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok(ProfileFormat),
    UnknownFormat(String),
    FlamegraphNeedsCallgraph,
}

pub fn dispatch(format: &str, callgraph: bool) -> DispatchVerdict {
    let Some(fmt) = ProfileFormat::from_str_strict(format) else {
        return DispatchVerdict::UnknownFormat(format.into());
    };
    if fmt == ProfileFormat::Flamegraph && !callgraph {
        return DispatchVerdict::FlamegraphNeedsCallgraph;
    }
    DispatchVerdict::Ok(fmt)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_profile_format_dispatcher")?;

    let cases = [
        ("human", false),
        ("json", false),
        ("flamegraph", true),
        ("flamegraph", false),
        ("perf-data", false),
    ];

    for (fmt, cg) in cases {
        println!(
            "--format {fmt} --callgraph={cg}  →  {:?}",
            dispatch(fmt, cg)
        );
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
    fn human_default_passes_without_callgraph() {
        assert_eq!(
            dispatch("human", false),
            DispatchVerdict::Ok(ProfileFormat::Human)
        );
    }

    #[test]
    fn json_passes_without_callgraph() {
        assert_eq!(
            dispatch("json", false),
            DispatchVerdict::Ok(ProfileFormat::Json)
        );
    }

    #[test]
    fn flamegraph_with_callgraph_passes() {
        assert_eq!(
            dispatch("flamegraph", true),
            DispatchVerdict::Ok(ProfileFormat::Flamegraph)
        );
    }

    #[test]
    fn flamegraph_without_callgraph_rejected() {
        // Without callgraph data, flamegraph would be empty — surface the conflict.
        assert_eq!(
            dispatch("flamegraph", false),
            DispatchVerdict::FlamegraphNeedsCallgraph
        );
    }

    #[test]
    fn unknown_format_rejected() {
        assert!(matches!(
            dispatch("perf-data", false),
            DispatchVerdict::UnknownFormat(_)
        ));
    }

    #[test]
    fn empty_format_rejected_as_unknown() {
        assert!(matches!(
            dispatch("", false),
            DispatchVerdict::UnknownFormat(_)
        ));
    }
}
