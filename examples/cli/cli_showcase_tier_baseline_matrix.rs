//! # apr showcase — Tier × Baseline Matrix
//!
//! `apr showcase --tier <T> --baseline <B>` runs the Qwen2.5-Coder demo
//! comparing one of four model tiers (tiny/small/medium/large) against a
//! comma-separated list of baseline runtimes (llama-cpp,ollama). This
//! recipe builds the (tier, baseline) cross-product and asserts the
//! invocation envelope so a CI pipeline can preview which (tier, baseline)
//! pairs would actually be exercised.
//!
//! Demonstrates the **SHOWCASE.3** recipe for PMAT-096 (apr showcase coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHOWCASE-001 + Qwen2.5 model card
//!
//! Run with: cargo run --example cli_showcase_tier_baseline_matrix
//!
//! Added by PMAT-096 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Tiny,
    Small,
    Medium,
    Large,
}

impl Tier {
    pub fn billion_params(self) -> f64 {
        match self {
            Tier::Tiny => 0.5,
            Tier::Small => 1.5,
            Tier::Medium => 7.0,
            Tier::Large => 32.0,
        }
    }

    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "tiny" => Some(Tier::Tiny),
            "small" => Some(Tier::Small),
            "medium" => Some(Tier::Medium),
            "large" => Some(Tier::Large),
            _ => None,
        }
    }
}

const KNOWN_BASELINES: &[&str] = &["llama-cpp", "ollama", "vllm", "tgi"];

#[derive(Debug, PartialEq, Eq)]
pub struct InvocationPlan {
    pub tier: Tier,
    pub baselines_kept: Vec<&'static str>,
    pub baselines_unknown: Vec<String>,
}

pub fn build_plan(tier: Tier, baseline_csv: &str) -> InvocationPlan {
    let mut kept = Vec::new();
    let mut unknown = Vec::new();
    for raw in baseline_csv
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if let Some(canonical) = KNOWN_BASELINES.iter().find(|b| **b == raw) {
            kept.push(*canonical);
        } else {
            unknown.push(raw.to_string());
        }
    }
    InvocationPlan {
        tier,
        baselines_kept: kept,
        baselines_unknown: unknown,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_showcase_tier_baseline_matrix")?;

    for tier_str in ["tiny", "small", "medium", "large", "xl"] {
        let tier = Tier::from_str_strict(tier_str);
        println!("--tier {tier_str:>6}  parses to  {tier:?}");
    }

    for baseline_csv in ["llama-cpp,ollama", "llama-cpp", "vllm,banana,tgi", ""] {
        let plan = build_plan(Tier::Small, baseline_csv);
        println!(
            "--baseline {baseline_csv:>22}  →  kept={:?}  unknown={:?}",
            plan.baselines_kept, plan.baselines_unknown
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn tier_param_count_grows_monotonically() {
        for w in [Tier::Tiny, Tier::Small, Tier::Medium, Tier::Large].windows(2) {
            assert!(w[1].billion_params() > w[0].billion_params());
        }
    }

    #[test]
    fn unknown_tier_returns_none() {
        assert!(Tier::from_str_strict("frontier").is_none());
        assert!(Tier::from_str_strict("").is_none());
        // Default in clap is "small" — caller, not parser, decides default.
    }

    #[test]
    fn known_baselines_kept() {
        let p = build_plan(Tier::Small, "llama-cpp,ollama");
        assert_eq!(p.baselines_kept, vec!["llama-cpp", "ollama"]);
        assert!(p.baselines_unknown.is_empty());
    }

    #[test]
    fn unknown_baseline_separated_from_known() {
        // Critical: typos like "ollam" must surface as warnings, not silently skip.
        let p = build_plan(Tier::Small, "ollam,vllm");
        assert_eq!(p.baselines_kept, vec!["vllm"]);
        assert_eq!(p.baselines_unknown, vec!["ollam".to_string()]);
    }

    #[test]
    fn whitespace_trimmed_in_csv() {
        let p = build_plan(Tier::Small, "llama-cpp,    ollama   ");
        assert_eq!(p.baselines_kept.len(), 2);
    }

    #[test]
    fn empty_csv_yields_empty_plan() {
        let p = build_plan(Tier::Small, "");
        assert!(p.baselines_kept.is_empty());
        assert!(p.baselines_unknown.is_empty());
    }
}
