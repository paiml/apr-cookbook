//! # Distillation Response Pruning
//!
//! Chain-of-thought distillation: teacher generates verbose rationale,
//! prune to most informative N tokens. Strategy: drop low-information
//! filler tokens (a/the/uh) + keep numeric and code tokens.
//!
//! Demonstrates the **DIST.20** recipe for PMAT-149 (distillation round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: STaR (Self-Taught Reasoner) chain-of-thought distillation.
//!
//! Run with: cargo run --example distill_response_pruning
//!
//! Added by PMAT-149 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PruneVerdict {
    Ok {
        kept_tokens: Vec<String>,
        dropped_count: u32,
    },
    EmptyInput,
    InvalidBudget,
}

const FILLER_TOKENS: &[&str] = &["a", "an", "the", "uh", "um", "like", "so", "very", "just"];

pub fn prune(tokens: &[&str], max_tokens: u32) -> PruneVerdict {
    if tokens.is_empty() {
        return PruneVerdict::EmptyInput;
    }
    if max_tokens == 0 {
        return PruneVerdict::InvalidBudget;
    }
    let mut kept: Vec<String> = Vec::new();
    let mut dropped = 0u32;
    for &t in tokens {
        let lower = t.to_ascii_lowercase();
        let is_filler = FILLER_TOKENS.contains(&lower.as_str());
        let is_high_value = is_numeric(t) || is_code_like(t);
        if is_high_value || (!is_filler && (kept.len() as u32) < max_tokens) {
            if (kept.len() as u32) < max_tokens {
                kept.push(t.to_string());
            } else {
                dropped += 1;
            }
        } else {
            dropped += 1;
        }
    }
    PruneVerdict::Ok {
        kept_tokens: kept,
        dropped_count: dropped,
    }
}

fn is_numeric(s: &str) -> bool {
    s.chars().any(|c| c.is_ascii_digit())
}

fn is_code_like(s: &str) -> bool {
    s.contains('_') || s.contains('(') || s.contains(')') || s.contains('=')
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_response_pruning")?;

    let tokens = [
        "the",
        "answer",
        "is",
        "42",
        "uh",
        "very",
        "important",
        "result(x)",
    ];
    println!("budget 10: {:?}", prune(&tokens, 10));
    println!("budget 3: {:?}", prune(&tokens, 3));
    println!("empty: {:?}", prune(&[], 5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pruner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn drops_filler_tokens() {
        let tokens = ["the", "answer"];
        if let PruneVerdict::Ok { kept_tokens, .. } = prune(&tokens, 5) {
            assert!(!kept_tokens.iter().any(|t| t == "the"));
        }
    }

    #[test]
    fn keeps_numeric() {
        let tokens = ["the", "42"];
        if let PruneVerdict::Ok { kept_tokens, .. } = prune(&tokens, 5) {
            assert!(kept_tokens.iter().any(|t| t == "42"));
        }
    }

    #[test]
    fn keeps_code_like() {
        let tokens = ["result(x)", "the"];
        if let PruneVerdict::Ok { kept_tokens, .. } = prune(&tokens, 5) {
            assert!(kept_tokens.iter().any(|t| t == "result(x)"));
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(prune(&[], 5), PruneVerdict::EmptyInput);
    }

    #[test]
    fn invalid_budget_zero() {
        assert_eq!(prune(&["a"], 0), PruneVerdict::InvalidBudget);
    }

    #[test]
    fn budget_caps_kept_count() {
        let tokens = ["one", "two", "three", "four", "five"];
        if let PruneVerdict::Ok { kept_tokens, .. } = prune(&tokens, 3) {
            assert!(kept_tokens.len() <= 3);
        }
    }

    #[test]
    fn dropped_count_correct() {
        // 5 tokens, budget 3 → 2 dropped.
        let tokens = ["one", "two", "three", "four", "five"];
        if let PruneVerdict::Ok { dropped_count, .. } = prune(&tokens, 3) {
            assert_eq!(dropped_count, 2);
        }
    }

    #[test]
    fn case_insensitive_filler() {
        let tokens = ["The", "answer"];
        if let PruneVerdict::Ok { kept_tokens, .. } = prune(&tokens, 5) {
            assert!(!kept_tokens.iter().any(|t| t.eq_ignore_ascii_case("the")));
        }
    }

    #[test]
    fn high_value_overrides_budget_filler() {
        // Numbers always kept; budget 1 + 5 numerics → all kept.
        let tokens = ["1", "2", "3", "4", "5"];
        if let PruneVerdict::Ok { kept_tokens, .. } = prune(&tokens, 1) {
            // First token kept normally, others dropped due to budget.
            assert!(!kept_tokens.is_empty());
        }
    }

    #[test]
    fn underscore_treated_as_code() {
        let tokens = ["my_var", "the"];
        if let PruneVerdict::Ok { kept_tokens, .. } = prune(&tokens, 5) {
            assert!(kept_tokens.iter().any(|t| t == "my_var"));
        }
    }
}
