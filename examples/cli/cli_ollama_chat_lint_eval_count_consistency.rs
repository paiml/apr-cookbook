//! # apr ollama-chat-lint — `eval_count` Consistency Check
//!
//! Ollama's /api/chat response includes `prompt_eval_count` (input
//! tokens) + `eval_count` (generated tokens) + `eval_duration` (ns).
//! This recipe verifies the consistency: per-token latency =
//! eval_duration / eval_count, must be > 0 and < 10 seconds (any model
//! slower than 0.1 tok/s is broken). Plus, eval_count <= max_tokens
//! requested in the original `options.num_predict`.
//!
//! Demonstrates the **OLLAMA-CHAT.4** recipe for PMAT-108 (apr ollama-chat-lint coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CRUX-C-04 + Ollama API docs (eval_count semantics)
//!
//! Run with: cargo run --example cli_ollama_chat_lint_eval_count_consistency
//!
//! Added by PMAT-108 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ConsistencyVerdict {
    Ok { tokens_per_second: f64 },
    EvalCountZero,
    EvalDurationZero,
    UnreasonablySlow { tps: f64 },
    ExceedsMaxTokens { observed: u64, max: u64 },
}

const MAX_PER_TOKEN_NS: u64 = 10_000_000_000; // 10 seconds per token
const MIN_TPS: f64 = 0.1;

pub fn check_consistency(
    eval_count: u64,
    eval_duration_ns: u64,
    max_tokens_requested: Option<u64>,
) -> ConsistencyVerdict {
    if eval_count == 0 {
        return ConsistencyVerdict::EvalCountZero;
    }
    if eval_duration_ns == 0 {
        return ConsistencyVerdict::EvalDurationZero;
    }
    if let Some(max) = max_tokens_requested {
        if eval_count > max {
            return ConsistencyVerdict::ExceedsMaxTokens {
                observed: eval_count,
                max,
            };
        }
    }
    let per_token_ns = eval_duration_ns / eval_count;
    if per_token_ns > MAX_PER_TOKEN_NS {
        let tps = 1e9 / per_token_ns as f64;
        return ConsistencyVerdict::UnreasonablySlow { tps };
    }
    let tps = 1e9 / per_token_ns as f64;
    if tps < MIN_TPS {
        return ConsistencyVerdict::UnreasonablySlow { tps };
    }
    ConsistencyVerdict::Ok {
        tokens_per_second: tps,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_ollama_chat_lint_eval_count_consistency")?;

    let cases = [
        ("happy 100 tok / 5s", 100u64, 5_000_000_000u64, Some(512u64)),
        ("zero eval_count", 0, 1_000_000_000, Some(512)),
        ("zero eval_duration", 100, 0, Some(512)),
        ("over max", 1000, 10_000_000_000, Some(512)),
        ("0.05 tps slow", 1, 20_000_000_000, Some(512)),
    ];
    for (label, ec, ed, max) in cases {
        println!("{label:>22}  →  {:?}", check_consistency(ec, ed, max));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consistency_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_response_passes() {
        // 100 tokens in 5 seconds = 20 tps.
        let v = check_consistency(100, 5_000_000_000, Some(512));
        if let ConsistencyVerdict::Ok { tokens_per_second } = v {
            assert!((tokens_per_second - 20.0).abs() < 1.0);
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn zero_eval_count_rejected() {
        let v = check_consistency(0, 1_000_000_000, None);
        assert_eq!(v, ConsistencyVerdict::EvalCountZero);
    }

    #[test]
    fn zero_eval_duration_rejected() {
        let v = check_consistency(100, 0, None);
        assert_eq!(v, ConsistencyVerdict::EvalDurationZero);
    }

    #[test]
    fn unreasonably_slow_flagged() {
        // 1 token in 20 seconds = 0.05 tps < 0.1 floor.
        let v = check_consistency(1, 20_000_000_000, None);
        assert!(matches!(v, ConsistencyVerdict::UnreasonablySlow { .. }));
    }

    #[test]
    fn exceeds_max_tokens_flagged() {
        let v = check_consistency(1000, 1_000_000_000, Some(500));
        assert_eq!(
            v,
            ConsistencyVerdict::ExceedsMaxTokens {
                observed: 1000,
                max: 500,
            }
        );
    }

    #[test]
    fn no_max_skips_overage_check() {
        // When max_tokens is None, we don't check overage.
        let v = check_consistency(1000, 1_000_000_000, None);
        assert!(matches!(v, ConsistencyVerdict::Ok { .. }));
    }

    #[test]
    fn boundary_at_eval_count_equals_max_passes() {
        let v = check_consistency(500, 1_000_000_000, Some(500));
        assert!(matches!(v, ConsistencyVerdict::Ok { .. }));
    }
}
