//! # Inference Beam Search Width Validator
//!
//! Beam search keeps top-K hypotheses at each step. Constraints:
//! 1 ≤ width ≤ 32 (beyond that, diminishing returns + memory bloat);
//! width=1 = greedy decoding. This recipe builds the validator + memory
//! cost estimator (width × seq_len × hidden_dim × dtype_bytes).
//!
//! Demonstrates the **INF.13** recipe for PMAT-129 (inference coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Sutskever et al. (2014). Sequence to Sequence Learning with Neural Networks.
//!
//! Run with: cargo run --example inference_beam_search_width
//!
//! Added by PMAT-129 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MAX_PRACTICAL_WIDTH: u32 = 32;
const TYPICAL_WIDTH: u32 = 4;

#[derive(Debug, PartialEq)]
pub enum WidthVerdict {
    GreedyDecoding,
    Optimal { width: u32 },
    DiminishingReturns { recommended: u32 },
    InvalidZero,
}

pub fn classify(width: u32) -> WidthVerdict {
    if width == 0 {
        return WidthVerdict::InvalidZero;
    }
    if width == 1 {
        return WidthVerdict::GreedyDecoding;
    }
    if width > MAX_PRACTICAL_WIDTH {
        return WidthVerdict::DiminishingReturns {
            recommended: MAX_PRACTICAL_WIDTH,
        };
    }
    WidthVerdict::Optimal { width }
}

pub fn auto_pick_width(quality_priority: bool, memory_constrained: bool) -> u32 {
    match (quality_priority, memory_constrained) {
        (true, false) => 8,
        (true, true) => TYPICAL_WIDTH,
        (false, false) => TYPICAL_WIDTH,
        (false, true) => 1,
    }
}

pub fn estimated_memory_bytes(
    width: u32,
    seq_len: u32,
    hidden_dim: u32,
    dtype_bytes: u32,
) -> Option<u64> {
    if width == 0 || seq_len == 0 || hidden_dim == 0 || dtype_bytes == 0 {
        return None;
    }
    let mut acc = 1u64;
    for v in [width, seq_len, hidden_dim, dtype_bytes] {
        acc = acc.checked_mul(u64::from(v))?;
    }
    Some(acc)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_beam_search_width")?;

    for w in [0u32, 1, 4, 8, 32, 64] {
        println!("width={w}  →  {:?}", classify(w));
    }
    for (q, m) in [(true, false), (true, true), (false, false), (false, true)] {
        println!("auto({q}, {m}) = {}", auto_pick_width(q, m));
    }
    println!(
        "memory(8, 2048, 4096, 2): {:?}",
        estimated_memory_bytes(8, 2048, 4096, 2)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_width_invalid() {
        assert_eq!(classify(0), WidthVerdict::InvalidZero);
    }

    #[test]
    fn width_one_greedy() {
        assert_eq!(classify(1), WidthVerdict::GreedyDecoding);
    }

    #[test]
    fn typical_width_optimal() {
        assert_eq!(classify(4), WidthVerdict::Optimal { width: 4 });
    }

    #[test]
    fn at_max_practical_optimal() {
        assert_eq!(
            classify(MAX_PRACTICAL_WIDTH),
            WidthVerdict::Optimal {
                width: MAX_PRACTICAL_WIDTH
            }
        );
    }

    #[test]
    fn over_max_diminishing_returns() {
        let v = classify(64);
        assert!(matches!(v, WidthVerdict::DiminishingReturns { .. }));
    }

    #[test]
    fn auto_pick_quality_unconstrained_is_8() {
        assert_eq!(auto_pick_width(true, false), 8);
    }

    #[test]
    fn auto_pick_memory_constrained_low_quality_is_1() {
        // No quality priority + memory pressure → greedy.
        assert_eq!(auto_pick_width(false, true), 1);
    }

    #[test]
    fn memory_estimate_correct() {
        // 8 × 2048 × 4096 × 2 = 134_217_728.
        let bytes = estimated_memory_bytes(8, 2048, 4096, 2).unwrap();
        assert_eq!(bytes, 134_217_728);
    }

    #[test]
    fn memory_zero_dimension_invalid() {
        assert!(estimated_memory_bytes(0, 2048, 4096, 2).is_none());
        assert!(estimated_memory_bytes(8, 0, 4096, 2).is_none());
    }

    #[test]
    fn memory_overflow_returns_none() {
        // u64::MAX × something → overflow.
        assert!(estimated_memory_bytes(u32::MAX, u32::MAX, u32::MAX, u32::MAX).is_none());
    }
}
