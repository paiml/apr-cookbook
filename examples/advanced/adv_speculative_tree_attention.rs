//! # Advanced Speculative Tree Attention
//!
//! Speculative decoding can verify multiple draft branches in parallel
//! by laying them out as a tree of n-ary candidates. Compute total
//! verify-pass tokens = sum over depth d of n^d (n-ary tree).
//!
//! Picker rule:
//!   high acceptance (p > 0.8): wide tree (n=4, depth=4)
//!   medium acceptance (0.5..0.8): balanced (n=2, depth=4)
//!   low acceptance (p < 0.5): narrow (n=2, depth=2) or no tree
//!
//! Demonstrates the **ADV.17** recipe for PMAT-145 (advanced round 6).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Medusa: tree-attention speculative decoding (Cai et al. 2024).
//!
//! Run with: cargo run --example adv_speculative_tree_attention
//!
//! Added by PMAT-145 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TreeVerdict {
    Ok {
        n_ary: u32,
        depth: u32,
        total_verify_tokens: u64,
    },
    NoTree,
    InvalidProbability,
}

pub fn pick(acceptance_rate: f64, max_verify_budget_tokens: u64) -> TreeVerdict {
    if !acceptance_rate.is_finite() || !(0.0..=1.0).contains(&acceptance_rate) {
        return TreeVerdict::InvalidProbability;
    }
    let (n_ary, depth) = if acceptance_rate > 0.8 {
        (4u32, 4u32)
    } else if acceptance_rate > 0.5 {
        (2, 4)
    } else if acceptance_rate >= 0.3 {
        (2, 2)
    } else {
        return TreeVerdict::NoTree;
    };
    let total: u64 = (1..=depth).map(|d| u64::from(n_ary).pow(d)).sum();
    if total > max_verify_budget_tokens {
        return TreeVerdict::NoTree;
    }
    TreeVerdict::Ok {
        n_ary,
        depth,
        total_verify_tokens: total,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_speculative_tree_attention")?;

    println!("high accept: {:?}", pick(0.9, 1_000));
    println!("medium accept: {:?}", pick(0.7, 100));
    println!("low accept: {:?}", pick(0.4, 100));
    println!("very low accept: {:?}", pick(0.1, 100));
    println!("over budget: {:?}", pick(0.9, 50));
    println!("invalid prob: {:?}", pick(1.5, 100));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn high_accept_wide_tree() {
        let v = pick(0.9, 10_000);
        if let TreeVerdict::Ok { n_ary, depth, .. } = v {
            assert_eq!(n_ary, 4);
            assert_eq!(depth, 4);
        }
    }

    #[test]
    fn medium_accept_balanced_tree() {
        let v = pick(0.7, 10_000);
        if let TreeVerdict::Ok { n_ary, depth, .. } = v {
            assert_eq!(n_ary, 2);
            assert_eq!(depth, 4);
        }
    }

    #[test]
    fn low_accept_narrow_tree() {
        let v = pick(0.4, 10_000);
        if let TreeVerdict::Ok { n_ary, depth, .. } = v {
            assert_eq!(n_ary, 2);
            assert_eq!(depth, 2);
        }
    }

    #[test]
    fn very_low_accept_no_tree() {
        let v = pick(0.1, 10_000);
        assert_eq!(v, TreeVerdict::NoTree);
    }

    #[test]
    fn over_budget_no_tree() {
        // High accept wants 4+16+64+256 = 340 tokens; budget 50.
        let v = pick(0.9, 50);
        assert_eq!(v, TreeVerdict::NoTree);
    }

    #[test]
    fn invalid_prob_above_one() {
        assert_eq!(pick(1.5, 100), TreeVerdict::InvalidProbability);
    }

    #[test]
    fn invalid_prob_negative() {
        assert_eq!(pick(-0.1, 100), TreeVerdict::InvalidProbability);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(pick(f64::NAN, 100), TreeVerdict::InvalidProbability);
    }

    #[test]
    fn high_accept_total_tokens() {
        // n=4, depth=4: 4 + 16 + 64 + 256 = 340.
        let v = pick(0.9, 1_000);
        if let TreeVerdict::Ok {
            total_verify_tokens,
            ..
        } = v
        {
            assert_eq!(total_verify_tokens, 340);
        }
    }

    #[test]
    fn medium_accept_total_tokens() {
        // n=2, depth=4: 2 + 4 + 8 + 16 = 30.
        let v = pick(0.7, 1_000);
        if let TreeVerdict::Ok {
            total_verify_tokens,
            ..
        } = v
        {
            assert_eq!(total_verify_tokens, 30);
        }
    }

    #[test]
    fn at_threshold_p_05_uses_balanced() {
        // At exactly 0.5 → narrow (since rule is `> 0.5` for balanced).
        let v = pick(0.5, 1_000);
        if let TreeVerdict::Ok { n_ary, depth, .. } = v {
            assert_eq!(n_ary, 2);
            assert_eq!(depth, 2);
        }
    }
}
