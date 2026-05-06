//! # Advanced Speculative Decoding Window Sizer
//!
//! Speculative decoding runs a small "draft" model to propose K
//! tokens, then a single forward pass through the big "verify" model
//! accepts/rejects them. Optimal K depends on:
//! - draft acceptance rate p
//! - speedup of draft vs verify (S)
//! - target overhead bound
//!
//! Heuristic: K* ≈ (S - 1) / (1 - p), clamped to [1, 16].
//!
//! Demonstrates the **ADV.8** recipe for PMAT-139 (advanced coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Leviathan et al. (2023). Speculative Decoding. arXiv:2211.17192.
//!
//! Run with: cargo run --example adv_speculative_decode_window
//!
//! Added by PMAT-139 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MAX_K: u32 = 16;

#[derive(Debug, PartialEq)]
pub enum WindowVerdict {
    Ok { k: u32, expected_speedup: f64 },
    InvalidAcceptance,
    InvalidSpeedup,
}

pub fn pick(acceptance_rate: f64, draft_speedup: f64) -> WindowVerdict {
    if !acceptance_rate.is_finite() || !(0.0..1.0).contains(&acceptance_rate) {
        return WindowVerdict::InvalidAcceptance;
    }
    if !draft_speedup.is_finite() || draft_speedup <= 1.0 {
        return WindowVerdict::InvalidSpeedup;
    }
    // Speedup = α(K) × c / (c + K), α(K) = (1 - p^(K+1)) / (1 - p).
    // Search K in [1, MAX_K] for the maximum.
    let p = acceptance_rate;
    let c = draft_speedup;
    let mut best_k = 1u32;
    let mut best_speedup = 0.0_f64;
    for k in 1..=MAX_K {
        let alpha = (1.0 - p.powi((k + 1) as i32)) / (1.0 - p);
        let speedup = alpha * c / (c + f64::from(k));
        if speedup > best_speedup {
            best_speedup = speedup;
            best_k = k;
        }
    }
    WindowVerdict::Ok {
        k: best_k,
        expected_speedup: best_speedup,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_speculative_decode_window")?;

    let cases = [(0.7_f64, 5.0_f64), (0.9, 10.0), (0.5, 3.0), (0.95, 20.0)];
    for (p, s) in cases {
        println!("p={p} S={s} → {:?}", pick(p, s));
    }
    println!("invalid p=1.0: {:?}", pick(1.0, 5.0));
    println!("invalid S=0.5: {:?}", pick(0.7, 0.5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_inputs_yield_window() {
        let v = pick(0.7, 5.0);
        if let WindowVerdict::Ok {
            k,
            expected_speedup,
        } = v
        {
            // For p=0.7, c=5, scanning K in [1, MAX_K] picks K=3, speedup ≈ 1.58.
            assert_eq!(k, 3);
            assert!(expected_speedup > 1.0);
        }
    }

    #[test]
    fn high_acceptance_increases_k() {
        let v_low = pick(0.5, 5.0);
        let v_high = pick(0.9, 5.0);
        if let (WindowVerdict::Ok { k: k_low, .. }, WindowVerdict::Ok { k: k_high, .. }) =
            (v_low, v_high)
        {
            assert!(k_high > k_low);
        }
    }

    #[test]
    fn k_clamped_to_max() {
        // Very high p + very high S should still cap at MAX_K.
        let v = pick(0.99, 1000.0);
        if let WindowVerdict::Ok { k, .. } = v {
            assert_eq!(k, MAX_K);
        }
    }

    #[test]
    fn k_clamped_to_min_one() {
        // Very low S close to 1 → would round to 0; clamp to 1.
        let v = pick(0.5, 1.01);
        if let WindowVerdict::Ok { k, .. } = v {
            assert!(k >= 1);
        }
    }

    #[test]
    fn acceptance_at_one_invalid() {
        assert_eq!(pick(1.0, 5.0), WindowVerdict::InvalidAcceptance);
    }

    #[test]
    fn negative_acceptance_invalid() {
        assert_eq!(pick(-0.1, 5.0), WindowVerdict::InvalidAcceptance);
    }

    #[test]
    fn speedup_at_one_invalid() {
        // Draft must be strictly faster than verify.
        assert_eq!(pick(0.7, 1.0), WindowVerdict::InvalidSpeedup);
    }

    #[test]
    fn speedup_below_one_invalid() {
        assert_eq!(pick(0.7, 0.5), WindowVerdict::InvalidSpeedup);
    }

    #[test]
    fn nan_inputs_invalid() {
        assert_eq!(pick(f64::NAN, 5.0), WindowVerdict::InvalidAcceptance);
        assert_eq!(pick(0.7, f64::NAN), WindowVerdict::InvalidSpeedup);
    }

    #[test]
    fn expected_speedup_at_least_one() {
        if let WindowVerdict::Ok {
            expected_speedup, ..
        } = pick(0.7, 5.0)
        {
            assert!(expected_speedup >= 1.0);
        }
    }
}
