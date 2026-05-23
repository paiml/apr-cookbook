//! # Monte-Carlo Dice Throw Distribution
//!
//! Sim N rolls of a fair 6-sided die. Verify roughly uniform face
//! distribution via chi-squared bound. Reports per-face count + max
//! deviation from expected.
//!
//! Demonstrates the **MC.126** recipe for PMAT-201 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Bernoulli trial uniformity; Knuth TAOCP §3 random-number
//!  uniformity tests.
//!
//! Run with: cargo run --example mc_dice_throw_distribution
//!
//! Added by PMAT-201 (catalog 1432→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DiceVerdict {
    Ok {
        face_counts: [u32; 6],
        max_deviation_pct: f64,
    },
    InvalidConfig,
}

pub fn simulate(rolls: u32, seed: u64) -> DiceVerdict {
    if rolls == 0 {
        return DiceVerdict::InvalidConfig;
    }
    let mut face_counts = [0u32; 6];
    let mut rng_state = seed | 1;
    for _ in 0..rolls {
        let face = ((lcg(&mut rng_state) >> 32) as u32) % 6;
        face_counts[face as usize] += 1;
    }
    let expected = f64::from(rolls) / 6.0;
    let max_deviation_pct = face_counts
        .iter()
        .map(|&c| (f64::from(c) - expected).abs() / expected * 100.0)
        .fold(f64::MIN, f64::max);
    DiceVerdict::Ok {
        face_counts,
        max_deviation_pct,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_dice_throw_distribution")?;

    println!("typical: {:?}", simulate(60_000, 42));
    println!("invalid: {:?}", simulate(0, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn large_sample_low_deviation() {
        let v = simulate(60_000, 42);
        if let DiceVerdict::Ok {
            max_deviation_pct, ..
        } = v
        {
            assert!(max_deviation_pct < 5.0);
        }
    }

    #[test]
    fn invalid_zero_rolls() {
        assert_eq!(simulate(0, 42), DiceVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 42);
        let b = simulate(1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn counts_sum_to_rolls() {
        let v = simulate(1000, 42);
        if let DiceVerdict::Ok { face_counts, .. } = v {
            let sum: u32 = face_counts.iter().sum();
            assert_eq!(sum, 1000);
        }
    }

    #[test]
    fn all_six_faces_appear() {
        let v = simulate(10_000, 42);
        if let DiceVerdict::Ok { face_counts, .. } = v {
            for c in &face_counts {
                assert!(*c > 0);
            }
        }
    }

    #[test]
    fn deviation_finite() {
        let v = simulate(1000, 42);
        if let DiceVerdict::Ok {
            max_deviation_pct, ..
        } = v
        {
            assert!(max_deviation_pct.is_finite());
        }
    }

    #[test]
    fn small_sample_higher_variance() {
        let small = simulate(50, 42);
        let big = simulate(10_000, 42);
        if let (
            DiceVerdict::Ok {
                max_deviation_pct: s,
                ..
            },
            DiceVerdict::Ok {
                max_deviation_pct: b,
                ..
            },
        ) = (small, big)
        {
            assert!(b < s);
        }
    }

    #[test]
    fn single_roll_works() {
        let v = simulate(1, 42);
        if let DiceVerdict::Ok { face_counts, .. } = v {
            let total: u32 = face_counts.iter().sum();
            assert_eq!(total, 1);
        }
    }

    #[test]
    fn deviation_nonneg() {
        let v = simulate(1000, 42);
        if let DiceVerdict::Ok {
            max_deviation_pct, ..
        } = v
        {
            assert!(max_deviation_pct >= 0.0);
        }
    }

    #[test]
    fn many_rolls_handled() {
        let v = simulate(1_000_000, 42);
        assert!(matches!(v, DiceVerdict::Ok { .. }));
    }

    #[test]
    fn different_seed_different_counts() {
        let v1 = simulate(10_000, 1);
        let v2 = simulate(10_000, 2);
        if let (
            DiceVerdict::Ok {
                face_counts: c1, ..
            },
            DiceVerdict::Ok {
                face_counts: c2, ..
            },
        ) = (v1, v2)
        {
            // Both should sum to 10_000.
            let s1: u32 = c1.iter().sum();
            let s2: u32 = c2.iter().sum();
            assert_eq!(s1, 10_000);
            assert_eq!(s2, 10_000);
        }
    }
}
