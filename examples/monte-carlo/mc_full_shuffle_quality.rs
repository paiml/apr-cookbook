//! # Monte-Carlo Full Shuffle Quality (Spotify-Style Test)
//!
//! Test pseudo-random Fisher-Yates shuffle uniformity by repeating
//! many shuffles and counting how often each item lands at each
//! position. Returns max deviation from uniform expected count.
//!
//! Demonstrates the **MC.162** recipe for PMAT-212 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Knuth, TAOCP §3.4.2 Algorithm P (Fisher-Yates); Spotify
//!  shuffle redesign post (2014) on perceived randomness.
//!
//! Run with: cargo run --example mc_full_shuffle_quality
//!
//! Added by PMAT-212 (catalog 1531→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ShuffleVerdict {
    Ok {
        max_deviation_pct_x10: u32,
        n_items: u32,
    },
    InvalidConfig,
}

pub fn simulate(n_items: u32, trials: u32, seed: u64) -> ShuffleVerdict {
    if n_items < 3 || trials < 1000 {
        return ShuffleVerdict::InvalidConfig;
    }
    let n = n_items as usize;
    let mut state = seed | 1;
    let mut counts = vec![vec![0u32; n]; n];
    for _ in 0..trials {
        let mut perm: Vec<u32> = (0..n_items).collect();
        for i in (1..perm.len()).rev() {
            let j = (lcg(&mut state) as usize) % (i + 1);
            perm.swap(i, j);
        }
        for (pos, item) in perm.iter().enumerate() {
            counts[*item as usize][pos] += 1;
        }
    }
    let expected = trials as f64 / n_items as f64;
    let mut max_dev_pct = 0.0f64;
    for row in &counts {
        for c in row {
            let dev = (((*c as f64) - expected) / expected * 100.0).abs();
            if dev > max_dev_pct {
                max_dev_pct = dev;
            }
        }
    }
    ShuffleVerdict::Ok {
        max_deviation_pct_x10: (max_dev_pct * 10.0) as u32,
        n_items,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_full_shuffle_quality")?;

    println!("n=10, T=10k: {:?}", simulate(10, 10_000, 42));
    println!("invalid: {:?}", simulate(2, 1000, 42));
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
    fn invalid_too_few_items() {
        assert_eq!(simulate(2, 1000, 42), ShuffleVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(5, 100, 42), ShuffleVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(5, 1000, 42);
        let b = simulate(5, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn high_trials_low_deviation() {
        // Large T → deviation should be small.
        let v = simulate(10, 100_000, 42);
        if let ShuffleVerdict::Ok {
            max_deviation_pct_x10,
            ..
        } = v
        {
            // With 100k trials of n=10, each cell expects 10000; deviation < 5%.
            assert!(max_deviation_pct_x10 < 50);
        }
    }

    #[test]
    fn n_items_returned() {
        let v = simulate(7, 1000, 42);
        if let ShuffleVerdict::Ok { n_items, .. } = v {
            assert_eq!(n_items, 7);
        }
    }

    #[test]
    fn deviation_finite() {
        let v = simulate(5, 1000, 42);
        if let ShuffleVerdict::Ok {
            max_deviation_pct_x10,
            ..
        } = v
        {
            assert!(max_deviation_pct_x10 < u32::MAX);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(3, 1000, 42);
        assert!(matches!(v, ShuffleVerdict::Ok { .. }));
    }

    #[test]
    fn many_items_handled() {
        let v = simulate(50, 5000, 42);
        assert!(matches!(v, ShuffleVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(5, 1000, 42);
        let b = simulate(5, 1000, 999);
        assert!(a != b);
    }

    #[test]
    fn smaller_n_smaller_deviation() {
        // Smaller n + same trials → more samples per cell → tighter.
        let small_n = simulate(5, 10_000, 42);
        let large_n = simulate(50, 10_000, 42);
        if let (
            ShuffleVerdict::Ok {
                max_deviation_pct_x10: s,
                ..
            },
            ShuffleVerdict::Ok {
                max_deviation_pct_x10: l,
                ..
            },
        ) = (small_n, large_n)
        {
            assert!(s <= l);
        }
    }

    #[test]
    fn very_long_run_handled() {
        let v = simulate(10, 100_000, 42);
        assert!(matches!(v, ShuffleVerdict::Ok { .. }));
    }
}
