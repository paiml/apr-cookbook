//! # Monte-Carlo Galton Board (Bean Machine)
//!
//! Sim balls falling through a Galton board with N rows of pegs;
//! at each peg, ball goes left or right with probability 0.5.
//! Returns final-bin distribution and mean bin index.
//!
//! Demonstrates the **MC.166** recipe for PMAT-214 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Galton, "Natural Inheritance" (1889); de Moivre-Laplace
//!  binomial-to-normal limit theorem.
//!
//! Run with: cargo run --example mc_galton_board_distribution
//!
//! Added by PMAT-214 (catalog 1549→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GaltonVerdict {
    Ok {
        bin_counts: Vec<u32>,
        mean_bin_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(rows: u32, balls: u32, seed: u64) -> GaltonVerdict {
    if rows < 2 || balls < 100 {
        return GaltonVerdict::InvalidConfig;
    }
    let n_bins = (rows + 1) as usize;
    let mut bins: Vec<u32> = vec![0; n_bins];
    let mut state = seed | 1;
    for _ in 0..balls {
        let mut bin = 0u32;
        for _ in 0..rows {
            if (lcg(&mut state) >> 32) & 1 == 1 {
                bin += 1;
            }
        }
        bins[bin as usize] += 1;
    }
    let total_bins: u64 = bins
        .iter()
        .enumerate()
        .map(|(i, c)| (i as u64) * (*c as u64))
        .sum();
    let mean = (total_bins as f64 / balls as f64) * 100.0;
    GaltonVerdict::Ok {
        bin_counts: bins,
        mean_bin_x100: mean as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_galton_board_distribution")?;

    println!("10 rows: {:?}", simulate(10, 10_000, 42));
    println!("invalid: {:?}", simulate(1, 100, 42));
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
    fn invalid_too_few_rows() {
        assert_eq!(simulate(1, 100, 42), GaltonVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_balls() {
        assert_eq!(simulate(5, 50, 42), GaltonVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(10, 1000, 42);
        let b = simulate(10, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn bin_count_equals_rows_plus_one() {
        let v = simulate(10, 1000, 42);
        if let GaltonVerdict::Ok { bin_counts, .. } = v {
            assert_eq!(bin_counts.len(), 11);
        }
    }

    #[test]
    fn total_balls_equals_input() {
        let v = simulate(10, 1000, 42);
        if let GaltonVerdict::Ok { bin_counts, .. } = v {
            let sum: u32 = bin_counts.iter().sum();
            assert_eq!(sum, 1000);
        }
    }

    #[test]
    fn mean_near_n_over_2() {
        // E[bin] = n/2 for fair coin → 5.0 for n=10 → 500 (×100).
        let v = simulate(10, 50_000, 42);
        if let GaltonVerdict::Ok { mean_bin_x100, .. } = v {
            assert!((480..=520).contains(&mean_bin_x100));
        }
    }

    #[test]
    fn distribution_peaked_at_center() {
        let v = simulate(20, 50_000, 42);
        if let GaltonVerdict::Ok { bin_counts, .. } = v {
            let n = bin_counts.len();
            let center = bin_counts[n / 2];
            assert!(center > bin_counts[0]);
            assert!(center > bin_counts[n - 1]);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(2, 100, 42);
        assert!(matches!(v, GaltonVerdict::Ok { .. }));
    }

    #[test]
    fn many_balls_handled() {
        let v = simulate(10, 100_000, 42);
        assert!(matches!(v, GaltonVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(10, 500, 42);
        let b = simulate(10, 500, 999);
        assert!(a != b);
    }

    #[test]
    fn extreme_bins_get_few_balls() {
        // Edge bins (0 and N) require all-left or all-right → very rare.
        let v = simulate(20, 10_000, 42);
        if let GaltonVerdict::Ok { bin_counts, .. } = v {
            assert!(bin_counts[0] < 10);
            assert!(bin_counts[bin_counts.len() - 1] < 10);
        }
    }
}
