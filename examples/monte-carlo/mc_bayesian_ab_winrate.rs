//! # Monte-Carlo Bayesian A/B Win Rate
//!
//! Compute posterior P(A wins) for two variants given (alpha, beta)
//! parameters of conjugate Beta priors. Use Monte-Carlo sampling
//! from each posterior.
//!
//! Demonstrates the **MC.79** recipe for PMAT-185 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Gelman et al., Bayesian Data Analysis ch.5; David Robinson,
//!  "Empirical Bayes Beta-Binomial" blog series.
//!
//! Run with: cargo run --example mc_bayesian_ab_winrate
//!
//! Added by PMAT-185 (catalog 1288→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BayesianVerdict {
    Ok {
        prob_a_wins: f64,
        mean_a: f64,
        mean_b: f64,
    },
    InvalidConfig,
}

#[allow(clippy::too_many_arguments)]
pub fn simulate(
    samples: u32,
    a_successes: u32,
    a_failures: u32,
    b_successes: u32,
    b_failures: u32,
    seed: u64,
) -> BayesianVerdict {
    if samples == 0 {
        return BayesianVerdict::InvalidConfig;
    }
    let alpha_a = a_successes + 1;
    let beta_a = a_failures + 1;
    let alpha_b = b_successes + 1;
    let beta_b = b_failures + 1;
    let mut a_wins = 0u32;
    let mut sum_a: f64 = 0.0;
    let mut sum_b: f64 = 0.0;
    let mut rng_state = seed | 1;
    for _ in 0..samples {
        let pa = sample_beta(alpha_a, beta_a, &mut rng_state);
        let pb = sample_beta(alpha_b, beta_b, &mut rng_state);
        sum_a += pa;
        sum_b += pb;
        if pa > pb {
            a_wins += 1;
        }
    }
    BayesianVerdict::Ok {
        prob_a_wins: f64::from(a_wins) / f64::from(samples),
        mean_a: sum_a / f64::from(samples),
        mean_b: sum_b / f64::from(samples),
    }
}

fn sample_beta(alpha: u32, beta: u32, rng_state: &mut u64) -> f64 {
    // Approximate Beta(α, β) via two Gamma samples (Marsaglia-Tsang
    // simplified to Erlang for integer shape).
    let g_a = sample_erlang(alpha, rng_state);
    let g_b = sample_erlang(beta, rng_state);
    g_a / (g_a + g_b)
}

fn sample_erlang(shape: u32, rng_state: &mut u64) -> f64 {
    // Sum of `shape` exponentials with rate 1.
    let mut sum: f64 = 0.0;
    for _ in 0..shape {
        let u = (lcg(rng_state) >> 32) as f64 / (u32::MAX as f64);
        let u = u.max(1e-12);
        sum += -(u.ln());
    }
    sum
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_bayesian_ab_winrate")?;

    println!(
        "a clearly better: {:?}",
        simulate(2000, 100, 50, 50, 100, 42)
    );
    println!("tie: {:?}", simulate(2000, 50, 50, 50, 50, 42));
    println!("invalid: {:?}", simulate(0, 100, 50, 50, 100, 42));
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
    fn a_clearly_better_high_winrate() {
        let v = simulate(2000, 100, 10, 10, 100, 42);
        if let BayesianVerdict::Ok { prob_a_wins, .. } = v {
            assert!(prob_a_wins > 0.95);
        }
    }

    #[test]
    fn b_clearly_better_low_winrate() {
        let v = simulate(2000, 10, 100, 100, 10, 42);
        if let BayesianVerdict::Ok { prob_a_wins, .. } = v {
            assert!(prob_a_wins < 0.05);
        }
    }

    #[test]
    fn tied_close_to_half() {
        let v = simulate(2000, 50, 50, 50, 50, 42);
        if let BayesianVerdict::Ok { prob_a_wins, .. } = v {
            assert!(prob_a_wins > 0.40 && prob_a_wins < 0.60);
        }
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(
            simulate(0, 50, 50, 50, 50, 42),
            BayesianVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 50, 50, 50, 50, 42);
        let b = simulate(500, 50, 50, 50, 50, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn winrate_in_unit_range() {
        let v = simulate(500, 50, 50, 50, 50, 42);
        if let BayesianVerdict::Ok { prob_a_wins, .. } = v {
            assert!((0.0..=1.0).contains(&prob_a_wins));
        }
    }

    #[test]
    fn means_in_unit_range() {
        let v = simulate(500, 50, 50, 50, 50, 42);
        if let BayesianVerdict::Ok { mean_a, mean_b, .. } = v {
            assert!((0.0..=1.0).contains(&mean_a));
            assert!((0.0..=1.0).contains(&mean_b));
        }
    }

    #[test]
    fn higher_a_successes_higher_mean() {
        let lo = simulate(2000, 10, 100, 50, 50, 42);
        let hi = simulate(2000, 100, 10, 50, 50, 42);
        if let (BayesianVerdict::Ok { mean_a: l, .. }, BayesianVerdict::Ok { mean_a: h, .. }) =
            (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn no_observations_uniform_prior() {
        // 0,0 successes/failures → Beta(1,1) = Uniform(0,1) → P(A>B) ≈ 0.5.
        let v = simulate(2000, 0, 0, 0, 0, 42);
        if let BayesianVerdict::Ok { prob_a_wins, .. } = v {
            assert!(prob_a_wins > 0.40 && prob_a_wins < 0.60);
        }
    }

    #[test]
    fn more_samples_lower_variance() {
        let small = simulate(50, 50, 50, 50, 50, 42);
        let big = simulate(5000, 50, 50, 50, 50, 42);
        if let (
            BayesianVerdict::Ok { prob_a_wins: s, .. },
            BayesianVerdict::Ok { prob_a_wins: b, .. },
        ) = (small, big)
        {
            // Both should converge to ~0.5; smaller samples may oscillate more.
            assert!((s - 0.5).abs() < 0.30);
            assert!((b - 0.5).abs() < 0.10);
        }
    }

    #[test]
    fn winrate_within_unit_inclusive() {
        // Strong skews can hit exactly 0.0 or 1.0 with finite sampling;
        // assert just within unit range.
        let v = simulate(500, 100, 1, 1, 100, 42);
        if let BayesianVerdict::Ok { prob_a_wins, .. } = v {
            assert!((0.0..=1.0).contains(&prob_a_wins));
        }
    }
}
