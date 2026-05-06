//! # Monte-Carlo Pareto Request-Size Generator
//!
//! Generate request sizes from a Pareto distribution (heavy-tailed):
//! `size = scale * (1 - U)^(-1/alpha)`. Returns mean / median / max
//! observed sizes.
//!
//! Demonstrates the **MC.21** recipe for PMAT-164 (catalog crosses 1100).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Pareto distribution (heavy-tail networking, Crovella & Bestavros).
//!
//! Run with: cargo run --example mc_request_size_pareto
//!
//! Added by PMAT-164 (catalog 1099→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ParetoVerdict {
    Ok {
        mean_size: f64,
        median_size: f64,
        max_size: f64,
        cap_hits: u32,
    },
    InvalidConfig,
}

pub fn generate(scale: f64, alpha: f64, samples: u32, cap: f64, seed: u64) -> ParetoVerdict {
    if !scale.is_finite()
        || scale <= 0.0
        || !alpha.is_finite()
        || alpha <= 0.0
        || samples == 0
        || !cap.is_finite()
        || cap <= scale
    {
        return ParetoVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut sizes: Vec<f64> = Vec::with_capacity(samples as usize);
    let mut cap_hits = 0u32;
    for _ in 0..samples {
        let u = unit(&mut rng_state).clamp(1e-12, 1.0 - 1e-12);
        let raw = scale * (1.0 - u).powf(-1.0 / alpha);
        let size = if raw > cap {
            cap_hits += 1;
            cap
        } else {
            raw
        };
        sizes.push(size);
    }
    let mean_size = sizes.iter().sum::<f64>() / f64::from(samples);
    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_size = sizes[sizes.len() / 2];
    let max_size = *sizes.last().unwrap_or(&scale);
    ParetoVerdict::Ok {
        mean_size,
        median_size,
        max_size,
        cap_hits,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_request_size_pareto")?;

    println!(
        "heavy tail: {:?}",
        generate(1024.0, 1.5, 10_000, 1_000_000.0, 42)
    );
    println!(
        "light tail: {:?}",
        generate(1024.0, 5.0, 10_000, 1_000_000.0, 42)
    );
    println!(
        "invalid: {:?}",
        generate(1024.0, -1.0, 10_000, 1_000_000.0, 42)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn heavy_tail_skews_above_median() {
        let v = generate(1024.0, 1.5, 10_000, 1_000_000.0, 42);
        if let ParetoVerdict::Ok {
            mean_size,
            median_size,
            ..
        } = v
        {
            // Pareto: mean > median.
            assert!(mean_size > median_size);
        }
    }

    #[test]
    fn light_tail_lower_max() {
        let heavy = generate(1024.0, 1.5, 10_000, 1_000_000.0, 42);
        let light = generate(1024.0, 5.0, 10_000, 1_000_000.0, 42);
        if let (ParetoVerdict::Ok { max_size: h, .. }, ParetoVerdict::Ok { max_size: l, .. }) =
            (heavy, light)
        {
            assert!(h >= l);
        }
    }

    #[test]
    fn invalid_zero_scale() {
        assert_eq!(
            generate(0.0, 1.5, 100, 1000.0, 42),
            ParetoVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_alpha() {
        assert_eq!(
            generate(1024.0, 0.0, 100, 1_000_000.0, 42),
            ParetoVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(
            generate(1024.0, 1.5, 0, 1_000_000.0, 42),
            ParetoVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_cap_below_scale() {
        assert_eq!(
            generate(1024.0, 1.5, 100, 100.0, 42),
            ParetoVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            generate(f64::NAN, 1.5, 100, 1_000_000.0, 42),
            ParetoVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = generate(1024.0, 1.5, 100, 1_000_000.0, 42);
        let b = generate(1024.0, 1.5, 100, 1_000_000.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn min_size_at_least_scale() {
        let v = generate(1024.0, 1.5, 10_000, 1_000_000.0, 42);
        if let ParetoVerdict::Ok { median_size, .. } = v {
            // Median ≥ scale (minimum of Pareto).
            assert!(median_size >= 1024.0);
        }
    }

    #[test]
    fn max_capped_at_cap() {
        let v = generate(1024.0, 0.5, 10_000, 5000.0, 42);
        if let ParetoVerdict::Ok {
            max_size, cap_hits, ..
        } = v
        {
            assert!(max_size <= 5000.0 + 1e-6);
            assert!(cap_hits > 0);
        }
    }
}
