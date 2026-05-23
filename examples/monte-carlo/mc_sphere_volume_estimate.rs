//! # Monte-Carlo n-Dimensional Sphere Volume Estimate
//!
//! Estimate the volume of the unit n-ball inscribed in the
//! n-dimensional hypercube [-1,1]^n using random sampling. Returns
//! estimate ×1000 and the dimension used.
//!
//! Demonstrates the **MC.143** recipe for PMAT-206 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hammersley & Handscomb (1964); Vol(B^n) = π^(n/2) /
//!  Γ(n/2+1) closed form for verification.
//!
//! Run with: cargo run --example mc_sphere_volume_estimate
//!
//! Added by PMAT-206 (catalog 1477→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SphereVerdict {
    Ok { volume_x1000: u32, dimension: u32 },
    InvalidConfig,
}

pub fn estimate(dimension: u32, samples: u32, seed: u64) -> SphereVerdict {
    if !(1..=20).contains(&dimension) || samples < 100 {
        return SphereVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut hits = 0u32;
    for _ in 0..samples {
        let mut r2 = 0.0f64;
        for _ in 0..dimension {
            // Sample x in [-1, 1]
            let u = (lcg(&mut state) as f64) / (u32::MAX as f64);
            let x = 2.0 * u - 1.0;
            r2 += x * x;
        }
        if r2 <= 1.0 {
            hits += 1;
        }
    }
    let cube_volume = 2.0_f64.powi(dimension as i32);
    let volume = (hits as f64 / samples as f64) * cube_volume;
    SphereVerdict::Ok {
        volume_x1000: (volume * 1000.0) as u32,
        dimension,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_sphere_volume_estimate")?;

    // 2D unit ball: π ≈ 3.14159 → 3142
    println!("dim-2: {:?}", estimate(2, 100_000, 42));
    // 3D unit ball: 4π/3 ≈ 4.18879 → 4189
    println!("dim-3: {:?}", estimate(3, 100_000, 42));
    println!("invalid: {:?}", estimate(0, 100_000, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_zero_dim() {
        assert_eq!(estimate(0, 1000, 42), SphereVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_high_dim() {
        assert_eq!(estimate(21, 1000, 42), SphereVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_samples() {
        assert_eq!(estimate(2, 50, 42), SphereVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = estimate(2, 500, 42);
        let b = estimate(2, 500, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn dim_2_estimates_pi() {
        // Vol(B²) = π ≈ 3.142. Allow ±5% at 100k samples.
        let v = estimate(2, 100_000, 42);
        if let SphereVerdict::Ok { volume_x1000, .. } = v {
            assert!((2900..=3300).contains(&volume_x1000));
        }
    }

    #[test]
    fn dim_3_estimates_4pi_3() {
        // Vol(B³) = 4π/3 ≈ 4.189. Allow ±5% at 100k samples.
        let v = estimate(3, 100_000, 42);
        if let SphereVerdict::Ok { volume_x1000, .. } = v {
            assert!((3950..=4400).contains(&volume_x1000));
        }
    }

    #[test]
    fn dimension_returned() {
        let v = estimate(5, 1000, 42);
        if let SphereVerdict::Ok { dimension, .. } = v {
            assert_eq!(dimension, 5);
        }
    }

    #[test]
    fn dim_1_estimates_2() {
        // Unit ball in 1D = [-1,1] → volume 2.
        let v = estimate(1, 100_000, 42);
        if let SphereVerdict::Ok { volume_x1000, .. } = v {
            assert!((1980..=2020).contains(&volume_x1000));
        }
    }

    #[test]
    fn higher_dim_smaller_relative_volume() {
        // Curse of dimensionality: ratio Vol(B^n)/Vol(cube^n) → 0 as n → ∞.
        let dim_5 = estimate(5, 100_000, 42);
        let dim_10 = estimate(10, 100_000, 42);
        if let (
            SphereVerdict::Ok {
                volume_x1000: v5, ..
            },
            SphereVerdict::Ok {
                volume_x1000: v10, ..
            },
        ) = (dim_5, dim_10)
        {
            // Vol(B^5)≈5.26, Vol(B^10)≈2.55.
            let ratio_5 = v5 as f64 / 32_000.0; // / 2^5*1000
            let ratio_10 = v10 as f64 / 1_024_000.0; // / 2^10*1000
            assert!(ratio_10 < ratio_5);
        }
    }

    #[test]
    fn high_dim_handled() {
        let v = estimate(15, 100_000, 42);
        assert!(matches!(v, SphereVerdict::Ok { .. }));
    }

    #[test]
    fn min_samples_accepted() {
        let v = estimate(2, 100, 42);
        assert!(matches!(v, SphereVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_estimates() {
        let a = estimate(3, 500, 42);
        let b = estimate(3, 500, 999);
        assert!(a != b);
    }
}
