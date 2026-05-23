//! # Monte-Carlo Gaussian KDE Density Estimate
//!
//! Build a Gaussian Kernel Density Estimator from sample data and
//! evaluate density at query points. Returns density values (×1000)
//! at each query.
//!
//! Demonstrates the **MC.179** recipe for PMAT-218 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Parzen, "On Estimation of a Probability Density Function"
//!  Annals of Math. Stat. 33(3) (1962); Silverman bandwidth (1986).
//!
//! Run with: cargo run --example mc_gaussian_kde_estimate
//!
//! Added by PMAT-218 (catalog 1585→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum KdeVerdict {
    Ok {
        density_x1000: Vec<u32>,
        bandwidth_x100: u32,
    },
    InvalidConfig,
}

pub fn estimate(samples: &[i32], queries: &[i32], bandwidth_x100: u32) -> KdeVerdict {
    if samples.is_empty() || queries.is_empty() || bandwidth_x100 == 0 {
        return KdeVerdict::InvalidConfig;
    }
    let h = bandwidth_x100 as f64 / 100.0;
    let n = samples.len() as f64;
    let norm = 1.0 / (n * h * (2.0 * std::f64::consts::PI).sqrt());
    let mut densities: Vec<u32> = Vec::with_capacity(queries.len());
    for q in queries {
        let mut sum = 0.0f64;
        for s in samples {
            let z = (*q as f64 - *s as f64) / h;
            sum += (-0.5 * z * z).exp();
        }
        let density = norm * sum;
        densities.push((density * 1000.0) as u32);
    }
    KdeVerdict::Ok {
        density_x1000: densities,
        bandwidth_x100,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_gaussian_kde_estimate")?;

    let samples = vec![-2, -1, 0, 1, 2];
    let queries = vec![-3, 0, 3];
    println!("kde: {:?}", estimate(&samples, &queries, 100));
    println!("invalid: {:?}", estimate(&[], &[], 100));
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
    fn empty_samples_rejected() {
        assert_eq!(estimate(&[], &[1], 100), KdeVerdict::InvalidConfig);
    }

    #[test]
    fn empty_queries_rejected() {
        assert_eq!(estimate(&[1], &[], 100), KdeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_bandwidth() {
        assert_eq!(estimate(&[1], &[1], 0), KdeVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = estimate(&[1, 2, 3], &[2], 100);
        let b = estimate(&[1, 2, 3], &[2], 100);
        assert_eq!(a, b);
    }

    #[test]
    fn density_at_sample_high() {
        // Density at a sample point is high.
        let v = estimate(&[0; 100], &[0], 100);
        if let KdeVerdict::Ok { density_x1000, .. } = v {
            assert!(density_x1000[0] > 100);
        }
    }

    #[test]
    fn density_far_from_samples_low() {
        let v = estimate(&[0], &[1000], 10);
        if let KdeVerdict::Ok { density_x1000, .. } = v {
            assert_eq!(density_x1000[0], 0);
        }
    }

    #[test]
    fn density_count_matches_queries() {
        let v = estimate(&[1, 2], &[1, 2, 3], 100);
        if let KdeVerdict::Ok { density_x1000, .. } = v {
            assert_eq!(density_x1000.len(), 3);
        }
    }

    #[test]
    fn bandwidth_returned() {
        let v = estimate(&[1], &[1], 250);
        if let KdeVerdict::Ok { bandwidth_x100, .. } = v {
            assert_eq!(bandwidth_x100, 250);
        }
    }

    #[test]
    fn larger_bandwidth_smoother() {
        // At a query point far from any sample, larger bandwidth gives
        // a higher (non-zero) density due to wider Gaussian tails.
        let small = estimate(&[0], &[10], 10);
        let large = estimate(&[0], &[10], 1000);
        if let (
            KdeVerdict::Ok {
                density_x1000: s, ..
            },
            KdeVerdict::Ok {
                density_x1000: l, ..
            },
        ) = (small, large)
        {
            assert!(l[0] >= s[0]);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = estimate(&[1], &[1], 100);
        assert!(matches!(v, KdeVerdict::Ok { .. }));
    }

    #[test]
    fn many_samples_handled() {
        let samples: Vec<i32> = (0..1000).collect();
        let queries: Vec<i32> = vec![500];
        let v = estimate(&samples, &queries, 100);
        assert!(matches!(v, KdeVerdict::Ok { .. }));
    }

    #[test]
    fn negative_query_handled() {
        let v = estimate(&[0], &[-10], 100);
        assert!(matches!(v, KdeVerdict::Ok { .. }));
    }
}
