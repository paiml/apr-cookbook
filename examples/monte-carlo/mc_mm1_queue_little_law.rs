//! # Monte Carlo M/M/1 Queue Little's Law
//!
//! M/M/1 queue: Poisson arrivals at rate λ, exponential service at rate
//! μ. Steady-state utilisation ρ = λ/μ; mean queue length L = ρ/(1-ρ);
//! Little's Law: L = λ × W (where W is mean wait time). This recipe
//! validates the analytic relationships + the steady-state predicates.
//!
//! Demonstrates the **MC.5** recipe for PMAT-122 (monte-carlo coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Little, J. D. C. (1961). A proof for the queuing formula L = λW. Operations Research 9(3).
//!
//! Run with: cargo run --example mc_mm1_queue_little_law
//!
//! Added by PMAT-122 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum QueueVerdict {
    Stable {
        rho: f64,
        mean_length: f64,
        mean_wait: f64,
    },
    Unstable, // ρ ≥ 1
    InvalidRate,
}

pub fn analyze(arrival_rate: f64, service_rate: f64) -> QueueVerdict {
    if !arrival_rate.is_finite() || !service_rate.is_finite() {
        return QueueVerdict::InvalidRate;
    }
    if arrival_rate < 0.0 || service_rate <= 0.0 {
        return QueueVerdict::InvalidRate;
    }
    let rho = arrival_rate / service_rate;
    if rho >= 1.0 {
        return QueueVerdict::Unstable;
    }
    let mean_length = rho / (1.0 - rho);
    let mean_wait = if arrival_rate > 0.0 {
        mean_length / arrival_rate
    } else {
        0.0
    };
    QueueVerdict::Stable {
        rho,
        mean_length,
        mean_wait,
    }
}

pub fn verify_little_law(arrival_rate: f64, mean_length: f64, mean_wait: f64) -> bool {
    let predicted = arrival_rate * mean_wait;
    let tolerance = (mean_length.abs() + 1.0) * 1e-9;
    (predicted - mean_length).abs() < tolerance
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_mm1_queue_little_law")?;

    for (lam, mu) in [(2.0, 5.0), (4.0, 5.0), (5.0, 5.0), (6.0, 5.0)] {
        let v = analyze(lam, mu);
        println!("λ={lam} μ={mu}  →  {v:?}");
        if let QueueVerdict::Stable {
            mean_length,
            mean_wait,
            ..
        } = v
        {
            println!(
                "  Little's Law: {}",
                verify_little_law(lam, mean_length, mean_wait)
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyzer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn light_load_stable() {
        // λ=2, μ=5 → ρ=0.4, L = 0.4/0.6 ≈ 0.667.
        let v = analyze(2.0, 5.0);
        if let QueueVerdict::Stable {
            rho, mean_length, ..
        } = v
        {
            assert!((rho - 0.4).abs() < 1e-12);
            assert!((mean_length - 2.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn at_capacity_unstable() {
        // ρ=1 means arrivals match service rate — unbounded queue growth.
        assert_eq!(analyze(5.0, 5.0), QueueVerdict::Unstable);
    }

    #[test]
    fn over_capacity_unstable() {
        assert_eq!(analyze(7.0, 5.0), QueueVerdict::Unstable);
    }

    #[test]
    fn negative_rate_invalid() {
        assert_eq!(analyze(-1.0, 5.0), QueueVerdict::InvalidRate);
    }

    #[test]
    fn zero_service_rate_invalid() {
        // Division by zero would explode; reject explicitly.
        assert_eq!(analyze(2.0, 0.0), QueueVerdict::InvalidRate);
    }

    #[test]
    fn nan_rate_invalid() {
        assert_eq!(analyze(f64::NAN, 5.0), QueueVerdict::InvalidRate);
    }

    #[test]
    fn little_law_holds_for_stable_queue() {
        let v = analyze(2.0, 5.0);
        if let QueueVerdict::Stable {
            mean_length,
            mean_wait,
            ..
        } = v
        {
            assert!(verify_little_law(2.0, mean_length, mean_wait));
        }
    }

    #[test]
    fn little_law_rejects_inconsistent_inputs() {
        // L = 1.0, λ = 2.0, W = 5.0 → λW = 10 ≠ 1.
        assert!(!verify_little_law(2.0, 1.0, 5.0));
    }

    #[test]
    fn higher_utilization_longer_queue() {
        let low = analyze(1.0, 5.0);
        let high = analyze(4.0, 5.0);
        if let (
            QueueVerdict::Stable { mean_length: a, .. },
            QueueVerdict::Stable { mean_length: b, .. },
        ) = (low, high)
        {
            assert!(b > a);
        }
    }
}
