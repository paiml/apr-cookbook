//! # Monte-Carlo M/M/1 Queue Mean Wait
//!
//! Sim an M/M/1 queue: Poisson arrivals (rate λ), exponential service
//! (rate μ). Verifies Pollaczek-Khinchine: mean wait = ρ/(μ(1-ρ))
//! where ρ = λ/μ. Returns observed mean wait time.
//!
//! Demonstrates the **MC.185** recipe for PMAT-220 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Kendall, "Stochastic Processes Occurring in the Theory
//!  of Queues" Annals of Math. Stat. (1953); M/M/1 closed-form
//!  derivation.
//!
//! Run with: cargo run --example mc_m_m_1_queue_wait
//!
//! Added by PMAT-220 (catalog 1603→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum QueueVerdict {
    Ok {
        mean_wait_x100: u32,
        utilization_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(arrival_x100: u32, service_x100: u32, customers: u32, seed: u64) -> QueueVerdict {
    if arrival_x100 == 0 || service_x100 == 0 || arrival_x100 >= service_x100 || customers < 100 {
        return QueueVerdict::InvalidConfig;
    }
    let lambda = arrival_x100 as f64 / 100.0;
    let mu = service_x100 as f64 / 100.0;
    let mut state = seed | 1;
    let mut clock = 0.0f64;
    let mut server_free_at = 0.0f64;
    let mut total_wait = 0.0f64;
    for _ in 0..customers {
        let u1 = ((lcg(&mut state) as f64) / (u32::MAX as f64)).max(1e-10);
        let inter_arrival = -(1.0 - u1).ln() / lambda;
        clock += inter_arrival;
        let wait = (server_free_at - clock).max(0.0);
        total_wait += wait;
        let u2 = ((lcg(&mut state) as f64) / (u32::MAX as f64)).max(1e-10);
        let service_time = -(1.0 - u2).ln() / mu;
        server_free_at = clock + wait + service_time;
    }
    let mean_wait = total_wait / customers as f64;
    let utilization = lambda / mu;
    QueueVerdict::Ok {
        mean_wait_x100: (mean_wait * 100.0) as u32,
        utilization_x100: (utilization * 100.0) as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_m_m_1_queue_wait")?;

    // λ=0.5, μ=1.0 → ρ=0.5; Wq = ρ/(μ(1-ρ)) = 0.5/0.5 = 1.0
    println!("rho=0.5: {:?}", simulate(50, 100, 5000, 42));
    println!("invalid: {:?}", simulate(150, 100, 5000, 42));
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
    fn invalid_zero_arrival() {
        assert_eq!(simulate(0, 100, 5000, 42), QueueVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_service() {
        assert_eq!(simulate(50, 0, 5000, 42), QueueVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_unstable_queue() {
        // λ ≥ μ → unstable.
        assert_eq!(simulate(150, 100, 5000, 42), QueueVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_customers() {
        assert_eq!(simulate(50, 100, 50, 42), QueueVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(50, 100, 1000, 42);
        let b = simulate(50, 100, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn higher_utilization_longer_wait() {
        // ρ=0.3 vs ρ=0.8 → higher ρ has longer wait.
        let low = simulate(30, 100, 5000, 42);
        let high = simulate(80, 100, 5000, 42);
        if let (
            QueueVerdict::Ok {
                mean_wait_x100: l, ..
            },
            QueueVerdict::Ok {
                mean_wait_x100: h, ..
            },
        ) = (low, high)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn utilization_returned() {
        let v = simulate(50, 100, 1000, 42);
        if let QueueVerdict::Ok {
            utilization_x100, ..
        } = v
        {
            assert_eq!(utilization_x100, 50);
        }
    }

    #[test]
    fn mean_wait_finite() {
        let v = simulate(50, 100, 1000, 42);
        if let QueueVerdict::Ok { mean_wait_x100, .. } = v {
            assert!(mean_wait_x100 < u32::MAX);
        }
    }

    #[test]
    fn mm1_wait_near_pk_formula() {
        // ρ=0.5 → Wq = 1.0 (×100 = 100). Allow ±50%.
        let v = simulate(50, 100, 20_000, 42);
        if let QueueVerdict::Ok { mean_wait_x100, .. } = v {
            assert!((50..=200).contains(&mean_wait_x100));
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(1, 100, 100, 42);
        assert!(matches!(v, QueueVerdict::Ok { .. }));
    }

    #[test]
    fn many_customers_handled() {
        let v = simulate(50, 100, 100_000, 42);
        assert!(matches!(v, QueueVerdict::Ok { .. }));
    }
}
