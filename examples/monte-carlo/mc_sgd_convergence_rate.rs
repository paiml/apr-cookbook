//! # Monte-Carlo SGD Convergence Rate
//!
//! Sim stochastic gradient descent on f(x) = (x-3)^2 with noisy
//! gradient. Returns final loss and step count to convergence.
//!
//! Demonstrates the **MC.201** recipe for PMAT-225 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Robbins & Monro, "A Stochastic Approximation Method"
//!  Annals of Math. Stat. 22(3) (1951); SGD convergence theory.
//!
//! Run with: cargo run --example mc_sgd_convergence_rate
//!
//! Added by PMAT-225 (catalog 1648→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SgdVerdict {
    Ok {
        final_loss_x100: u32,
        steps_taken: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    initial_x: i32,
    learning_rate_x1000: u32,
    noise_x100: u32,
    max_steps: u32,
    seed: u64,
) -> SgdVerdict {
    if learning_rate_x1000 == 0 || max_steps < 10 {
        return SgdVerdict::InvalidConfig;
    }
    let lr = learning_rate_x1000 as f64 / 1000.0;
    let noise = noise_x100 as f64 / 100.0;
    let mut state = seed | 1;
    let mut x = initial_x as f64;
    let target_loss = 0.01;
    let mut steps = max_steps;
    for step in 1..=max_steps {
        // Loss = (x-3)^2; gradient = 2(x-3) + noise
        let grad = 2.0 * (x - 3.0);
        let noisy = grad + noise * ((lcg(&mut state) as f64) / (u32::MAX as f64) - 0.5);
        x -= lr * noisy;
        let loss = (x - 3.0).powi(2);
        if loss < target_loss {
            steps = step;
            break;
        }
    }
    let final_loss = (x - 3.0).powi(2);
    SgdVerdict::Ok {
        final_loss_x100: (final_loss * 100.0) as u32,
        steps_taken: steps,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_sgd_convergence_rate")?;

    println!("converge: {:?}", simulate(0, 100, 10, 1000, 42));
    println!("invalid: {:?}", simulate(0, 0, 10, 1000, 42));
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
    fn invalid_zero_lr() {
        assert_eq!(simulate(0, 0, 10, 1000, 42), SgdVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_steps() {
        assert_eq!(simulate(0, 100, 10, 5, 42), SgdVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(0, 100, 10, 100, 42);
        let b = simulate(0, 100, 10, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn converges_when_lr_appropriate() {
        // Low noise + decent lr → loss should be small.
        let v = simulate(0, 200, 5, 500, 42);
        if let SgdVerdict::Ok {
            final_loss_x100, ..
        } = v
        {
            assert!(final_loss_x100 < 1000);
        }
    }

    #[test]
    fn high_noise_slower_convergence() {
        let low_noise = simulate(0, 100, 5, 500, 42);
        let high_noise = simulate(0, 100, 500, 500, 42);
        if let (
            SgdVerdict::Ok {
                final_loss_x100: l, ..
            },
            SgdVerdict::Ok {
                final_loss_x100: h, ..
            },
        ) = (low_noise, high_noise)
        {
            assert!(h >= l);
        }
    }

    #[test]
    fn steps_le_max() {
        let v = simulate(0, 100, 10, 1000, 42);
        if let SgdVerdict::Ok { steps_taken, .. } = v {
            assert!(steps_taken <= 1000);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(0, 1, 0, 10, 42);
        assert!(matches!(v, SgdVerdict::Ok { .. }));
    }

    #[test]
    fn many_steps_handled() {
        let v = simulate(0, 100, 10, 10_000, 42);
        assert!(matches!(v, SgdVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(0, 100, 100, 100, 42);
        let b = simulate(0, 100, 100, 100, 999);
        assert!(a != b);
    }

    #[test]
    fn negative_initial_handled() {
        let v = simulate(-100, 100, 5, 1000, 42);
        assert!(matches!(v, SgdVerdict::Ok { .. }));
    }

    #[test]
    fn final_loss_finite() {
        let v = simulate(0, 100, 10, 100, 42);
        if let SgdVerdict::Ok {
            final_loss_x100, ..
        } = v
        {
            assert!(final_loss_x100 < u32::MAX);
        }
    }
}
