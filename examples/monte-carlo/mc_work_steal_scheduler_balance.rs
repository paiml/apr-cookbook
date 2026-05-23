//! # Monte-Carlo Work-Stealing Scheduler Load Balance
//!
//! Sim a work-stealing scheduler with N workers and a task queue.
//! At each tick, idle workers steal from the busiest neighbor.
//! Returns final load distribution std-dev (×100) and total ticks.
//!
//! Demonstrates the **MC.169** recipe for PMAT-215 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Cilk work-stealing scheduler (Frigo et al. 1998); cf.
//!  simular's scheduler at ../aprender/crates/aprender-simulate/src/
//!  engine/scheduler.rs.
//!
//! Run with: cargo run --example mc_work_steal_scheduler_balance
//!
//! Added by PMAT-215 (catalog 1558→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SchedVerdict {
    Ok {
        final_stddev_x100: u32,
        ticks_simulated: u32,
    },
    InvalidConfig,
}

pub fn simulate(workers: u32, initial_imbalance: u32, ticks: u32, seed: u64) -> SchedVerdict {
    if workers < 2 || initial_imbalance == 0 || ticks == 0 {
        return SchedVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut loads: Vec<u32> = vec![0; workers as usize];
    loads[0] = initial_imbalance;
    for _ in 0..ticks {
        // Pick a random pair; if one has > the other by > 1, steal half the diff.
        let i = (lcg(&mut state) as usize) % workers as usize;
        let mut j = (lcg(&mut state) as usize) % workers as usize;
        while j == i {
            j = (lcg(&mut state) as usize) % workers as usize;
        }
        if loads[i] > loads[j] + 1 {
            let steal = (loads[i] - loads[j]) / 2;
            loads[i] -= steal;
            loads[j] += steal;
        } else if loads[j] > loads[i] + 1 {
            let steal = (loads[j] - loads[i]) / 2;
            loads[j] -= steal;
            loads[i] += steal;
        }
    }
    let mean = loads.iter().sum::<u32>() as f64 / loads.len() as f64;
    let var = loads
        .iter()
        .map(|l| (*l as f64 - mean).powi(2))
        .sum::<f64>()
        / loads.len() as f64;
    let stddev = var.sqrt();
    SchedVerdict::Ok {
        final_stddev_x100: (stddev * 100.0) as u32,
        ticks_simulated: ticks,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_work_steal_scheduler_balance")?;

    println!("8 workers: {:?}", simulate(8, 1000, 1000, 42));
    println!("invalid: {:?}", simulate(1, 100, 100, 42));
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
    fn invalid_too_few_workers() {
        assert_eq!(simulate(1, 100, 100, 42), SchedVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_imbalance() {
        assert_eq!(simulate(4, 0, 100, 42), SchedVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_ticks() {
        assert_eq!(simulate(4, 100, 0, 42), SchedVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(4, 100, 100, 42);
        let b = simulate(4, 100, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn stealing_reduces_stddev() {
        // After many ticks, work should equalize.
        let v = simulate(8, 1000, 5000, 42);
        if let SchedVerdict::Ok {
            final_stddev_x100, ..
        } = v
        {
            // Initial stddev for 1000-imbalance over 8 workers is ~330; after 5000
            // steals it should be near zero.
            assert!(final_stddev_x100 < 100);
        }
    }

    #[test]
    fn ticks_returned() {
        let v = simulate(4, 100, 250, 42);
        if let SchedVerdict::Ok {
            ticks_simulated, ..
        } = v
        {
            assert_eq!(ticks_simulated, 250);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(2, 1, 1, 42);
        assert!(matches!(v, SchedVerdict::Ok { .. }));
    }

    #[test]
    fn many_workers_handled() {
        let v = simulate(50, 1000, 1000, 42);
        assert!(matches!(v, SchedVerdict::Ok { .. }));
    }

    #[test]
    fn many_ticks_handled() {
        let v = simulate(8, 100, 100_000, 42);
        assert!(matches!(v, SchedVerdict::Ok { .. }));
    }

    #[test]
    fn finite_stddev() {
        let v = simulate(4, 100, 100, 42);
        if let SchedVerdict::Ok {
            final_stddev_x100, ..
        } = v
        {
            assert!(final_stddev_x100 < u32::MAX);
        }
    }

    #[test]
    fn no_ticks_no_balancing_high_stddev() {
        let v = simulate(8, 1000, 1, 42);
        if let SchedVerdict::Ok {
            final_stddev_x100, ..
        } = v
        {
            // After 1 tick from severe imbalance → stddev still high.
            assert!(final_stddev_x100 > 1000);
        }
    }
}
