//! # Monte-Carlo Disk Failure RAID
//!
//! Sim N disks under MTBF-driven failures. RAID 5 tolerates 1 disk
//! failure; RAID 6 tolerates 2. Reports data-loss probability over
//! a 1-year operating window.
//!
//! Demonstrates the **MC.96** recipe for PMAT-191 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Patterson, Gibson, Katz, "A Case for RAID" (SIGMOD 1988);
//!  MTBF reliability calculations.
//!
//! Run with: cargo run --example mc_disk_failure_raid
//!
//! Added by PMAT-191 (catalog 1342→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RaidVerdict {
    Ok {
        raid5_loss_rate: f64,
        raid6_loss_rate: f64,
    },
    InvalidConfig,
}

pub fn simulate(trials: u32, disks: u32, annual_failure_prob: f64, seed: u64) -> RaidVerdict {
    if trials == 0 || disks < 4 || !(0.0..=1.0).contains(&annual_failure_prob) {
        return RaidVerdict::InvalidConfig;
    }
    let mut raid5_losses = 0u32;
    let mut raid6_losses = 0u32;
    let mut rng_state = seed | 1;
    for _ in 0..trials {
        let mut failures = 0u32;
        for _ in 0..disks {
            let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            if r < annual_failure_prob {
                failures += 1;
            }
        }
        if failures > 1 {
            raid5_losses += 1;
        }
        if failures > 2 {
            raid6_losses += 1;
        }
    }
    RaidVerdict::Ok {
        raid5_loss_rate: f64::from(raid5_losses) / f64::from(trials),
        raid6_loss_rate: f64::from(raid6_losses) / f64::from(trials),
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_disk_failure_raid")?;

    println!("8 disks 5% annual: {:?}", simulate(2000, 8, 0.05, 42));
    println!("16 disks 10% annual: {:?}", simulate(2000, 16, 0.1, 42));
    println!("invalid: {:?}", simulate(0, 8, 0.05, 42));
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
    fn raid6_safer_than_raid5() {
        let v = simulate(2000, 8, 0.10, 42);
        if let RaidVerdict::Ok {
            raid5_loss_rate,
            raid6_loss_rate,
        } = v
        {
            assert!(raid6_loss_rate <= raid5_loss_rate);
        }
    }

    #[test]
    fn higher_failure_prob_higher_loss() {
        let lo = simulate(2000, 8, 0.01, 42);
        let hi = simulate(2000, 8, 0.30, 42);
        if let (
            RaidVerdict::Ok {
                raid5_loss_rate: l, ..
            },
            RaidVerdict::Ok {
                raid5_loss_rate: h, ..
            },
        ) = (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(0, 8, 0.05, 42), RaidVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_disks() {
        assert_eq!(simulate(100, 3, 0.05, 42), RaidVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_prob_out_of_range() {
        assert_eq!(simulate(100, 8, 1.5, 42), RaidVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 8, 0.05, 42);
        let b = simulate(500, 8, 0.05, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn rates_in_unit_range() {
        let v = simulate(500, 8, 0.05, 42);
        if let RaidVerdict::Ok {
            raid5_loss_rate,
            raid6_loss_rate,
        } = v
        {
            assert!((0.0..=1.0).contains(&raid5_loss_rate));
            assert!((0.0..=1.0).contains(&raid6_loss_rate));
        }
    }

    #[test]
    fn zero_failure_no_loss() {
        let v = simulate(100, 8, 0.0, 42);
        if let RaidVerdict::Ok {
            raid5_loss_rate,
            raid6_loss_rate,
        } = v
        {
            assert_eq!(raid5_loss_rate, 0.0);
            assert_eq!(raid6_loss_rate, 0.0);
        }
    }

    #[test]
    fn always_failing_full_loss() {
        let v = simulate(100, 8, 1.0, 42);
        if let RaidVerdict::Ok {
            raid5_loss_rate,
            raid6_loss_rate,
        } = v
        {
            assert_eq!(raid5_loss_rate, 1.0);
            assert_eq!(raid6_loss_rate, 1.0);
        }
    }

    #[test]
    fn larger_pool_higher_failure() {
        let small = simulate(2000, 4, 0.10, 42);
        let big = simulate(2000, 32, 0.10, 42);
        if let (
            RaidVerdict::Ok {
                raid5_loss_rate: s, ..
            },
            RaidVerdict::Ok {
                raid5_loss_rate: l, ..
            },
        ) = (small, big)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn small_failure_rare_loss() {
        let v = simulate(2000, 8, 0.001, 42);
        if let RaidVerdict::Ok {
            raid5_loss_rate, ..
        } = v
        {
            assert!(raid5_loss_rate < 0.05);
        }
    }
}
