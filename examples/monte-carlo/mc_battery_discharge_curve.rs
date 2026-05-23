//! # Monte-Carlo Battery Discharge Curve
//!
//! Sim a Li-ion-style battery: each cycle drops capacity by a small
//! random amount. Returns cycle count to reach 80% of original (the
//! standard "end of useful life" threshold).
//!
//! Demonstrates the **MC.67** recipe for PMAT-181 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: IEEE 1188-2005 §5; Wang et al., J. Power Sources 196 (2011).
//!
//! Run with: cargo run --example mc_battery_discharge_curve
//!
//! Added by PMAT-181 (catalog 1252→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BatteryVerdict {
    Ok {
        cycles_to_80pct: u32,
        final_capacity: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    max_cycles: u32,
    initial_capacity: f64,
    avg_loss_per_cycle: f64,
    seed: u64,
) -> BatteryVerdict {
    if max_cycles == 0 || initial_capacity <= 0.0 || avg_loss_per_cycle <= 0.0 {
        return BatteryVerdict::InvalidConfig;
    }
    let threshold = initial_capacity * 0.8;
    let mut capacity = initial_capacity;
    let mut cycles_to_80pct = max_cycles;
    let mut rng_state = seed | 1;
    for cycle in 1..=max_cycles {
        let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        let loss = avg_loss_per_cycle * (0.5 + r); // [0.5, 1.5] * avg
        capacity = (capacity - loss).max(0.0);
        if capacity <= threshold {
            cycles_to_80pct = cycle;
            break;
        }
    }
    BatteryVerdict::Ok {
        cycles_to_80pct,
        final_capacity: capacity,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_battery_discharge_curve")?;

    println!("typical: {:?}", simulate(2000, 1.0, 0.0002, 42));
    println!("rapid: {:?}", simulate(2000, 1.0, 0.001, 42));
    println!("invalid: {:?}", simulate(0, 1.0, 0.001, 42));
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
    fn typical_cycles_in_realistic_range() {
        // 0.02% loss per cycle → ~1000 cycles to 80%.
        let v = simulate(2000, 1.0, 0.0002, 42);
        if let BatteryVerdict::Ok {
            cycles_to_80pct, ..
        } = v
        {
            assert!(cycles_to_80pct >= 500 && cycles_to_80pct <= 2000);
        }
    }

    #[test]
    fn rapid_loss_fewer_cycles() {
        let slow = simulate(2000, 1.0, 0.0002, 42);
        let fast = simulate(2000, 1.0, 0.001, 42);
        if let (
            BatteryVerdict::Ok {
                cycles_to_80pct: s, ..
            },
            BatteryVerdict::Ok {
                cycles_to_80pct: f, ..
            },
        ) = (slow, fast)
        {
            assert!(f < s);
        }
    }

    #[test]
    fn invalid_zero_cycles() {
        assert_eq!(simulate(0, 1.0, 0.001, 42), BatteryVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_capacity() {
        assert_eq!(simulate(100, 0.0, 0.001, 42), BatteryVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_loss() {
        assert_eq!(simulate(100, 1.0, 0.0, 42), BatteryVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 1.0, 0.0005, 42);
        let b = simulate(1000, 1.0, 0.0005, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn final_capacity_ge_threshold_or_zero() {
        let v = simulate(2000, 1.0, 0.001, 42);
        if let BatteryVerdict::Ok { final_capacity, .. } = v {
            assert!(final_capacity >= 0.0);
            assert!(final_capacity < 1.0);
        }
    }

    #[test]
    fn small_loss_max_cycles_returned() {
        // Tiny loss → never crosses 80% within 100 cycles.
        let v = simulate(100, 1.0, 1e-6, 42);
        if let BatteryVerdict::Ok {
            cycles_to_80pct, ..
        } = v
        {
            assert_eq!(cycles_to_80pct, 100);
        }
    }

    #[test]
    fn higher_initial_more_cycles() {
        let lo = simulate(2000, 0.5, 0.0005, 42);
        let hi = simulate(2000, 2.0, 0.0005, 42);
        if let (
            BatteryVerdict::Ok {
                cycles_to_80pct: l, ..
            },
            BatteryVerdict::Ok {
                cycles_to_80pct: h, ..
            },
        ) = (lo, hi)
        {
            assert!(h >= l);
        }
    }

    #[test]
    fn cycles_le_max() {
        let v = simulate(500, 1.0, 0.001, 42);
        if let BatteryVerdict::Ok {
            cycles_to_80pct, ..
        } = v
        {
            assert!(cycles_to_80pct <= 500);
        }
    }

    #[test]
    fn cycles_at_least_one() {
        let v = simulate(1, 1.0, 1.0, 42);
        if let BatteryVerdict::Ok {
            cycles_to_80pct, ..
        } = v
        {
            assert!(cycles_to_80pct >= 1);
        }
    }
}
