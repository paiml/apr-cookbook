//! # Monte-Carlo Drunkard's Walk Cliff
//!
//! Sim N drunkards on a 1D line: each step ±1 uniformly random.
//! Falls off cliff at +cliff_pos. Reports fraction who fall in
//! `max_steps` and mean steps to fall.
//!
//! Demonstrates the **MC.123** recipe for PMAT-200 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: simple random walk (Pearson, Nature 1905); first-passage
//!  time analysis.
//!
//! Run with: cargo run --example mc_random_walk_drunkard
//!
//! Added by PMAT-200 (catalog 1423→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DrunkardVerdict {
    Ok {
        fall_rate: f64,
        mean_steps_to_fall: f64,
    },
    InvalidConfig,
}

pub fn simulate(drunkards: u32, cliff_pos: i32, max_steps: u32, seed: u64) -> DrunkardVerdict {
    if drunkards == 0 || cliff_pos == 0 || max_steps == 0 {
        return DrunkardVerdict::InvalidConfig;
    }
    let mut fell = 0u32;
    let mut total_fall_steps: u64 = 0;
    let mut rng_state = seed | 1;
    for _ in 0..drunkards {
        let mut pos: i32 = 0;
        let mut steps_to_fall = 0u32;
        for step in 0..max_steps {
            let dir = (lcg(&mut rng_state) >> 32) % 2;
            pos += if dir == 0 { 1 } else { -1 };
            if pos.abs() >= cliff_pos.abs() {
                fell += 1;
                steps_to_fall = step + 1;
                break;
            }
        }
        if steps_to_fall > 0 {
            total_fall_steps += u64::from(steps_to_fall);
        }
    }
    let fall_rate = f64::from(fell) / f64::from(drunkards);
    let mean_steps_to_fall = if fell > 0 {
        total_fall_steps as f64 / f64::from(fell)
    } else {
        0.0
    };
    DrunkardVerdict::Ok {
        fall_rate,
        mean_steps_to_fall,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_random_walk_drunkard")?;

    println!("close cliff: {:?}", simulate(1000, 5, 1000, 42));
    println!("far cliff: {:?}", simulate(1000, 50, 1000, 42));
    println!("invalid: {:?}", simulate(0, 5, 1000, 42));
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
    fn close_cliff_high_fall() {
        let v = simulate(2000, 3, 5000, 42);
        if let DrunkardVerdict::Ok { fall_rate, .. } = v {
            assert!(fall_rate > 0.95);
        }
    }

    #[test]
    fn far_cliff_lower_fall() {
        let close = simulate(1000, 5, 1000, 42);
        let far = simulate(1000, 100, 1000, 42);
        if let (
            DrunkardVerdict::Ok { fall_rate: c, .. },
            DrunkardVerdict::Ok { fall_rate: f, .. },
        ) = (close, far)
        {
            assert!(f < c);
        }
    }

    #[test]
    fn invalid_zero_drunkards() {
        assert_eq!(simulate(0, 5, 1000, 42), DrunkardVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_cliff() {
        assert_eq!(simulate(100, 0, 1000, 42), DrunkardVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(simulate(100, 5, 0, 42), DrunkardVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 5, 1000, 42);
        let b = simulate(100, 5, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn fall_rate_in_unit_range() {
        let v = simulate(500, 5, 1000, 42);
        if let DrunkardVerdict::Ok { fall_rate, .. } = v {
            assert!((0.0..=1.0).contains(&fall_rate));
        }
    }

    #[test]
    fn mean_steps_at_least_cliff() {
        let v = simulate(500, 5, 5000, 42);
        if let DrunkardVerdict::Ok {
            mean_steps_to_fall, ..
        } = v
        {
            // Min steps to fall ≥ cliff_pos.
            assert!(mean_steps_to_fall >= 5.0);
        }
    }

    #[test]
    fn negative_cliff_works() {
        let v = simulate(500, -5, 1000, 42);
        assert!(matches!(v, DrunkardVerdict::Ok { .. }));
    }

    #[test]
    fn many_drunkards_handled() {
        let v = simulate(10_000, 10, 500, 42);
        assert!(matches!(v, DrunkardVerdict::Ok { .. }));
    }

    #[test]
    fn finite_outputs() {
        let v = simulate(100, 5, 1000, 42);
        if let DrunkardVerdict::Ok {
            fall_rate,
            mean_steps_to_fall,
        } = v
        {
            assert!(fall_rate.is_finite());
            assert!(mean_steps_to_fall.is_finite());
        }
    }
}
