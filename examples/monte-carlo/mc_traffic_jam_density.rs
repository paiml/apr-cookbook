//! # Monte-Carlo Highway Traffic Jam Density (Nagel-Schreckenberg)
//!
//! Sim 1D cellular automaton highway with N cells; cars accelerate
//! up to v_max, slow if blocked, with random brake prob. Reports
//! jam fraction (cars stopped) and avg velocity.
//!
//! Demonstrates the **MC.114** recipe for PMAT-197 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Nagel & Schreckenberg, J. Phys. I France 2 (1992)
//!  cellular automaton model.
//!
//! Run with: cargo run --example mc_traffic_jam_density
//!
//! Added by PMAT-197 (catalog 1396→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TrafficVerdict {
    Ok {
        jam_fraction: f64,
        avg_velocity: f64,
    },
    InvalidConfig,
}

#[allow(clippy::too_many_arguments)]
pub fn simulate(
    cells: u32,
    cars: u32,
    v_max: u32,
    steps: u32,
    brake_prob: f64,
    seed: u64,
) -> TrafficVerdict {
    if cells == 0
        || cars == 0
        || cars > cells
        || v_max == 0
        || steps == 0
        || !(0.0..=1.0).contains(&brake_prob)
    {
        return TrafficVerdict::InvalidConfig;
    }
    let n = cells as usize;
    // Place cars at evenly-spaced cells.
    let mut positions: Vec<u32> = (0..cars).map(|i| i * cells / cars).collect();
    let mut velocities: Vec<u32> = vec![0; cars as usize];
    let mut total_v: u64 = 0;
    let mut stopped: u64 = 0;
    let mut samples: u64 = 0;
    let mut rng_state = seed | 1;
    for _ in 0..steps {
        // Sort cars by position to compute gap easily.
        let mut order: Vec<usize> = (0..cars as usize).collect();
        order.sort_by_key(|&i| positions[i]);
        for k in 0..cars as usize {
            let i = order[k];
            let next_idx = order[(k + 1) % cars as usize];
            let gap = if next_idx == i {
                cells
            } else {
                let next_pos = if positions[next_idx] > positions[i] {
                    positions[next_idx]
                } else {
                    positions[next_idx] + cells
                };
                (next_pos - positions[i]).saturating_sub(1)
            };
            // Accelerate.
            if velocities[i] < v_max {
                velocities[i] += 1;
            }
            // Slow due to gap.
            if velocities[i] > gap {
                velocities[i] = gap;
            }
            // Random brake.
            let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            if velocities[i] > 0 && r < brake_prob {
                velocities[i] -= 1;
            }
            // Move.
            positions[i] = (positions[i] + velocities[i]) % cells;
            total_v += u64::from(velocities[i]);
            samples += 1;
            if velocities[i] == 0 {
                stopped += 1;
            }
        }
        let _ = n;
    }
    TrafficVerdict::Ok {
        jam_fraction: stopped as f64 / samples as f64,
        avg_velocity: total_v as f64 / samples as f64,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_traffic_jam_density")?;

    println!("low density: {:?}", simulate(100, 10, 5, 100, 0.1, 42));
    println!("high density: {:?}", simulate(100, 80, 5, 100, 0.1, 42));
    println!("invalid: {:?}", simulate(0, 10, 5, 100, 0.1, 42));
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
    fn high_density_more_jam() {
        let lo = simulate(100, 10, 5, 100, 0.1, 42);
        let hi = simulate(100, 80, 5, 100, 0.1, 42);
        if let (
            TrafficVerdict::Ok {
                jam_fraction: l, ..
            },
            TrafficVerdict::Ok {
                jam_fraction: h, ..
            },
        ) = (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn invalid_zero_cells() {
        assert_eq!(
            simulate(0, 10, 5, 100, 0.1, 42),
            TrafficVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_cars() {
        assert_eq!(
            simulate(100, 0, 5, 100, 0.1, 42),
            TrafficVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_cars_exceed_cells() {
        assert_eq!(
            simulate(10, 100, 5, 100, 0.1, 42),
            TrafficVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_vmax() {
        assert_eq!(
            simulate(100, 10, 0, 100, 0.1, 42),
            TrafficVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(
            simulate(100, 10, 5, 0, 0.1, 42),
            TrafficVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_brake_out_of_range() {
        assert_eq!(
            simulate(100, 10, 5, 100, 1.5, 42),
            TrafficVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 10, 5, 100, 0.1, 42);
        let b = simulate(100, 10, 5, 100, 0.1, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn jam_fraction_in_unit_range() {
        let v = simulate(100, 50, 5, 50, 0.1, 42);
        if let TrafficVerdict::Ok { jam_fraction, .. } = v {
            assert!((0.0..=1.0).contains(&jam_fraction));
        }
    }

    #[test]
    fn avg_velocity_le_vmax() {
        let v = simulate(100, 10, 5, 50, 0.1, 42);
        if let TrafficVerdict::Ok { avg_velocity, .. } = v {
            assert!(avg_velocity <= 5.0);
        }
    }

    #[test]
    fn no_brake_higher_velocity() {
        let no_brake = simulate(100, 20, 5, 100, 0.0, 42);
        let with_brake = simulate(100, 20, 5, 100, 0.5, 42);
        if let (
            TrafficVerdict::Ok {
                avg_velocity: nb, ..
            },
            TrafficVerdict::Ok {
                avg_velocity: wb, ..
            },
        ) = (no_brake, with_brake)
        {
            assert!(nb > wb);
        }
    }
}
