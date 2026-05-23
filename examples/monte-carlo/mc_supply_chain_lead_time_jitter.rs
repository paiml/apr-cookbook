//! # Monte-Carlo Supply Chain Lead Time Jitter
//!
//! Sim N-stage supply chain where each stage has its own jittered
//! delay. Returns per-shipment total lead-time, plus mean and p95.
//!
//! Demonstrates the **MC.71** recipe for PMAT-182 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Forrester, Industrial Dynamics (1961); bullwhip effect.
//!
//! Run with: cargo run --example mc_supply_chain_lead_time_jitter
//!
//! Added by PMAT-182 (catalog 1261→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SupplyVerdict {
    Ok {
        mean_lead_time: f64,
        p95_lead_time: u32,
        max_lead_time: u32,
    },
    InvalidConfig,
}

pub fn simulate(shipments: u32, stage_avg_delays: &[u32], seed: u64) -> SupplyVerdict {
    if shipments == 0 || stage_avg_delays.is_empty() {
        return SupplyVerdict::InvalidConfig;
    }
    if stage_avg_delays.contains(&0) {
        return SupplyVerdict::InvalidConfig;
    }
    let mut leads: Vec<u32> = Vec::with_capacity(shipments as usize);
    let mut rng_state = seed | 1;
    for _ in 0..shipments {
        let mut total: u32 = 0;
        for &avg in stage_avg_delays {
            // Jitter: uniform [avg/2, 3*avg/2]
            let jitter = avg / 2 + ((lcg(&mut rng_state) >> 32) as u32) % avg.max(1);
            total += jitter;
        }
        leads.push(total);
    }
    leads.sort_unstable();
    let total: u64 = leads.iter().map(|l| u64::from(*l)).sum();
    let mean_lead_time = total as f64 / f64::from(shipments);
    let p95_idx = (leads.len() as f64 * 0.95) as usize;
    let p95_lead_time = leads[p95_idx.min(leads.len() - 1)];
    let max_lead_time = *leads.last().unwrap_or(&0);
    SupplyVerdict::Ok {
        mean_lead_time,
        p95_lead_time,
        max_lead_time,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_supply_chain_lead_time_jitter")?;

    let stages = [3u32, 5, 7, 10];
    println!("4-stage: {:?}", simulate(1000, &stages, 42));
    println!("invalid: {:?}", simulate(0, &stages, 42));
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
    fn mean_le_p95() {
        let stages = [3u32, 5, 7];
        let v = simulate(1000, &stages, 42);
        if let SupplyVerdict::Ok {
            mean_lead_time,
            p95_lead_time,
            ..
        } = v
        {
            assert!(mean_lead_time <= f64::from(p95_lead_time));
        }
    }

    #[test]
    fn p95_le_max() {
        let stages = [3u32, 5, 7];
        let v = simulate(1000, &stages, 42);
        if let SupplyVerdict::Ok {
            p95_lead_time,
            max_lead_time,
            ..
        } = v
        {
            assert!(p95_lead_time <= max_lead_time);
        }
    }

    #[test]
    fn invalid_zero_shipments() {
        let stages = [3u32];
        assert_eq!(simulate(0, &stages, 42), SupplyVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_empty_stages() {
        assert_eq!(simulate(100, &[], 42), SupplyVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_stage_delay() {
        let stages = [3u32, 0, 5];
        assert_eq!(simulate(100, &stages, 42), SupplyVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let stages = [3u32, 5];
        let a = simulate(500, &stages, 42);
        let b = simulate(500, &stages, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn longer_chain_more_lead_time() {
        let short = [5u32];
        let long = [5u32, 5, 5, 5, 5];
        let s = simulate(1000, &short, 42);
        let l = simulate(1000, &long, 42);
        if let (
            SupplyVerdict::Ok {
                mean_lead_time: s_m,
                ..
            },
            SupplyVerdict::Ok {
                mean_lead_time: l_m,
                ..
            },
        ) = (s, l)
        {
            assert!(l_m > s_m);
        }
    }

    #[test]
    fn higher_stage_delays_higher_mean() {
        let lo = [1u32, 1];
        let hi = [10u32, 10];
        let l = simulate(1000, &lo, 42);
        let h = simulate(1000, &hi, 42);
        if let (
            SupplyVerdict::Ok {
                mean_lead_time: lm, ..
            },
            SupplyVerdict::Ok {
                mean_lead_time: hm, ..
            },
        ) = (l, h)
        {
            assert!(hm > lm);
        }
    }

    #[test]
    fn single_shipment_works() {
        let stages = [3u32];
        let v = simulate(1, &stages, 42);
        if let SupplyVerdict::Ok { mean_lead_time, .. } = v {
            assert!(mean_lead_time >= 1.0);
        }
    }

    #[test]
    fn single_stage_lead_time_in_range() {
        let stages = [10u32];
        let v = simulate(10_000, &stages, 42);
        if let SupplyVerdict::Ok { mean_lead_time, .. } = v {
            // [avg/2, 3*avg/2] → mean ≈ avg.
            assert!(mean_lead_time >= 5.0 && mean_lead_time <= 15.0);
        }
    }

    #[test]
    fn max_le_3half_sum() {
        let stages = [4u32, 6];
        let v = simulate(100, &stages, 42);
        if let SupplyVerdict::Ok { max_lead_time, .. } = v {
            // Each stage max ≈ 3*avg/2 → total max ≤ 3*sum/2 = 15.
            assert!(max_lead_time <= 15);
        }
    }
}
