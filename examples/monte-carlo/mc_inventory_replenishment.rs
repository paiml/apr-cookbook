//! # Monte-Carlo (s, S) Inventory Replenishment
//!
//! Sim a simple `(s, S)` inventory policy: when stock ≤ s, reorder
//! up to S. Demand is uniform [0, 2*mean]. Returns service-level
//! (1 - stockout_rate), avg holding cost, and orders placed.
//!
//! Demonstrates the **MC.63** recipe for PMAT-180 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hadley & Whitin, Analysis of Inventory Systems (1963).
//!
//! Run with: cargo run --example mc_inventory_replenishment
//!
//! Added by PMAT-180 (catalog 1243→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum InventoryVerdict {
    Ok {
        service_level: f64,
        avg_holding: f64,
        orders_placed: u32,
    },
    InvalidConfig,
}

#[allow(clippy::too_many_arguments)]
pub fn simulate(
    days: u32,
    initial_stock: u32,
    reorder_point_s: u32,
    reorder_to_big_s: u32,
    mean_demand: u32,
    lead_time_days: u32,
    seed: u64,
) -> InventoryVerdict {
    if days == 0 || mean_demand == 0 || reorder_to_big_s <= reorder_point_s || lead_time_days == 0 {
        return InventoryVerdict::InvalidConfig;
    }
    let mut stock = i64::from(initial_stock);
    let mut total_holding: u64 = 0;
    let mut stockout_days: u32 = 0;
    let mut orders_placed: u32 = 0;
    // (arrival_day, qty)
    let mut pending: Vec<(u32, u32)> = Vec::new();
    let mut rng_state = seed | 1;
    for day in 0..days {
        // Receive any deliveries scheduled for today.
        pending.retain(|&(arr, qty)| {
            if arr <= day {
                stock += i64::from(qty);
                false
            } else {
                true
            }
        });
        // Demand draw.
        let demand = ((lcg(&mut rng_state) >> 32) as u32) % (2 * mean_demand + 1);
        if i64::from(demand) > stock {
            stockout_days += 1;
            stock = 0;
        } else {
            stock -= i64::from(demand);
        }
        // Holding cost = end-of-day stock.
        total_holding += stock.max(0) as u64;
        // Reorder check.
        let on_order: u32 = pending.iter().map(|(_, q)| *q).sum();
        if (stock + i64::from(on_order)) <= i64::from(reorder_point_s) {
            let qty = reorder_to_big_s - (stock.max(0) as u32 + on_order).min(reorder_to_big_s);
            pending.push((day + lead_time_days, qty));
            orders_placed += 1;
        }
    }
    let service_level = 1.0 - f64::from(stockout_days) / f64::from(days);
    let avg_holding = total_holding as f64 / f64::from(days);
    InventoryVerdict::Ok {
        service_level,
        avg_holding,
        orders_placed,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_inventory_replenishment")?;

    println!("balanced: {:?}", simulate(365, 100, 30, 100, 5, 7, 42));
    println!("low s: {:?}", simulate(365, 100, 5, 50, 10, 7, 42));
    println!("invalid: {:?}", simulate(0, 100, 30, 100, 5, 7, 42));
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
    fn high_initial_stock_high_service() {
        let v = simulate(365, 1000, 100, 500, 5, 7, 42);
        if let InventoryVerdict::Ok { service_level, .. } = v {
            assert!(service_level > 0.9);
        }
    }

    #[test]
    fn very_low_s_lowers_service() {
        let high = simulate(365, 100, 50, 100, 10, 7, 42);
        let low = simulate(365, 100, 1, 50, 10, 7, 42);
        if let (
            InventoryVerdict::Ok {
                service_level: h, ..
            },
            InventoryVerdict::Ok {
                service_level: l, ..
            },
        ) = (high, low)
        {
            assert!(h >= l);
        }
    }

    #[test]
    fn orders_placed_nonneg() {
        let v = simulate(100, 100, 30, 100, 5, 5, 42);
        if let InventoryVerdict::Ok { orders_placed, .. } = v {
            // u32 always nonneg; just exercise the path.
            let _ = orders_placed;
        }
    }

    #[test]
    fn invalid_zero_days() {
        assert_eq!(
            simulate(0, 100, 30, 100, 5, 7, 42),
            InventoryVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_demand() {
        assert_eq!(
            simulate(100, 100, 30, 100, 0, 7, 42),
            InventoryVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_s_geq_big_s() {
        assert_eq!(
            simulate(100, 100, 100, 100, 5, 7, 42),
            InventoryVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_lead_time() {
        assert_eq!(
            simulate(100, 100, 30, 100, 5, 0, 42),
            InventoryVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 100, 30, 100, 5, 7, 42);
        let b = simulate(100, 100, 30, 100, 5, 7, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn service_level_in_unit_range() {
        let v = simulate(100, 100, 30, 100, 5, 7, 42);
        if let InventoryVerdict::Ok { service_level, .. } = v {
            assert!((0.0..=1.0).contains(&service_level));
        }
    }

    #[test]
    fn higher_initial_stock_more_holding() {
        let lo = simulate(100, 50, 20, 60, 5, 5, 42);
        let hi = simulate(100, 500, 20, 600, 5, 5, 42);
        if let (
            InventoryVerdict::Ok { avg_holding: l, .. },
            InventoryVerdict::Ok { avg_holding: h, .. },
        ) = (lo, hi)
        {
            assert!(h > l);
        }
    }
}
