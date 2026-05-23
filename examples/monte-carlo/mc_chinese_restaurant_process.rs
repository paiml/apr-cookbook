//! # Monte-Carlo Chinese Restaurant Process
//!
//! Sim the Chinese Restaurant Process (CRP): seat n customers at
//! tables; each new customer joins an existing table with probability
//! proportional to its size, or starts a new table with probability
//! proportional to α. Returns table sizes and table count.
//!
//! Demonstrates the **MC.152** recipe for PMAT-209 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Aldous, "Exchangeability and related topics" (1985);
//!  Pitman, Combinatorial Stochastic Processes (2006).
//!
//! Run with: cargo run --example mc_chinese_restaurant_process
//!
//! Added by PMAT-209 (catalog 1504→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CrpVerdict {
    Ok {
        table_count: u32,
        largest_table: u32,
    },
    InvalidConfig,
}

pub fn simulate(n_customers: u32, alpha_x100: u32, seed: u64) -> CrpVerdict {
    if n_customers == 0 || alpha_x100 == 0 {
        return CrpVerdict::InvalidConfig;
    }
    let alpha = alpha_x100 as f64 / 100.0;
    let mut state = seed | 1;
    let mut tables: Vec<u32> = vec![1];
    for i in 1..n_customers {
        let total = i as f64 + alpha;
        let r = (lcg(&mut state) as f64) / (u32::MAX as f64) * total;
        let mut cum = 0.0f64;
        let mut chose_existing = false;
        for t in &mut tables {
            cum += *t as f64;
            if r < cum {
                *t += 1;
                chose_existing = true;
                break;
            }
        }
        if !chose_existing {
            tables.push(1);
        }
    }
    let largest = *tables.iter().max().unwrap_or(&0);
    CrpVerdict::Ok {
        table_count: tables.len() as u32,
        largest_table: largest,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_chinese_restaurant_process")?;

    println!("alpha=1: {:?}", simulate(100, 100, 42));
    println!("alpha=5: {:?}", simulate(100, 500, 42));
    println!("invalid: {:?}", simulate(0, 100, 42));
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
    fn invalid_zero_customers() {
        assert_eq!(simulate(0, 100, 42), CrpVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_alpha() {
        assert_eq!(simulate(10, 0, 42), CrpVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(50, 100, 42);
        let b = simulate(50, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn first_customer_one_table() {
        let v = simulate(1, 100, 42);
        if let CrpVerdict::Ok { table_count, .. } = v {
            assert_eq!(table_count, 1);
        }
    }

    #[test]
    fn table_count_le_customers() {
        let v = simulate(50, 100, 42);
        if let CrpVerdict::Ok { table_count, .. } = v {
            assert!(table_count <= 50);
        }
    }

    #[test]
    fn larger_alpha_more_tables() {
        let small = simulate(100, 50, 42);
        let large = simulate(100, 1000, 42);
        if let (CrpVerdict::Ok { table_count: s, .. }, CrpVerdict::Ok { table_count: l, .. }) =
            (small, large)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn largest_table_at_least_one() {
        let v = simulate(10, 100, 42);
        if let CrpVerdict::Ok { largest_table, .. } = v {
            assert!(largest_table >= 1);
        }
    }

    #[test]
    fn largest_table_le_customers() {
        let v = simulate(20, 100, 42);
        if let CrpVerdict::Ok { largest_table, .. } = v {
            assert!(largest_table <= 20);
        }
    }

    #[test]
    fn many_customers_handled() {
        let v = simulate(1000, 100, 42);
        assert!(matches!(v, CrpVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcome() {
        let a = simulate(100, 100, 42);
        let b = simulate(100, 100, 999);
        assert!(a != b);
    }

    #[test]
    fn small_alpha_few_tables() {
        // alpha=0.1 → most customers join existing tables.
        let v = simulate(100, 10, 42);
        if let CrpVerdict::Ok { table_count, .. } = v {
            assert!(table_count < 30);
        }
    }
}
