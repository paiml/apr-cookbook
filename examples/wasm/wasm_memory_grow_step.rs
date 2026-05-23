//! # WASM Memory Grow Step Budget
//!
//! Validate `memory.grow` requests against page budget. Each WASM
//! page is 64 KiB; grow requests must not exceed `max_pages`.
//! Returns categorical verdict.
//!
//! Demonstrates the **WASM.X** recipe for PMAT-218 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly Core §4.5.5 memory.grow trap conditions;
//!  V8 wasm-engine memory-grow heuristics.
//!
//! Run with: cargo run --example wasm_memory_grow_step
//!
//! Added by PMAT-218 (catalog 1585→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GrowVerdict {
    Ok { new_pages: u32, kib_added: u64 },
    OverBudget { requested: u32, available: u32 },
    InvalidConfig,
}

pub fn check(current_pages: u32, delta_pages: u32, max_pages: u32) -> GrowVerdict {
    if max_pages == 0 || max_pages > 65_536 {
        return GrowVerdict::InvalidConfig;
    }
    if current_pages > max_pages {
        return GrowVerdict::InvalidConfig;
    }
    let new_pages = current_pages.saturating_add(delta_pages);
    if new_pages > max_pages {
        return GrowVerdict::OverBudget {
            requested: new_pages,
            available: max_pages - current_pages,
        };
    }
    GrowVerdict::Ok {
        new_pages,
        kib_added: delta_pages as u64 * 64,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_memory_grow_step")?;

    println!("ok: {:?}", check(10, 5, 100));
    println!("over: {:?}", check(95, 10, 100));
    println!("invalid: {:?}", check(10, 5, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn within_budget_ok() {
        let v = check(10, 5, 100);
        assert_eq!(
            v,
            GrowVerdict::Ok {
                new_pages: 15,
                kib_added: 320,
            }
        );
    }

    #[test]
    fn over_budget_rejected() {
        let v = check(95, 10, 100);
        if let GrowVerdict::OverBudget { available, .. } = v {
            assert_eq!(available, 5);
        }
    }

    #[test]
    fn at_max_pages_ok() {
        let v = check(95, 5, 100);
        assert!(matches!(v, GrowVerdict::Ok { .. }));
    }

    #[test]
    fn invalid_zero_max() {
        assert_eq!(check(10, 5, 0), GrowVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_max_over_64ki() {
        assert_eq!(check(10, 5, 65_537), GrowVerdict::InvalidConfig);
    }

    #[test]
    fn current_over_max_rejected() {
        assert_eq!(check(150, 5, 100), GrowVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = check(10, 5, 100);
        let r2 = check(10, 5, 100);
        assert_eq!(r1, r2);
    }

    #[test]
    fn zero_delta_no_change() {
        let v = check(10, 0, 100);
        if let GrowVerdict::Ok { kib_added, .. } = v {
            assert_eq!(kib_added, 0);
        }
    }

    #[test]
    fn kib_added_correct() {
        // 1 page = 64 KiB
        let v = check(0, 1, 100);
        if let GrowVerdict::Ok { kib_added, .. } = v {
            assert_eq!(kib_added, 64);
        }
    }

    #[test]
    fn requested_in_overbudget_includes_total() {
        let v = check(95, 100, 100);
        if let GrowVerdict::OverBudget { requested, .. } = v {
            assert_eq!(requested, 195);
        }
    }

    #[test]
    fn max_64ki_pages_accepted() {
        let v = check(0, 1, 65_536);
        assert!(matches!(v, GrowVerdict::Ok { .. }));
    }

    #[test]
    fn full_max_grow_handled() {
        let v = check(0, 65_536, 65_536);
        if let GrowVerdict::Ok { new_pages, .. } = v {
            assert_eq!(new_pages, 65_536);
        }
    }
}
