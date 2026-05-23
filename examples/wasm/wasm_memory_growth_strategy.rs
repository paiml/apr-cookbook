//! # WASM Memory Growth Strategy Picker
//!
//! WASM linear memory grows in 64 KiB pages. Picker chooses growth
//! policy based on workload:
//!   Linear: +1 page per request (frequent grow_memory; fragmentation-friendly)
//!   Doubling: ×2 capacity (amortized O(1); wastes memory)
//!   Custom (1.5×): middle ground for inference (moderate working sets)
//!
//! Demonstrates the **WASM.17** recipe for PMAT-142 (wasm round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly Memory growth semantics + V8 grow_memory cost.
//!
//! Run with: cargo run --example wasm_memory_growth_strategy
//!
//! Added by PMAT-142 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const PAGE_BYTES: u32 = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthStrategy {
    Linear,
    Custom1_5,
    Doubling,
}

#[derive(Debug, PartialEq)]
pub enum GrowthVerdict {
    Ok {
        strategy: GrowthStrategy,
        new_pages: u32,
        new_total_bytes: u64,
    },
    InvalidCurrent,
    InvalidNeeded,
    AlreadyEnoughCapacity,
}

pub fn pick(
    current_pages: u32,
    requested_extra_bytes: u64,
    workload_growth_pattern: GrowthStrategy,
) -> GrowthVerdict {
    if current_pages == 0 {
        return GrowthVerdict::InvalidCurrent;
    }
    if requested_extra_bytes == 0 {
        return GrowthVerdict::AlreadyEnoughCapacity;
    }
    let needed_extra_pages = (requested_extra_bytes.div_ceil(u64::from(PAGE_BYTES))) as u32;
    let new_pages = match workload_growth_pattern {
        GrowthStrategy::Linear => current_pages + needed_extra_pages,
        GrowthStrategy::Custom1_5 => {
            let target = (f64::from(current_pages) * 1.5) as u32;
            target.max(current_pages + needed_extra_pages)
        }
        GrowthStrategy::Doubling => {
            let target = current_pages * 2;
            target.max(current_pages + needed_extra_pages)
        }
    };
    let new_total_bytes = u64::from(new_pages) * u64::from(PAGE_BYTES);
    GrowthVerdict::Ok {
        strategy: workload_growth_pattern,
        new_pages,
        new_total_bytes,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_memory_growth_strategy")?;

    println!(
        "linear, +100KiB: {:?}",
        pick(10, 100 * 1024, GrowthStrategy::Linear)
    );
    println!(
        "1.5x, +100KiB: {:?}",
        pick(10, 100 * 1024, GrowthStrategy::Custom1_5)
    );
    println!(
        "doubling, +100KiB: {:?}",
        pick(10, 100 * 1024, GrowthStrategy::Doubling)
    );
    println!("no extra: {:?}", pick(10, 0, GrowthStrategy::Linear));
    println!("zero current: {:?}", pick(0, 1000, GrowthStrategy::Linear));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn linear_grows_only_what_needed() {
        // Need 2 pages extra; linear adds exactly 2.
        let v = pick(10, 100 * 1024, GrowthStrategy::Linear);
        if let GrowthVerdict::Ok { new_pages, .. } = v {
            assert_eq!(new_pages, 12);
        }
    }

    #[test]
    fn custom_1_5_uses_max_of_target_and_needed() {
        // 10 pages × 1.5 = 15. Need 12. Picks 15.
        let v = pick(10, 100 * 1024, GrowthStrategy::Custom1_5);
        if let GrowthVerdict::Ok { new_pages, .. } = v {
            assert_eq!(new_pages, 15);
        }
    }

    #[test]
    fn doubling_doubles_current() {
        let v = pick(10, 100 * 1024, GrowthStrategy::Doubling);
        if let GrowthVerdict::Ok { new_pages, .. } = v {
            assert_eq!(new_pages, 20);
        }
    }

    #[test]
    fn doubling_overrides_when_huge_request() {
        // Need 30 pages extra; doubling alone gives 20; pick max → 40.
        let v = pick(10, 30 * 64 * 1024, GrowthStrategy::Doubling);
        if let GrowthVerdict::Ok { new_pages, .. } = v {
            assert_eq!(new_pages, 40);
        }
    }

    #[test]
    fn zero_current_invalid() {
        assert_eq!(
            pick(0, 1000, GrowthStrategy::Linear),
            GrowthVerdict::InvalidCurrent
        );
    }

    #[test]
    fn zero_extra_already_enough() {
        assert_eq!(
            pick(10, 0, GrowthStrategy::Linear),
            GrowthVerdict::AlreadyEnoughCapacity
        );
    }

    #[test]
    fn rounded_up_to_page_boundary() {
        // 1 byte extra still needs 1 full page.
        let v = pick(10, 1, GrowthStrategy::Linear);
        if let GrowthVerdict::Ok { new_pages, .. } = v {
            assert_eq!(new_pages, 11);
        }
    }

    #[test]
    fn new_total_bytes_matches_pages() {
        if let GrowthVerdict::Ok {
            new_pages,
            new_total_bytes,
            ..
        } = pick(10, 100 * 1024, GrowthStrategy::Linear)
        {
            assert_eq!(
                new_total_bytes,
                u64::from(new_pages) * u64::from(PAGE_BYTES)
            );
        }
    }

    #[test]
    fn doubling_min_one_extra_when_request_tiny() {
        let v = pick(8, 1, GrowthStrategy::Doubling);
        if let GrowthVerdict::Ok { new_pages, .. } = v {
            assert_eq!(new_pages, 16);
        }
    }

    #[test]
    fn linear_below_doubling() {
        let lin = pick(100, 64 * 1024, GrowthStrategy::Linear);
        let dbl = pick(100, 64 * 1024, GrowthStrategy::Doubling);
        if let (GrowthVerdict::Ok { new_pages: l, .. }, GrowthVerdict::Ok { new_pages: d, .. }) =
            (lin, dbl)
        {
            assert!(d > l);
        }
    }
}
