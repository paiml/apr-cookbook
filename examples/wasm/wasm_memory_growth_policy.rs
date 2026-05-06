//! # WASM Memory Growth Policy
//!
//! WebAssembly linear memory grows in 64-KiB pages. Browsers cap the
//! limit (Chromium 4 GiB on 64-bit, 1 GiB on 32-bit; Safari ~2 GiB).
//! Growth strategies: 2× doubling (fast, fragments), fixed-step
//! (predictable, slow), reservation (over-commit upfront). This recipe
//! builds the policy picker + page-budget validator.
//!
//! Demonstrates the **WASM.6** recipe for PMAT-123 (wasm coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly Core Spec §4.2.8 (Memory Instances)
//!
//! Run with: cargo run --example wasm_memory_growth_policy
//!
//! Added by PMAT-123 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const PAGE_BYTES: u32 = 65_536;
const MAX_PAGES_32BIT: u32 = 16_384; // 1 GiB
const MAX_PAGES_64BIT: u32 = 65_536; // 4 GiB

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthStrategy {
    Doubling,
    FixedStep,
    Reservation,
}

#[derive(Debug, PartialEq)]
pub enum PolicyVerdict {
    Ok,
    ExceedsBrowserCap { requested: u32, max: u32 },
    InvalidInitialPages,
}

pub fn validate_pages(initial: u32, max: u32, is_64bit: bool) -> PolicyVerdict {
    if initial == 0 {
        return PolicyVerdict::InvalidInitialPages;
    }
    let cap = if is_64bit {
        MAX_PAGES_64BIT
    } else {
        MAX_PAGES_32BIT
    };
    if max > cap {
        return PolicyVerdict::ExceedsBrowserCap {
            requested: max,
            max: cap,
        };
    }
    if initial > max {
        return PolicyVerdict::InvalidInitialPages;
    }
    PolicyVerdict::Ok
}

pub fn next_size(current_pages: u32, strategy: GrowthStrategy, fixed_step: u32) -> u32 {
    match strategy {
        GrowthStrategy::Doubling => current_pages.saturating_mul(2).max(1),
        GrowthStrategy::FixedStep => current_pages.saturating_add(fixed_step.max(1)),
        GrowthStrategy::Reservation => current_pages, // reservation pre-allocates; no growth
    }
}

pub fn pages_to_bytes(pages: u32) -> u64 {
    u64::from(pages) * u64::from(PAGE_BYTES)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_memory_growth_policy")?;

    println!("16 pages → {} bytes", pages_to_bytes(16));
    println!(
        "validate(16, 256, 32bit): {:?}",
        validate_pages(16, 256, false)
    );
    println!(
        "validate(16, 65536, 32bit): {:?}",
        validate_pages(16, 65_536, false)
    );
    for s in [
        GrowthStrategy::Doubling,
        GrowthStrategy::FixedStep,
        GrowthStrategy::Reservation,
    ] {
        println!("next_size(16, {s:?}, 4) = {}", next_size(16, s, 4));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_initial_within_cap() {
        assert_eq!(validate_pages(16, 256, false), PolicyVerdict::Ok);
    }

    #[test]
    fn zero_initial_invalid() {
        assert_eq!(
            validate_pages(0, 256, false),
            PolicyVerdict::InvalidInitialPages
        );
    }

    #[test]
    fn initial_above_max_invalid() {
        assert_eq!(
            validate_pages(500, 256, false),
            PolicyVerdict::InvalidInitialPages
        );
    }

    #[test]
    fn exceeds_32bit_cap_rejected() {
        let v = validate_pages(16, MAX_PAGES_32BIT + 1, false);
        assert!(matches!(v, PolicyVerdict::ExceedsBrowserCap { .. }));
    }

    #[test]
    fn at_64bit_cap_passes() {
        assert_eq!(validate_pages(16, MAX_PAGES_64BIT, true), PolicyVerdict::Ok);
    }

    #[test]
    fn pages_to_bytes_64kib_per_page() {
        assert_eq!(pages_to_bytes(1), 65_536);
        assert_eq!(pages_to_bytes(16), 16 * 65_536);
    }

    #[test]
    fn doubling_strategy_2x() {
        assert_eq!(next_size(16, GrowthStrategy::Doubling, 0), 32);
    }

    #[test]
    fn fixed_step_adds_increment() {
        assert_eq!(next_size(16, GrowthStrategy::FixedStep, 4), 20);
    }

    #[test]
    fn reservation_does_not_grow() {
        assert_eq!(next_size(16, GrowthStrategy::Reservation, 999), 16);
    }

    #[test]
    fn doubling_saturates_at_max() {
        // Doesn't panic on overflow.
        let v = next_size(u32::MAX, GrowthStrategy::Doubling, 0);
        assert_eq!(v, u32::MAX);
    }

    #[test]
    fn fixed_step_zero_clamps_to_one() {
        // Don't get stuck at the same size — clamp to ≥ 1.
        assert_eq!(next_size(16, GrowthStrategy::FixedStep, 0), 17);
    }
}
